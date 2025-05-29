import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

######################################################################################
import torch
cache_path    = '' 
model_id      = "meta-llama/Llama-3.2-3B-Instruct"
compute_dtype = torch.float16
device        = 'cuda:0'

#HQQ Quantize
######################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=2, group_size=8, quant_scale=False, quant_zero=False, axis=0)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Add Peft
######################################################################################
from hqq.core.peft import PeftUtils

train_dtype       = torch.torch.float32
atten_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':32, 'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}
mlp_lora_params   = {'lora_type':'default', 'r':8,  'lora_alpha':8,  'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}

lora_params       = {'self_attn.q_proj': atten_lora_params,
                     'self_attn.k_proj': atten_lora_params,
                     'self_attn.v_proj': atten_lora_params,
                     'self_attn.o_proj': atten_lora_params,
                     'mlp.gate_proj'   : mlp_lora_params,
                     'mlp.up_proj'     : mlp_lora_params,
                     'mlp.down_proj'   : mlp_lora_params}
#Apply LoRA
PeftUtils.add_lora(model, lora_params)
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
model.config.use_cache = False

#Dataset 
######################################################################################
from datasets import load_dataset, Dataset
from tqdm import tqdm
import transformers
import numpy as np 
import random

tokenizer.pad_token     = tokenizer.eos_token 
tokenizer.padding_side  = "right" 
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#####################################################################################
#Train
from trl import SFTTrainer

#Play with these parameters 
grad_acc   = 4
logging_st = 1
max_steps  = -1
lr         = 1e-4 
batch_size = 2
n_epochs   = 1
max_tokens = 1024 

training_args = transformers.TrainingArguments(	
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    learning_rate=lr,
    logging_steps=logging_st,
    num_train_epochs=n_epochs,
    max_steps=max_steps,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    save_steps=10000000,
    warmup_ratio=0.1,
    lr_scheduler_type= "cosine", 
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    max_seq_length=max_tokens,
    train_dataset=dataset,
    eval_dataset=None,
    peft_config=None,
    args=training_args,
    dataset_text_field="text",
    packing=True,
)

model.is_parallelizable       = False
trainer.is_model_parallel     = False
trainer.place_model_on_device = False
model.train()
trainer.train()

# #Prediction/Eval
# ######################################################################################
from datasets import load_dataset
import torch, time
import numpy as np
from tqdm import tqdm
import gc

tokenizer.add_bos_token = True
tokenizer.add_eos_token = False
PeftUtils.cast_lora_weights(model, dtype=compute_dtype)
model.eval()

#Save lora weights
PeftUtils.save_lora_weights(model, 'hqq_peft.pt')

PeftUtils.load_lora_weights(model, 'hqq_peft.pt')

def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

max_new_tokens = 256
# === (Optional) Uncomment the following lines if using the custom generate() function. ===
model.prefill_forward = model.forward
model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)


warmup_prompt = "Explain what AI is."
inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# === (Optional) Set up StaticCache for manual KV cache management ===
from transformers import StaticCache
past_key_values = StaticCache(
    config=model.config, 
    max_batch_size=1, 
    max_cache_len=max_new_tokens + 16, 
    device=model.device, 
    dtype=torch.float16
)

for i in tqdm(range(5), desc="Warm Up..."):
    generated = generate(model, input_ids, past_key_values, max_new_tokens)
    past_key_values.reset()
    
prompt = "How to learn a new language?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
tputs = []
time_record = []
for _ in tqdm(range(10), desc="Test Inference"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated = generate(model, input_ids, past_key_values, max_new_tokens)
    past_key_values.reset()

    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    tput = max_new_tokens / (elapsed_ms / 1000)
    time_record.append(elapsed_ms / 1000)
    tputs.append(tput)
    
response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
sorted_tputs = np.sort(tputs)[2:-2]
org_tput = np.mean(sorted_tputs)
print(f'Prompt: {prompt}\nResponse: {response}\n')

print(f'Time Record: {time_record}')
print(f'Throughput Record: {tputs} toks/s\n')

### Your final throughput result ###
print(f'Throughput: {org_tput} toks/s')
ppl = evaluate_ppl(model, tokenizer, device)
print(f"Perplexity (PPL): {ppl}")

# Save results to CSV
import csv
rounded_tput = round(org_tput, 1)
ppl = round(ppl, 2)

with open("result.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "value"])
    writer.writerow([0, ppl])
    writer.writerow([1, rounded_tput])