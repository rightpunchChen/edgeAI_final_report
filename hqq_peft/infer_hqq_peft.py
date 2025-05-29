import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HqqConfig
)
from datasets import load_dataset
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

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

device = 'cuda:0'
model_id = "meta-llama/Llama-3.2-3B-Instruct"
peft_id = "HQQ-model/checkpoint-1719"

quant_config = HqqConfig(dynamic_config={
        'self_attn.q_proj':{'nbits':2, 'group_size':16},
        'self_attn.k_proj':{'nbits':2, 'group_size':16},
        'self_attn.v_proj':{'nbits':2, 'group_size':8},
        'self_attn.o_proj':{'nbits':2, 'group_size':8},
        'mlp.gate_proj':{'nbits':4, 'group_size':128},
        'mlp.up_proj'  :{'nbits':4, 'group_size':128},
        'mlp.down_proj':{'nbits':4, 'group_size':128},
        })

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device,
    quantization_config=quant_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, peft_id)
model = model.merge_and_unload()

model = AutoModelForCausalLM.from_pretrained(
    peft_id,
    torch_dtype=torch.float16,
    device_map=device,
    quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained(peft_id)

print(model)

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
    # tput = max_new_tokens / (elapsed_ms / 1000)
    tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
    time_record.append(elapsed_ms / 1000)
    tputs.append(tput)
    
response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
sorted_tputs = np.sort(tputs)[2:-2]
org_tput = np.mean(sorted_tputs)
print(f'Prompt: {prompt}\nResponse: {response}\n')

print(f'Time Record: {time_record}')
print(f'Throughput Record: {tputs} toks/s\n')

## Your final throughput result ###
print(f'Throughput: {org_tput} toks/s')
ppl = evaluate_ppl(model, tokenizer, device)
print(f"Perplexity (PPL): {ppl}")

# Save results to CSV
import csv
rounded_tput = round(org_tput, 1)
ppl = round(ppl, 2)

with open("result2.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "value"])
    writer.writerow([0, ppl])
    writer.writerow([1, rounded_tput])