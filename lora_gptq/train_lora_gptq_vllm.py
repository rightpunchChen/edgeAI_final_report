import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GPTQConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from peft import (
    PeftModel, PeftConfig, get_peft_model, LoraConfig,
    TaskType, prepare_model_for_kbit_training
)
from vllm import LLM, SamplingParams

def prepare_gptq_model(model_name, gptq_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = GPTQConfig(
        bits=4,
        dataset="wikitext2",
        tokenizer=tokenizer
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        quantization_config=quantization_config)
    model.save_pretrained(gptq_path)
    tokenizer.save_pretrained(gptq_path)

def train_lora_model(base_model_name, output_dir, device):
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, LoraConfig, TaskType

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, config)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenized = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir=output_dir,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=200
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=args,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(output_dir)

def generate_vllm(engine, prompt: str, sampling_params: SamplingParams):
    """Generate text with vLLM and return the RequestOutput object."""
    outputs = engine.generate([prompt], sampling_params=sampling_params)
    return outputs[0]


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

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    peft_model_path = "lora-ckpt-bs6"
    merged_path = "./lora-quant/merged"
    gptq_path = './lora-quant/gptq-4bit_model'
    
    ### === TODO: Load your model (you may change this part) ===
    if not os.path.exists(peft_model_path):
        train_lora_model(model_name, peft_model_path, device)
    
    if not os.path.exists(merged_path):
        print("load peft model")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        peft_tokenizer = AutoTokenizer.from_pretrained(model_name)
        peft_tokenizer.save_pretrained(merged_path)
    
    print("load gptq model")
    if not os.path.exists(gptq_path):
        prepare_gptq_model(merged_path, gptq_path, device)
    
    gptq_model = AutoModelForCausalLM.from_pretrained(
        gptq_path,
        device_map=device
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # --- vLLM engine ---
    print("vLLM")
    llm = LLM(
        model=gptq_path,
        tokenizer=gptq_path,
        dtype="float16",
        trust_remote_code=True,
        quantization="gptq",
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    torch.cuda.empty_cache()
    #####################################
    
    warmup_prompt = "Explain what AI is."
    for _ in tqdm(range(5), desc="Warm Up..."):
        _ = llm.generate([warmup_prompt], sampling_params=sampling_params)
    torch.cuda.empty_cache()
        
    prompt = "How to learn a new language?"

    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        generated_ids = outputs[0].outputs[0].token_ids

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = len(generated_ids) / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    
    response = outputs[0].outputs[0].text
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    
    ppl = evaluate_ppl(gptq_model, tokenizer, device)
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
        
if __name__ == '__main__':
    main()