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
    merged_path = "./HQQ-model3/merged"
    
    ### === TODO: Load your model (you may change this part) ===
    
    base_model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        device_map=device
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    # --- vLLM engine ---
    from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
    set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)
    
    print("vLLM")
    llm = LLM(
        model=merged_path,
        tokenizer=model_name,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens
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
    
    ppl = evaluate_ppl(base_model, tokenizer, device)
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