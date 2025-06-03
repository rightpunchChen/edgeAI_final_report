# Llama-3.2-3B-Instruct LoRA + GPTQ Compression & Inference with vLLM

This project demonstrates how to fine-tune a LLaMA-3 model using LoRA (Low-Rank Adaptation), merge the adapter back into the base model, apply GPTQ (4-bit quantization) for compression, and finally perform efficient inference using vLLM.

## 📌 Overview

Pipeline Summary:
1. Fine-tune a LLaMA-3 model using LoRA on WikiText-2.
2. Merge LoRA weights into the base model.
3. Quantize the merged model using GPTQ (4-bit).
4. Load and infer with the quantized model via `vLLM`.
5. Evaluate inference throughput and perplexity (PPL).

## 🔧 Installation

Make sure you have the following dependencies:

`Python 3.10`
```bash
pip install huggingface-hub[cli]
pip install transformers==4.51.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 triton==3.2.0
pip install timm==1.0.15
pip install datasets==3.5.0
pip install accelerate==1.6.0
pip install gemlite==0.4.4
pip install hqq==0.2.5
pip install vllm
pip install optimum
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
pip install peft
```
Note: Ensure your environment supports CUDA and has the required GPU memory (16GB+ recommended).

## 🚀 Usage
Run the full pipeline:
```bash
    cd lora_gptq
    python train_lora_gptq_vllm.py
```
This will:
- Fine-tune a LLaMA-3 model using LoRA (if not already trained).
- Merge the LoRA adapter and quantize the model using GPTQ.
- Launch vLLM to generate responses and benchmark performance.
- Print throughput and perplexity.
- Save metrics into result.csv.

## 📊 Output Example

```text
Prompt: How to learn a new language?
Response: [Generated response here...]

Throughput: 86.5 toks/s
Perplexity (PPL): 9.75
```

CSV format:
```csv
Id,value
0,9.75       # PPL
1,86.5       # Throughput
```