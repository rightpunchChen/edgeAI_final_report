# Llama-3.2-3B-Instruct LoRA + GPTQ Compression & Inference with vLLM

This project demonstrates how to fine-tune a LLaMA-3 model using LoRA (Low-Rank Adaptation), merge the adapter back into the base model, apply GPTQ (4-bit quantization) for compression, and finally perform efficient inference using vLLM.

Huggingface model page: rightpunch/Llama-3.2-3B-Instruct_PEFT_GPTQ

## 📌 Overview

Pipeline Summary:
1. Fine-tune a LLaMA-3 model using LoRA on WikiText-2.
2. Merge LoRA weights into the base model.
3. Quantize the merged model using GPTQ (4-bit).
4. Load and infer with the quantized model via `vLLM`.
5. Evaluate inference throughput and perplexity (PPL).

## 🚀 Usage and Reproduce

To set up the environment and reproduce the experiment, follow the steps below.

These instructions assume you are using a GPU-enabled system with at least 16GB of VRAM and CUDA support.

```bash
git clone https://github.com/rightpunchChen/edgeAI_final_report.git
conda create -n llama_env python=3.10
conda activate llama_env
cd edgeAI_final_report
pip install -r requirements.txt
huggingface-cli login
python main.py
```
If the installation fails using requirement.txt, you can install the relevant dependencies through the following steps.
```bash
pip install huggingface-hub[cli]
pip install transformers==4.51.1
pip install torch==2.6.0
pip install torchvision==0.21.0
pip install torchaudio==2.6.0
pip install triton==3.2.0
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


## 🚀 Training
If you want to train the new model run the full training pipeline:
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

## 🛠 Custom Paths
You can change the output directories for your models by modifying the following variables in the `main()` function in `lora_gptq/train_lora_gptq_vllm.py`:

```python
peft_model_path = "your_custom_peft_path"
merged_path = "./your_custom_merged_path"
gptq_path = './your_custom_gptq_path'
```
This allows you to organize multiple experiments or avoid overwriting results.

## 📊 Experimental Results
All experiments were performed on a T4 gpu.
```text
Prompt: How to learn a new language?
Response: Learning a new language can be a challenging but rewarding experience. Here are some steps you can take to learn a new language: 
1. Set your goals : Decide what you want to achieve with your language learning. Are you looking to travel to a foreign country, communicate with a foreign family member, or simply improve your language skills ?
2. Choose your learning method: There are many ways to learn a new language, including
* Language classes : Enroll in a class at a language school or community college
* Language exchange programs: Find a language partner to practice with
* Language learning apps : Use apps like Duolingo, Babbel, or Rosetta Stone
* Language learning software : Use software like Rosetta Stone or Pimseler
* Language learning books : Use books like " Language Hacking " or " Fluent Forever "
* Language learning podcasts : Listen to podcasts like " Coffee Break " or " News in Slow "
* Language learning YouTube channels : Watch YouTube channels like " EnglishClass101 " or
" English With Lucy "
• Learn the basics : Start with the basics of the language, such as the alphabet, basic grammar rules, and common phrases .
4 . Practice regularly : Practice speaking


Throughput: 87.0 toks/s
Perplexity (PPL): 9.75
```

CSV format:
```csv
Id,value
0,9.75       # PPL
1,87.0       # Throughput
```
