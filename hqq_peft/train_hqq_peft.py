import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HqqConfig
)
from trl import SFTTrainer, SFTConfig
import torch, multiprocessing

device = 'cuda:0'
model_id = "meta-llama/Llama-3.2-3B-Instruct"
quant_config  = HqqConfig(dynamic_config={
        'self_attn.q_proj':{'nbits':2, 'group_size':8},
        'self_attn.k_proj':{'nbits':2, 'group_size':8},
        'self_attn.v_proj':{'nbits':2, 'group_size':8},
        'self_attn.o_proj':{'nbits':2, 'group_size':8},
        'mlp.gate_proj':{'nbits':4, 'group_size':128},
        'mlp.up_proj'  :{'nbits':4, 'group_size':128},
        'mlp.down_proj':{'nbits':4, 'group_size':128},
        })

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device,
    quantization_config=quant_config
)

model = prepare_model_for_kbit_training(model)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def process(row):
    row["text"] = row["text"]+"<|end_of_text|>"
    return row

ds = dataset.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

training_arguments = SFTConfig(
        output_dir="./HQQ-model2",
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()
model.save_pretrained('HQQ-model2_pre')