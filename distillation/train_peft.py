import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch, multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    HqqConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments
    )
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_OUTPUT = "llama1b_lora_distilled_peft2"
# TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TEACHER_MODEL = "lora-ckpt-bs6"
STUDENT_MODEL = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
TEMPERATURE = 2.0

class DistillationTrainingArguments(SFTConfig):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            teacher_logits = outputs_teacher.logits

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def process(row):
    row["text"] = row["text"]+"<|end_of_text|>"
    return row

ds = dataset.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE
)

quant_config  = HqqConfig(dynamic_config={
        'self_attn.q_proj':{'nbits':2, 'group_size':64},
        'self_attn.k_proj':{'nbits':2, 'group_size':64},
        'self_attn.v_proj':{'nbits':4, 'group_size':64},
        'self_attn.o_proj':{'nbits':4, 'group_size':64},
        'mlp.gate_proj':{'nbits':4, 'group_size':64},
        'mlp.up_proj'  :{'nbits':4, 'group_size':64},
        'mlp.down_proj':{'nbits':4, 'group_size':64},
        })
# quant_config = HqqConfig(nbits=4, group_size=64, quant_zero=False, axis=1)

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    quantization_config=quant_config
)
student_model = prepare_model_for_kbit_training(student_model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
# student_model = get_peft_model(student_model, lora_cfg).to(DEVICE)

training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    do_eval=True,
    save_strategy = "steps",
    save_steps=250,
    eval_strategy = "steps",
    eval_steps=250,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=8,
    save_total_limit=3,  # Set to zero to avoid saving
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    learning_rate=5e-5,
    logging_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    alpha=0.5,
    temperature=TEMPERATURE,
)


trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        teacher_model=teacher_model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        dataset_text_field="text",
        peft_config=lora_cfg,
        max_seq_length=512,
        tokenizer=tokenizer,
    )


trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)