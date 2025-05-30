import torch, multiprocessing
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
STUDENT_MODEL = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
TEMPERATURE = 2.0

prompts = [
    "Explain what AI is.",
    "How to learn a new language?",
]

def process(row):
    row["text"] = row["text"]+"<|end_of_text|>"
    return row

ds = dataset.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    torch_dtype=torch.float16
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
student_model = get_peft_model(student_model, lora_cfg).to(DEVICE)


optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
student_model.train()

student_model.save_pretrained("llama1b_lora_distilled")
tokenizer.save_pretrained("llama1b_lora_distilled")