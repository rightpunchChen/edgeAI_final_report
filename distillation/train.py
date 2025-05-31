import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
STUDENT_MODEL = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
TEMPERATURE = 2.0

prompts = [
    "Explain what AI is.",
    "How to learn a new language?",
]
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# texts = [item["text"] for item in dataset]
# texts = texts[:1000]

class DistillDataset(Dataset):
    def __init__(self, prompts, tokenizer, teacher_model):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model.eval()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "teacher_logits": teacher_logits.squeeze(0)
        }

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


dataset = DistillDataset(prompts, tokenizer, teacher_model)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
student_model.train()

for epoch in range(5):
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        teacher_logits = batch["teacher_logits"].to(DEVICE)

        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits

        student_log_probs = torch.nn.functional.log_softmax(student_logits / TEMPERATURE, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / TEMPERATURE, dim=-1)

        # loss = torch.nn.functional.kl_div(
        #     student_log_probs, teacher_probs,
        #     reduction='batchmean'
        # ) * (TEMPERATURE ** 2)
        

        loss_mask = attention_mask.float()
        kl = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none"
        ).sum(dim=-1)
        masked_kl = (kl * loss_mask).sum() / loss_mask.sum()
        loss = masked_kl * (TEMPERATURE ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

student_model.save_pretrained("llama1b_lora_distilled2")
tokenizer.save_pretrained("llama1b_lora_distilled2")