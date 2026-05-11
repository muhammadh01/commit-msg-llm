"""LoRA fine-tuning of Qwen2.5-1.5B on commit messages. Local sanity-check version."""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---------- Config ----------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
DATA_DIR = Path("data/processed")
OUTPUT_DIR = "training/checkpoints/sanity"
N_EXAMPLES = 20   # tiny: just to prove training works
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ---------- Load tokenizer + model ----------
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

# ---------- LoRA config ----------
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # attend layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------- Load + format data ----------
def load_jsonl(p):
    return [json.loads(l) for l in p.open()]

raw = load_jsonl(DATA_DIR / "train.jsonl")[:N_EXAMPLES]

def to_prompt(ex):
    """Wrap into a clear instruction format."""
    text = (
        "Write a concise git commit message for the following diff.\n\n"
        f"### Diff:\n{ex['input']}\n\n"
        f"### Commit message:\n{ex['output']}"
    )
    return {"text": text}

ds = Dataset.from_list([to_prompt(e) for e in raw])

# ---------- Train ----------
cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=2,
    save_strategy="no",
    report_to="none",
    max_length=1024,
    bf16=False,
    fp16=False,
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
)

print("Training...")
trainer.train()
print("Saving adapter...")
trainer.save_model(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
