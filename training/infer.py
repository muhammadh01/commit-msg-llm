"""Test the trained adapter on a validation example."""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B"
ADAPTER = "training/checkpoints/sanity"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading base model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER).to(device)
model.eval()

# Take one val example
ex = json.loads(Path("data/processed/val.jsonl").open().readline())
prompt = (
    "Write a concise git commit message for the following diff.\n\n"
    f"### Diff:\n{ex['input']}\n\n"
    "### Commit message:\n"
)

inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
print("\nGenerating...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n" + "="*60)
print("GROUND TRUTH:", ex["output"])
print("MODEL OUTPUT:", generated.split("\n")[0].strip())
print("="*60)
