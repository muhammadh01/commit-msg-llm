"""Verify Qwen2.5-1.5B loads on this Mac."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)

print("Loading model (this downloads ~3GB the first time)...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)

print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# Quick generation test
prompt = "def hello_world():"
inputs = tok(prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=20)
print("Test output:", tok.decode(out[0], skip_special_tokens=True))
