"""FastAPI server: takes a git diff, returns a commit message."""
import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "training/checkpoints/sanity")
MAX_NEW_TOKENS = 40

state = {}  # holds model + tokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lifespan] device={device}")
    print("[lifespan] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    print("[lifespan] loading base model...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

    print(f"[lifespan] loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(device)
    model.eval()

    state["tok"] = tok
    state["model"] = model
    state["device"] = device
    print("[lifespan] ready.")
    yield
    state.clear()

app = FastAPI(title="commit-msg-llm", lifespan=lifespan)

class GenerateRequest(BaseModel):
    diff: str = Field(..., min_length=10, max_length=8000)

class GenerateResponse(BaseModel):
    message: str
    model: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    tok, model, device = state["tok"], state["model"], state["device"]
    prompt = (
        "Write a concise git commit message for the following diff.\n\n"
        f"### Diff:\n{req.diff}\n\n"
        "### Commit message:\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    first_line = text.split("\n")[0].strip(" -")
    return GenerateResponse(message=first_line, model=MODEL_ID)
