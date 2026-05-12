"""FastAPI server with Redis cache for identical diffs."""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import redis
except ImportError:
    redis = None

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "training/checkpoints/sanity")
REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
MAX_NEW_TOKENS = 40

state = {}


def _cache_key(diff: str) -> str:
    h = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]
    return f"commit-msg:{MODEL_ID}:{h}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lifespan] device={device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True
    )
    try:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(device)
        print(f"[lifespan] adapter loaded from {ADAPTER_PATH}")
    except Exception as e:
        print(f"[lifespan] no adapter loaded ({e}) - using base model only")
        model = base
    model.eval()
    state.update(tok=tok, model=model, device=device)

    if REDIS_URL and redis is not None:
        try:
            r = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
            r.ping()
            state["redis"] = r
            print(f"[lifespan] redis connected: {REDIS_URL}")
        except Exception as e:
            print(f"[lifespan] redis unavailable ({e}) — running without cache")
    print("[lifespan] ready.")
    yield
    state.clear()


app = FastAPI(title="commit-msg-llm", lifespan=lifespan)
# Custom metrics
GEN_TOTAL = Counter("generate_total", "Total /generate calls", ["status"])
GEN_LATENCY = Histogram(
    "generate_latency_seconds", "Inference latency", buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
)
CACHE_EVENTS = Counter("cache_events_total", "Cache hits / misses", ["event"])

Instrumentator().instrument(app).expose(app)


class GenerateRequest(BaseModel):
    diff: str = Field(..., min_length=10, max_length=8000)


class GenerateResponse(BaseModel):
    message: str
    model: str
    cached: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "cache_enabled": "redis" in state,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        GEN_TOTAL.labels(status="error").inc()
        raise HTTPException(503, "Model not loaded")
    t0 = time.perf_counter()
    r = state.get("redis")
    key = _cache_key(req.diff)

    if r is not None:
        cached = r.get(key)
        if cached:
            CACHE_EVENTS.labels(event="hit").inc()
            GEN_TOTAL.labels(status="cached").inc()
            GEN_LATENCY.observe(time.perf_counter() - t0)
            data = json.loads(cached)
            return GenerateResponse(message=data["message"], model=data["model"], cached=True)
        CACHE_EVENTS.labels(event="miss").inc()

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
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    msg = text.split("\n")[0].strip(" -")

    if r is not None:
        try:
            r.setex(key, CACHE_TTL, json.dumps({"message": msg, "model": MODEL_ID}))
        except Exception:
            pass

    GEN_TOTAL.labels(status="ok").inc()
    GEN_LATENCY.observe(time.perf_counter() - t0)
    return GenerateResponse(message=msg, model=MODEL_ID, cached=False)
