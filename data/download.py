"""Download a small sample of CommitChronicle for inspection."""
from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("data/raw/sample.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Streaming dataset (this avoids downloading the full thing)...")
ds = load_dataset(
    "JetBrains-Research/commit-chronicle",
    "default",
    split="train",
    streaming=True,
)

N = 500
with OUT.open("w") as f:
    for i, row in enumerate(ds):
        if i >= N:
            break
        f.write(json.dumps(row) + "\n")
        if (i + 1) % 100 == 0:
            print(f"  saved {i+1}/{N}")

print(f"Done. Saved {N} examples to {OUT}")
