"""Filter + format CommitChronicle rows into prompt/completion pairs."""

import json
import random
from pathlib import Path

RAW = Path("data/raw/sample.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filter rules (from stats.py)
MIN_DIFF, MAX_DIFF = 100, 8000
MIN_MSG, MAX_MSG = 10, 250
SKIP_PREFIXES = ("Merge ", "Revert ", "[skip ci]")


def format_input(row):
    """Turn the row's file changes into a readable diff string."""
    parts = []
    for m in row["mods"]:
        path = m["new_path"] or m["old_path"]
        parts.append(f"# {m['change_type']} {path}\n{m['diff']}")
    return "\n\n".join(parts)


def keep(row):
    msg = row["message"].strip()
    if not (MIN_MSG <= len(msg) <= MAX_MSG):
        return False
    if msg.startswith(SKIP_PREFIXES):
        return False
    diff_len = sum(len(m["diff"]) for m in row["mods"])
    if not (MIN_DIFF <= diff_len <= MAX_DIFF):
        return False
    return True


examples = []
with RAW.open() as f:
    for line in f:
        row = json.loads(line)
        if not keep(row):
            continue
        examples.append(
            {
                "input": format_input(row),
                "output": row["message"].strip(),
            }
        )

print(f"Kept {len(examples)} / 500 examples after filtering")

# Shuffle + split 80/10/10
random.seed(42)
random.shuffle(examples)
n = len(examples)
train, val, test = (
    examples[: int(n * 0.8)],
    examples[int(n * 0.8) : int(n * 0.9)],
    examples[int(n * 0.9) :],
)

for name, data in [("train", train), ("val", val), ("test", test)]:
    p = OUT_DIR / f"{name}.jsonl"
    with p.open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"  {name}: {len(data)} -> {p}")
