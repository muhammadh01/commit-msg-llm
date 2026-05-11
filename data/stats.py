import json
from pathlib import Path

msgs, diffs, mods = [], [], []
with Path("data/raw/sample.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        msgs.append(len(r["message"]))
        mods.append(len(r["mods"]))
        diffs.append(sum(len(m["diff"]) for m in r["mods"]))


def stats(name, arr):
    arr = sorted(arr)
    print(
        f"{name}: min={arr[0]} median={arr[len(arr) // 2]} max={arr[-1]} avg={sum(arr) // len(arr)}"
    )


stats("message chars", msgs)
stats("files per commit", mods)
stats("total diff chars", diffs)
