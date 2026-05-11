"""Use GPT-4o-mini to grade commit message quality on 1–10 scale."""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_PROMPT = """You are grading auto-generated git commit messages.

The diff:
---
{diff}
---

Ground truth message: "{ref}"
Candidate message:    "{pred}"

Rate the candidate on:
1. CLARITY (1-10) - is it readable English / proper format?
2. RELEVANCE (1-10) - does it describe what the diff actually does?
3. CONCISENESS (1-10) - is it short and to the point, no fluff?

Reply ONLY with JSON: {{"clarity": X, "relevance": Y, "conciseness": Z}}
"""


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).open()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out", default="eval/results_llm_judge.json")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (in .env or env var)")

    data = json.loads(Path(args.preds).read_text())
    examples = data["examples"][: args.limit]
    print(f"Judging {len(examples)} predictions with {args.model}")

    client = OpenAI()
    scores = []
    for i, ex in enumerate(examples, 1):
        prompt = JUDGE_PROMPT.format(
            diff="(truncated)" if ex["diff_chars"] > 3000 else ex.get("diff", ""),
            ref=ex["ref"],
            pred=ex["pred"],
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                s = json.loads(resp.choices[0].message.content)
                scores.append(s)
                break
            except Exception as e:
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(2)
        if i % 5 == 0:
            print(f"  {i}/{len(examples)}")

    avg = {
        "clarity": round(sum(s["clarity"] for s in scores) / len(scores), 2),
        "relevance": round(sum(s["relevance"] for s in scores) / len(scores), 2),
        "conciseness": round(sum(s["conciseness"] for s in scores) / len(scores), 2),
    }
    avg["overall"] = round(sum(avg.values()) / 3, 2)

    out = {"n": len(scores), "averages": avg, "per_example": scores}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nAVERAGE SCORES (1-10):")
    for k, v in avg.items():
        print(f"  {k:12s} {v}")
    print(f"  saved to {args.out}")


if __name__ == "__main__":
    main()
