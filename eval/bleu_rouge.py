"""Run a trained model on test set; compute BLEU + ROUGE-L vs ground truth."""

import argparse
import json
from pathlib import Path

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).open()]


def build_prompt(diff):
    return (
        "Write a concise git commit message for the following diff.\n\n"
        f"### Diff:\n{diff}\n\n"
        "### Commit message:\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    p.add_argument("--test", default="data/processed/test.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--out", default="eval/results_bleu_rouge.json")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    print("loading model + adapter...")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()

    test = load_jsonl(args.test)[: args.limit]
    print(f"evaluating on {len(test)} examples")

    preds, refs = [], []
    for i, ex in enumerate(test, 1):
        prompt = build_prompt(ex["input"])
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        gen = gen.split("\n")[0].strip(" -")
        preds.append(gen)
        refs.append(ex["output"])
        if i % 10 == 0:
            print(f"  {i}/{len(test)}")

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_score = rouge.compute(predictions=preds, references=refs)

    results = {
        "n": len(test),
        "bleu": round(bleu_score["score"], 2),
        "rouge1": round(rouge_score["rouge1"] * 100, 2),
        "rougeL": round(rouge_score["rougeL"] * 100, 2),
        "examples": [
            {"diff_chars": len(t["input"]), "ref": r, "pred": p}
            for t, r, p in zip(test, refs, preds)
        ][:5],
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nRESULTS:")
    print(f"  BLEU:    {results['bleu']}")
    print(f"  ROUGE-1: {results['rouge1']}")
    print(f"  ROUGE-L: {results['rougeL']}")
    print(f"  saved to {args.out}")


if __name__ == "__main__":
    main()
