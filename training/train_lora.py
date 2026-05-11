"""LoRA fine-tuning of Qwen2.5-1.5B with MLflow tracking."""

import json
import os
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# ---------- Config (overridable via env) ----------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B")
DATA_DIR = Path("data/processed")
RUN_NAME = os.getenv("RUN_NAME", "sanity")
OUTPUT_DIR = f"training/checkpoints/{RUN_NAME}"
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "20"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LR = float(os.getenv("LR", "2e-4"))
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
MAX_LEN = int(os.getenv("MAX_LEN", "1024"))

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ---------- MLflow ----------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("commit-msg-llm")


class MLflowLogger(TrainerCallback):
    """Push every logged metric into the active MLflow run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=state.global_step)


with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params(
        {
            "model_id": MODEL_ID,
            "n_examples": N_EXAMPLES,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_len": MAX_LEN,
            "device": device,
        }
    )

    # ---------- Load tokenizer + model ----------
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ap = sum(p.numel() for p in model.parameters())
    print(f"trainable: {tp:,} / {ap:,} ({100 * tp / ap:.3f}%)")
    mlflow.log_metric("trainable_params", tp)

    # ---------- Data ----------
    raw = [json.loads(line) for line in (DATA_DIR / "train.jsonl").open()][:N_EXAMPLES]

    def to_prompt(ex):
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
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        max_length=MAX_LEN,
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[MLflowLogger()],
    )

    print("Training...")
    trainer.train()

    # ---------- Save ----------
    trainer.save_model(OUTPUT_DIR)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="adapter")
    print(f"Done. Adapter saved to {OUTPUT_DIR}, logged to MLflow.")
