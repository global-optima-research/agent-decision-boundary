"""
Step 3: TriDecision DPO training on Qwen2.5-7B-Instruct (full 500-task data).

~4000 weighted preference pairs → DPO + LoRA training.
Hyperparameters tuned down from pilot to avoid overfitting on larger data.

Usage (on 5090):
    HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision-full/03_train_dpo.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PAIRS_PATH = Path(__file__).parent / "preference_pairs.json"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "tridecision-full"

# Hyperparameters — adjusted for larger dataset (vs pilot: lr=5e-5, epochs=10)
LEARNING_RATE = 2e-5      # lower lr for more data (pilot used 5e-5)
NUM_EPOCHS = 3            # fewer epochs to avoid overfitting (pilot used 10)
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 8
MAX_LENGTH = 1024
BETA = 0.1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_preference_data(tokenizer):
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)

    records = []
    for pair in pairs:
        prompt_text = tokenizer.apply_chat_template(
            pair["prompt"], tokenize=False, add_generation_prompt=True
        )
        records.append({
            "prompt": prompt_text,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} preference pairs")
    return dataset


def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )

    dataset = load_preference_data(tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    output_dir = str(OUTPUT_DIR)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"\nTraining config:")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Beta: {BETA}")
    print(f"  Train samples: {len(split['train'])}")
    print(f"  Eval samples: {len(split['test'])}")
    print(f"  Output: {output_dir}")

    print("\nStarting training...")
    trainer.train()

    final_dir = str(OUTPUT_DIR / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
