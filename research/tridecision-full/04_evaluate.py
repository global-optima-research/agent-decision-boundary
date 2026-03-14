"""
Step 4: Evaluate on held-out test set (100 tasks).

Runs both baseline Qwen2.5-7B and TriDecision-trained model on the
100 test tasks that were NEVER seen during training.

Usage (on 5090):
    HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision-full/04_evaluate.py

Options:
    --baseline-only   Only run baseline evaluation (skip trained model)
    --trained-only    Only run trained model evaluation (skip baseline)
    --checkpoint PATH Path to LoRA checkpoint
"""

import json
import argparse
import sys
import time
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "askbench-pilot"))
from evaluate import format_task, parse_response, Metrics, print_results, ERROR_WEIGHTS

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "tridecision-full" / "final"
TEST_PATH = Path(__file__).parent.parent / "askbench" / "test.json"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_base_model():
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    return model, tokenizer


def load_trained_model(checkpoint_path):
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model = model.merge_and_unload()
    print("LoRA merged.")
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=300):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate(model, tokenizer, tasks, label="Model"):
    metrics = Metrics()
    results = []

    t0 = time.time()
    for i, task in enumerate(tasks):
        messages = format_task(task)
        raw = generate(model, tokenizer, messages)
        parsed = parse_response(raw)

        gold = task["gold_label"]
        pred = parsed["decision"]
        metrics.update(gold, pred)

        results.append({
            "task_id": task["task_id"],
            "gold": gold,
            "pred": pred,
            "risk_level": task["risk_level"],
            "domain": task.get("domain", ""),
            "raw_response": raw,
        })

        status = "✓" if gold == pred else "✗"
        print(f"  [{i+1}/{len(tasks)}] {task['task_id']}: gold={gold}, pred={pred} {status}")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s ({elapsed/len(tasks):.1f}s/task)")
    return metrics, results


def print_comparison(baseline_metrics, trained_metrics):
    print(f"\n{'='*70}")
    print(f"  TriDecision DPO: Baseline vs Trained (HELD-OUT TEST SET)")
    print(f"{'='*70}")

    rows = [
        ("Accuracy", baseline_metrics.accuracy(), trained_metrics.accuracy()),
        ("Macro-F1", baseline_metrics.macro_f1(), trained_metrics.macro_f1()),
        ("WES (↓)", baseline_metrics.wes(), trained_metrics.wes()),
        ("SVR (↓)", baseline_metrics.safety_violation_rate(), trained_metrics.safety_violation_rate()),
        ("ULR (↓)", baseline_metrics.usability_loss_rate(), trained_metrics.usability_loss_rate()),
        ("Ask-F1", baseline_metrics.f1("ask")[2], trained_metrics.f1("ask")[2]),
    ]

    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    print(f"  {'-'*42}")
    for name, base_val, train_val in rows:
        delta = train_val - base_val
        sign = ("↓" if delta < 0 else "↑") if "↓" in name else ("↑" if delta > 0 else "↓")
        print(f"  {name:<12} {base_val:>10.3f} {train_val:>10.3f} {delta:>+9.3f} {sign}")

    print(f"\n  Per-class F1:")
    print(f"  {'Class':<10} {'Base':>8} {'Trained':>8} {'Delta':>8}")
    print(f"  {'-'*34}")
    for cls in ["act", "ask", "refuse"]:
        base_f1 = baseline_metrics.f1(cls)[2]
        train_f1 = trained_metrics.f1(cls)[2]
        delta = train_f1 - base_f1
        print(f"  {cls:<10} {base_f1:>8.3f} {train_f1:>8.3f} {delta:>+7.3f}")

    # Per-domain breakdown
    print(f"\n  (Full per-domain breakdown saved in results JSON)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--trained-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT

    with open(TEST_PATH) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} held-out test tasks")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_metrics = None
    trained_metrics = None

    # Baseline evaluation
    if not args.trained_only:
        print(f"\n{'='*50}")
        print(f"  Evaluating BASELINE Qwen2.5-7B on test set")
        print(f"{'='*50}")
        model, tokenizer = load_base_model()
        baseline_metrics, baseline_results = evaluate(model, tokenizer, tasks, "Baseline")
        print_results("Baseline-Qwen2.5-7B", baseline_metrics, baseline_results)

        with open(OUTPUT_DIR / "baseline_test.json", "w") as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        print(f"Saved to {OUTPUT_DIR / 'baseline_test.json'}")

        # Free memory before loading trained model
        del model, tokenizer
        torch.cuda.empty_cache()

    # Trained model evaluation
    if not args.baseline_only:
        if not checkpoint.exists():
            print(f"\nERROR: Checkpoint not found at {checkpoint}")
            print("Run 03_train_dpo.py first.")
            if baseline_metrics:
                return  # still show baseline results
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"  Evaluating TRAINED TriDecision on test set")
        print(f"{'='*50}")
        model, tokenizer = load_trained_model(checkpoint)
        trained_metrics, trained_results = evaluate(model, tokenizer, tasks, "TriDecision")
        print_results("TriDecision-Qwen2.5-7B", trained_metrics, trained_results)

        with open(OUTPUT_DIR / "trained_test.json", "w") as f:
            json.dump(trained_results, f, indent=2, ensure_ascii=False)
        print(f"Saved to {OUTPUT_DIR / 'trained_test.json'}")

    # Comparison
    if baseline_metrics and trained_metrics:
        print_comparison(baseline_metrics, trained_metrics)

        # Load baseline from file if only running trained
    elif args.trained_only and trained_metrics:
        baseline_path = OUTPUT_DIR / "baseline_test.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                br = json.load(f)
            baseline_metrics = Metrics()
            for r in br:
                baseline_metrics.update(r["gold"], r["pred"])
            print_comparison(baseline_metrics, trained_metrics)


if __name__ == "__main__":
    main()
