"""
Step 4: Evaluate the TriDecision-trained model vs baseline on 30 pilot tasks.

Loads the LoRA-trained model and runs the same AskBench pilot evaluation,
then prints a comparison with the baseline Qwen2.5-7B results.

Usage (on 5090):
    HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision/04_evaluate.py

Options:
    --checkpoint PATH   Path to LoRA checkpoint (default: auto-detect latest)
    --baseline PATH     Path to baseline results JSON for comparison
"""

import json
import argparse
import sys
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent for shared evaluate code
sys.path.insert(0, str(Path(__file__).parent.parent / "askbench-pilot"))
from evaluate import (
    format_task, parse_response, Metrics, print_results, ERROR_WEIGHTS
)

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "tridecision-pilot" / "final"
TASKS_PATH = Path(__file__).parent.parent / "askbench-pilot" / "tasks.json"
BASELINE_PATH = Path(__file__).parent.parent / "askbench-pilot" / "results" / "results_qwen2.5-7b-local.json"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_trained_model(checkpoint_path):
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model = model.merge_and_unload()
    print("Model loaded and LoRA merged.")
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=300):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_trained(model, tokenizer, tasks):
    metrics = Metrics()
    results = []

    for i, task in enumerate(tasks):
        messages = format_task(task)
        raw = generate(model, tokenizer, messages)
        parsed = parse_response(raw)

        gold = task["gold_label"]
        pred = parsed["decision"]
        metrics.update(gold, pred)

        result = {
            "task_id": task["task_id"],
            "gold": gold,
            "pred": pred,
            "risk_level": task["risk_level"],
            "pred_confidence": parsed.get("confidence"),
            "pred_reason": parsed.get("reason"),
            "pred_question": parsed.get("question"),
        }
        results.append(result)

        status = "✓" if gold == pred else "✗"
        print(f"  [{i+1}/{len(tasks)}] {task['task_id']}: gold={gold}, pred={pred} {status}")

    return metrics, results


def load_baseline_metrics(baseline_path):
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    metrics = Metrics()
    for r in baseline_results:
        metrics.update(r["gold"], r["pred"])
    return metrics, baseline_results


def print_comparison(baseline_metrics, trained_metrics):
    print(f"\n{'='*70}")
    print(f"  TriDecision DPO: Before vs After")
    print(f"{'='*70}")

    headers = ["Metric", "Baseline", "Trained", "Delta"]
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
        # For WES/SVR/ULR, negative delta is good
        if "↓" in name:
            sign = "↓" if delta < 0 else "↑"
        else:
            sign = "↑" if delta > 0 else "↓"
        print(f"  {name:<12} {base_val:>10.3f} {train_val:>10.3f} {delta:>+9.3f} {sign}")

    # Per-class comparison
    print(f"\n  Per-class F1:")
    print(f"  {'Class':<10} {'Base':>8} {'Trained':>8} {'Delta':>8}")
    print(f"  {'-'*34}")
    for cls in ["act", "ask", "refuse"]:
        base_f1 = baseline_metrics.f1(cls)[2]
        train_f1 = trained_metrics.f1(cls)[2]
        delta = train_f1 - base_f1
        print(f"  {cls:<10} {base_f1:>8.3f} {train_f1:>8.3f} {delta:>+7.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TriDecision-trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline results JSON")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    baseline_path = Path(args.baseline) if args.baseline else BASELINE_PATH

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        print("Run 03_train_dpo.py first.")
        sys.exit(1)

    # Load tasks
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    # Load and evaluate trained model
    model, tokenizer = load_trained_model(checkpoint)
    print(f"\nEvaluating trained model on {len(tasks)} tasks...")
    trained_metrics, trained_results = evaluate_trained(model, tokenizer, tasks)
    print_results("TriDecision-Qwen2.5-7B", trained_metrics, trained_results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "results_tridecision.json"
    with open(output_file, "w") as f:
        json.dump(trained_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

    # Compare with baseline
    if baseline_path.exists():
        baseline_metrics, _ = load_baseline_metrics(baseline_path)
        print_comparison(baseline_metrics, trained_metrics)
    else:
        print(f"\nBaseline results not found at {baseline_path}, skipping comparison.")


if __name__ == "__main__":
    main()
