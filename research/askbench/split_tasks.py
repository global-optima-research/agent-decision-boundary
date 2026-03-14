"""
Split 600 AskBench tasks into train/test sets.

Stratified by domain × label to ensure proportional representation.
Split: 500 train / 100 test (5:1 ratio per stratum).

Usage:
    python3 research/askbench/split_tasks.py

Output:
    research/askbench/train.json  (400 tasks)
    research/askbench/test.json   (200 tasks)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 5 / 6  # 500/600

def main():
    tasks_path = Path(__file__).parent / "tasks.json"
    with open(tasks_path) as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks")

    # Group by (domain, gold_label) for stratified split
    strata = defaultdict(list)
    for task in tasks:
        domain = task["task_id"].split("-")[0]  # D1, D2, ...
        label = task["gold_label"]
        strata[(domain, label)].append(task)

    random.seed(SEED)
    train, test = [], []

    # First pass: floor-based split per stratum
    splits = {}
    for key in sorted(strata.keys()):
        group = strata[key]
        random.shuffle(group)
        n_train = int(len(group) * TRAIN_RATIO)
        splits[key] = (group, n_train)

    # Adjust to hit exactly 500 train
    total_train = sum(v[1] for v in splits.values())
    deficit = 500 - total_train
    # Add 1 to the largest strata first to fill the gap
    sorted_keys = sorted(splits.keys(), key=lambda k: len(splits[k][0]), reverse=True)
    for key in sorted_keys:
        if deficit <= 0:
            break
        group, n_train = splits[key]
        if n_train < len(group):
            splits[key] = (group, n_train + 1)
            deficit -= 1

    print(f"\n{'Stratum':<12} {'Total':>5} {'Train':>5} {'Test':>5}")
    print("-" * 35)

    for key in sorted(splits.keys()):
        group, n_train = splits[key]
        train.extend(group[:n_train])
        test.extend(group[n_train:])
        print(f"{key[0]}-{key[1]:<7} {len(group):>5} {n_train:>5} {len(group) - n_train:>5}")

    print("-" * 35)
    print(f"{'Total':<12} {len(tasks):>5} {len(train):>5} {len(test):>5}")

    # Verify label distribution
    from collections import Counter
    print(f"\n{'Split':<6} {'Act':>4} {'Ask':>4} {'Refuse':>6} {'Total':>6}")
    print("-" * 30)
    for name, split in [("Train", train), ("Test", test)]:
        c = Counter(t["gold_label"] for t in split)
        print(f"{name:<6} {c['act']:>4} {c['ask']:>4} {c['refuse']:>6} {sum(c.values()):>6}")

    # Verify domain distribution
    print(f"\n{'Domain':<6} {'Train':>5} {'Test':>5}")
    print("-" * 20)
    for d in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]:
        tr = sum(1 for t in train if t["task_id"].startswith(d))
        te = sum(1 for t in test if t["task_id"].startswith(d))
        print(f"{d:<6} {tr:>5} {te:>5}")

    # Save
    out_dir = Path(__file__).parent
    for name, split in [("train", train), ("test", test)]:
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(split)} tasks to {path}")


if __name__ == "__main__":
    main()
