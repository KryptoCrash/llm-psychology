"""Build data/combined_{mmlu,bbh}_correct.json.

Runs answer_without_context.py for Llama and Qwen on successive 200-question
batches (one subprocess at a time) and appends questions both models answer
correctly until each dataset has TARGET_SIZE items.
"""
import json
import subprocess
import sys
from pathlib import Path

TARGET_SIZE = 100
BATCH_SIZE = 200
MODELS = ["llama", "qwen"]
SCRIPT = Path(__file__).with_name("answer_without_context.py")
DATA_DIR = Path("data")
SELECTION_DIR = Path("results/item_selection")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def combine(correct_a, correct_b):
    correct_b_qs = {r["question"] for r in correct_b}
    return [r for r in correct_a if r["question"] in correct_b_qs]


def run_batch(model, dataset, offset):
    cmd = [sys.executable, str(SCRIPT), "--model", model, "--dataset", dataset,
           "--offset", str(offset), "--size", str(BATCH_SIZE), "--output-dir", str(SELECTION_DIR)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


combined = {d: load_json(DATA_DIR / f"combined_{d}_correct.json") for d in ["mmlu", "bbh"]}
print(f"Starting sizes — BBH: {len(combined['bbh'])}/{TARGET_SIZE}, MMLU: {len(combined['mmlu'])}/{TARGET_SIZE}")

offset = 0
while any(len(items) < TARGET_SIZE for items in combined.values()):
    print(f"\n--- Batch offset={offset} size={BATCH_SIZE} ---")
    for dataset, items in combined.items():
        if len(items) >= TARGET_SIZE:
            continue
        for model in MODELS:
            run_batch(model, dataset, offset)
        new_intersect = combine(
            load_json(SELECTION_DIR / f"llama_{dataset}_correct.json"),
            load_json(SELECTION_DIR / f"qwen_{dataset}_correct.json"),
        )
        items.extend(new_intersect)
        del items[TARGET_SIZE:]
        save_json(DATA_DIR / f"combined_{dataset}_correct.json", items)
        print(f"{dataset.upper()}: {len(items)}/{TARGET_SIZE} (+{len(new_intersect)} new)")
    offset += BATCH_SIZE

print(f"\nDone — BBH: {len(combined['bbh'])}/{TARGET_SIZE}, MMLU: {len(combined['mmlu'])}/{TARGET_SIZE}")
