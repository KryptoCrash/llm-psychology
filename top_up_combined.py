"""
Runs llama and qwen scripts on new question batches (one subprocess at a time)
and appends intersections to combined_*_correct.json until both reach TARGET_SIZE.
"""
import json
import subprocess
import sys

TARGET_SIZE = 100
BATCH_SIZE = 200
# First 200 questions were already used by the original runs
INITIAL_OFFSET = 200

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def combine(correct_a, correct_b):
    correct_b_qs = {r["question"] for r in correct_b}
    return [r for r in correct_a if r["question"] in correct_b_qs]

def run_script(script, offset, size):
    cmd = [sys.executable, script, "--offset", str(offset), "--size", str(size)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

combined_bbh = load_json("combined_bbh_correct.json")
combined_mmlu = load_json("combined_mmlu_correct.json")
print(f"Starting sizes — BBH: {len(combined_bbh)}/100, MMLU: {len(combined_mmlu)}/100")

offset = INITIAL_OFFSET

while len(combined_bbh) < TARGET_SIZE or len(combined_mmlu) < TARGET_SIZE:
    need_bbh = len(combined_bbh) < TARGET_SIZE
    need_mmlu = len(combined_mmlu) < TARGET_SIZE

    print(f"\n--- Batch offset={offset} size={BATCH_SIZE} ---")

    if need_bbh:
        print("Running llama_bbh.py...")
        run_script("llama_bbh.py", offset, BATCH_SIZE)
        print("Running qwen_bbh.py...")
        run_script("qwen_bbh.py", offset, BATCH_SIZE)

        llama_correct = load_json("llama_bbh_correct.json")
        qwen_correct = load_json("qwen_bbh_correct.json")
        new_intersect = combine(llama_correct, qwen_correct)
        combined_bbh.extend(new_intersect)
        if len(combined_bbh) > TARGET_SIZE:
            combined_bbh = combined_bbh[:TARGET_SIZE]
        save_json("combined_bbh_correct.json", combined_bbh)
        print(f"BBH: {len(combined_bbh)}/100 (+{len(new_intersect)} new)")

    if need_mmlu:
        print("Running llama_mmlu.py...")
        run_script("llama_mmlu.py", offset, BATCH_SIZE)
        print("Running qwen_mmlu.py...")
        run_script("qwen_mmlu.py", offset, BATCH_SIZE)

        llama_correct = load_json("llama_mmlu_correct.json")
        qwen_correct = load_json("qwen_mmlu_correct.json")
        new_intersect = combine(llama_correct, qwen_correct)
        combined_mmlu.extend(new_intersect)
        if len(combined_mmlu) > TARGET_SIZE:
            combined_mmlu = combined_mmlu[:TARGET_SIZE]
        save_json("combined_mmlu_correct.json", combined_mmlu)
        print(f"MMLU: {len(combined_mmlu)}/100 (+{len(new_intersect)} new)")

    offset += BATCH_SIZE

print(f"\nDone — BBH: {len(combined_bbh)}/100, MMLU: {len(combined_mmlu)}/100")
