import json
import random
import re
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset, get_dataset_config_names
import os
import plotly.io as pio
import circuitsvis as cv
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
from jaxtyping import Float
from functools import partial
import transformer_lens.utilities as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import FactoredMatrix, HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

torch.set_grad_enabled(False)
device = utils.get_device()

# NBVAL_IGNORE_OUTPUT
model = TransformerBridge.boot_transformers(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    device=device,
)
# model.enable_compatibility_mode(disable_warnings=True)


print("Loading BigBenchHard dataset...")
subtasks = get_dataset_config_names("lukaemon/bbh")
all_rows = []
for subtask in subtasks:
    ds = load_dataset("lukaemon/bbh", subtask, cache_dir="./bbh_cache")
    for ex in ds["test"]:
        all_rows.append({
            "subject": subtask,
            "question": ex["input"],
            "target": ex["target"],
        })

random.seed(42)
random.shuffle(all_rows)
all_rows = all_rows[:200]

correct = 0
attempts = 0
parseable = 0
correct_records = []
all_records = []
for row in all_rows:
    prompt = (
        "Answer the following question by giving the most appropriate response. "
        "Provide your answer inside <answer></answer> XML tags.\n\n"
        "Ex: <answer>{your answer here}</answer>\n\n"
        f"Question: {row['question']}\n\n"
        "Answer: "
    )

    outputs = model.generate(
        prompt,
        max_new_tokens=50,
        do_sample=False,
    )
    response = outputs[len(prompt):]
    match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    answer = match.group(1).strip() if match else ""
    ground_truth = row["target"].strip()
    attempts += 1
    is_parseable = bool(answer)
    is_correct = answer.lower() == ground_truth.lower()
    if is_parseable:
        parseable += 1
    if is_correct:
        correct += 1

    record = {
        "subject": row["subject"],
        "question": row["question"],
        "prompt": prompt,
        "target": ground_truth,
        "model_answer": answer,
        "raw_response": response,
        "parseable": is_parseable,
        "correct": is_correct,
    }
    all_records.append(record)
    if is_correct:
        correct_records.append(record)

accuracy = correct / parseable if parseable else 0.0

with open("qwen_bbh_correct.json", "w") as f:
    json.dump(correct_records, f, indent=2)

summary = {"accuracy": accuracy, "attempts": attempts, "parseable": parseable, "correct": correct, "results": all_records}
with open("qwen_bbh_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Qwen BBH: {correct} correct out of {attempts} attempts (accuracy={accuracy:.3f})")
print("Correct answers saved to qwen_bbh_correct.json")
print("Full results saved to qwen_bbh_results.json")
