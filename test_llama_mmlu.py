import json
import random
import re
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
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
    "meta-llama/Llama-3.1-8B-Instruct",
    device=device,
)
model.enable_compatibility_mode(disable_warnings=True)


print("Loading MMLU dataset...")
dataset = load_dataset("cais/mmlu", "all", cache_dir="./mmlu_cache")
test_df = dataset["test"].to_pandas().sample(frac=1, random_state=42).reset_index(drop=True)

LETTER = ["A", "B", "C", "D"]
TARGET_CORRECT = 1

correct = 0
attempts = 0
parseable = 0
correct_records = []
for _, row in test_df.iterrows():
    if correct >= TARGET_CORRECT:
        break
    choices = list(row["choices"])
    example_letter = random.choice(LETTER)
    prompt = (
        "Answer the following multiple choice question by giving the most appropriate response. "
        "Answer should be one among [A, B, C, D]. "
        "Provide your answer inside <answer></answer> XML tags.\n\n"
        f"Ex: <answer>{example_letter}</answer>\n\n"
        f"Question: {row['question']}\n"
        f"A: {choices[0]}\n"
        f"B: {choices[1]}\n"
        f"C: {choices[2]}\n"
        f"D: {choices[3]}\n\n"
        "Answer: "
    )

    outputs = model.generate(
        prompt,
        max_new_tokens=50,
        do_sample=False,
    )
    response = outputs[len(prompt):]
    match = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", response)
    answer = match.group(1).upper() if match else ""
    ground_truth = LETTER[row["answer"]]
    attempts += 1
    if answer:
        parseable += 1
    if answer != ground_truth:
        continue

    correct += 1
    correct_records.append({
        "subject": row["subject"],
        "question": row["question"],
        "choices": choices,
        "answer": int(row["answer"]),
        "model_answer": answer,
        "raw_response": response.strip(),
    })

accuracy = correct / parseable if parseable else 0.0

questions = [
    {
        "subject": r["subject"],
        "question": r["question"],
        "choices": r["choices"],
        "answer": r["answer"],
    }
    for r in correct_records
]
with open("correct_questions.json", "w") as f:
    json.dump(questions, f, indent=2)

summary = {"accuracy": accuracy, "attempts": attempts, "parseable": parseable, "correct": correct, "results": correct_records}
with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Collected {correct} correct responses in {attempts} attempts (accuracy={accuracy:.3f})")
print("Questions saved to correct_questions.json")
print("Full results saved to results.json")
