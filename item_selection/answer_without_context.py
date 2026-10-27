"""Answer a batch of MMLU or BBH questions with no social context.

Writes {model}_{dataset}_correct.json (correctly answered records) and
{model}_{dataset}_results.json (all records plus accuracy) to --output-dir.
build_item_sets.py intersects the Llama and Qwen correct sets to build the
fixed 100-item subsets in data/.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
import transformer_lens.utilities as utils
from datasets import get_dataset_config_names, load_dataset
from transformer_lens.model_bridge import TransformerBridge

MODEL_IDS = {
    "llama": ("Llama", "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen": ("Qwen", "Qwen/Qwen3-8B"),
}
LETTER = ["A", "B", "C", "D"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
parser.add_argument("--dataset", choices=["mmlu", "bbh"], required=True)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--size", type=int, default=200)
parser.add_argument("--output-dir", type=Path, default=Path("results/item_selection"))
parser.add_argument("--cache-dir", type=Path, default=Path("data"))
args = parser.parse_args()

torch.set_grad_enabled(False)
device = utils.get_device()

display_name, model_id = MODEL_IDS[args.model]
model = TransformerBridge.boot_transformers(
    model_id,
    dtype=torch.bfloat16,
    device=device,
)


def load_mmlu_rows():
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", cache_dir=str(args.cache_dir / "mmlu_cache"))
    test_df = dataset["test"].to_pandas().sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.iloc[args.offset : args.offset + args.size]
    return [row for _, row in test_df.iterrows()]


def load_bbh_rows():
    print("Loading BigBenchHard dataset...")
    all_rows = []
    for subtask in get_dataset_config_names("lukaemon/bbh"):
        ds = load_dataset("lukaemon/bbh", subtask, cache_dir=str(args.cache_dir / "bbh_cache"))
        for ex in ds["test"]:
            all_rows.append({
                "subject": subtask,
                "question": ex["input"],
                "target": ex["target"],
            })
    random.seed(42)
    random.shuffle(all_rows)
    return all_rows[args.offset : args.offset + args.size]


def mmlu_prompt(row):
    choices = list(row["choices"])
    example_letter = random.choice(LETTER)
    return (
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


def bbh_prompt(row):
    return (
        "Answer the following question by giving the most appropriate response. "
        "Provide your answer inside <answer></answer> XML tags.\n\n"
        "If the question is multiple choice respond just with the letter (ex  <answer>(A)</answer>)"
        "Otherwise format your answer like this: <answer>{your answer here}</answer>\n\n"
        f"Question: {row['question']}\n\n"
        "Answer: "
    )


def grade_mmlu(row, prompt, response):
    match = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", response)
    answer = match.group(1).upper() if match else ""
    is_correct = answer == LETTER[row["answer"]]
    record = {
        "subject": row["subject"],
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer": int(row["answer"]),
        "prompt": prompt,
        "model_answer": answer,
        "raw_response": response,
        "parseable": bool(answer),
        "correct": is_correct,
    }
    return record


def grade_bbh(row, prompt, response):
    match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    answer = match.group(1).strip() if match else ""
    ground_truth = row["target"].strip()
    record = {
        "subject": row["subject"],
        "question": row["question"],
        "prompt": prompt,
        "target": ground_truth,
        "model_answer": answer,
        "raw_response": response,
        "parseable": bool(answer),
        "correct": answer.lower() == ground_truth.lower(),
    }
    return record


if args.dataset == "mmlu":
    rows, make_prompt, grade = load_mmlu_rows(), mmlu_prompt, grade_mmlu
else:
    rows, make_prompt, grade = load_bbh_rows(), bbh_prompt, grade_bbh

correct_records = []
all_records = []
for row in rows:
    prompt = make_prompt(row)
    outputs = model.generate(
        prompt,
        max_new_tokens=50,
        do_sample=False,
    )
    record = grade(row, prompt, outputs[len(prompt):])
    all_records.append(record)
    if record["correct"]:
        correct_records.append(record)

attempts = len(all_records)
parseable = sum(r["parseable"] for r in all_records)
correct = len(correct_records)
accuracy = correct / parseable if parseable else 0.0

args.output_dir.mkdir(parents=True, exist_ok=True)
correct_file = args.output_dir / f"{args.model}_{args.dataset}_correct.json"
results_file = args.output_dir / f"{args.model}_{args.dataset}_results.json"
with correct_file.open("w") as f:
    json.dump(correct_records, f, indent=2)

summary = {"accuracy": accuracy, "attempts": attempts, "parseable": parseable, "correct": correct, "results": all_records}
with results_file.open("w") as f:
    json.dump(summary, f, indent=2)
print(f"{display_name} {args.dataset.upper()}: {correct} correct out of {attempts} attempts (accuracy={accuracy:.3f})")
print(f"Correct answers saved to {correct_file}")
print(f"Full results saved to {results_file}")
