import json
import transformers
import torch
from datasets import load_dataset

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Load 10 random questions from MMLU test split
print("Loading MMLU dataset...")
dataset = load_dataset("cais/mmlu", "all", cache_dir="./mmlu_cache")
test_df = dataset["test"].to_pandas().sample(n=10, random_state=42).reset_index(drop=True)

LETTER = ["A", "B", "C", "D"]

correct = 0
results = []
for i, row in test_df.iterrows():
    choices = row["choices"]
    prompt = (
        "Answer the following multiple choice question by giving the most appropriate response. "
        "Answer should be one among [A, B, C, D]. \n\n"
        f"Question: {row['question']}\n"
        f"A: {choices[0]}\n"
        f"B: {choices[1]}\n"
        f"C: {choices[2]}\n"
        f"D: {choices[3]}\n\n"
        "Answer: "
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=50,
        do_sample=False,
    )

    response = outputs[0]["generated_text"][len(prompt):]
    first_char = response.strip()[0]
    if first_char.isalpha():
        answer = first_char.upper()
    else:
        # Collect all characters up to the first alphabetic character, strip whitespace,
        # then match against the choice values
        prefix = ""
        for ch in response.strip():
            if ch.isalpha():
                break
            if not ch.isspace():
                prefix += ch
        matched = next(
            (LETTER[j] for j, choice in enumerate(choices) if prefix == choice.strip()),
            first_char,
        )
        answer = matched
    ground_truth = LETTER[row["answer"]]
    is_correct = answer == ground_truth
    correct += int(is_correct)

    results.append({
        "subject": row["subject"],
        "question": row["question"],
        "options": {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]},
        "model_answer": answer,
        "raw_response": response.strip(),
        "ground_truth": ground_truth,
        "correct": is_correct,
    })

accuracy = correct / 10

output = {"accuracy": accuracy, "results": results}
with open("results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Results saved to results.json")
