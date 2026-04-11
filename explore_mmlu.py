import pandas as pd
from datasets import load_dataset

# ── 1. Load MMLU ──────────────────────────────────────────────────────────────
# 'all' loads every subject at once. You can also pass a subject name like
# "abstract_algebra", "clinical_knowledge", "high_school_mathematics", etc.

print("Loading MMLU dataset...")
dataset = load_dataset("cais/mmlu", "all", cache_dir="./mmlu_cache")
print(dataset)

# ── 2. Inspect splits ─────────────────────────────────────────────────────────
# Splits: 'test', 'validation', 'dev' (few-shot examples), 'auxiliary_train'
for split in dataset:
    print(f"\n{split}: {len(dataset[split])} examples")
    print(dataset[split][0])   # print first example

# ── 3. Convert to pandas for easy exploration ─────────────────────────────────
test_df = dataset["test"].to_pandas()
print("\nColumns:", test_df.columns.tolist())
print(test_df.head())

# ── 4. Subjects breakdown ─────────────────────────────────────────────────────
print("\nSubject counts (test set):")
print(test_df["subject"].value_counts().to_string())

print(f"\nTotal subjects: {test_df['subject'].nunique()}")
print(f"Total test questions: {len(test_df)}")

# ── 5. Look at a few examples from a specific subject ─────────────────────────
subject = "clinical_knowledge"
subset = test_df[test_df["subject"] == subject].head(3)

for _, row in subset.iterrows():
    print(f"\n{'─'*60}")
    print(f"Subject : {row['subject']}")
    print(f"Question: {row['question']}")
    print(f"Choices : A) {row['choices'][0]}")
    print(f"          B) {row['choices'][1]}")
    print(f"          C) {row['choices'][2]}")
    print(f"          D) {row['choices'][3]}")
    print(f"Answer  : {['A','B','C','D'][row['answer']]}")