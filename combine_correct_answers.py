import json
import sys

if len(sys.argv) != 4:
    print("Usage: python combine_correct_answers.py <model_a_correct.json> <model_b_correct.json> <output.json>")
    sys.exit(1)

file_a, file_b, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(file_a) as f:
    records_a = json.load(f)
with open(file_b) as f:
    records_b = json.load(f)

correct_b = {r["question"] for r in records_b}

intersection = [r for r in records_a if r["question"] in correct_b]

with open(output_file, "w") as f:
    json.dump(intersection, f, indent=2)

print(f"Model A correct: {len(records_a)}")
print(f"Model B correct: {len(records_b)}")
print(f"Intersection (both correct): {len(intersection)}")
print(f"Saved to {output_file}")
