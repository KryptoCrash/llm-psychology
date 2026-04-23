# LLM Psychology – Conformity Experiments

This project investigates whether LLMs exhibit conformity bias: does a model change its answer to a multiple-choice question when it sees prior (incorrect) answers from other "participants"?

The model used throughout is **Llama-3.2-3B-Instruct**, loaded via TransformerLens.
Questions are drawn from the **MMLU** benchmark.

---

## Pipeline

### Step 1 – Build the question bank
**`test_llama.py`** samples MMLU questions, prompts the model with no social context, and collects the first 25 questions it answers correctly. These become the controlled stimulus set: questions the model *can* answer correctly under neutral conditions.

Output: `correct_questions.json`, `results.json`

### Step 2 – Run conformity conditions
**`multi_actor.py`** re-runs the model on `correct_questions.json` under varying levels of social pressure. Pass `n` (total number of participants) as an argument — the model plays Participant n, and the n−1 prior participants all give the same incorrect answer.

| n | Condition |
|---|---|
| 1 | Baseline — no prior participants shown |
| 2 | One prior participant shown, answering incorrectly |
| ≥3 | n−1 prior participants shown, all answering incorrectly |

Output: `multi_actor_{n}_results.json`

Run as:
```
python multi_actor.py 3   # 3 total participants, 2 prior wrong answers
python multi_actor.py 10  # 10 total participants, 9 prior wrong answers
```

If the model's response doesn't contain a parseable `<answer>X</answer>` tag, the question is retried (up to 10 times, with sampling) before being marked unparseable.

---

## Supporting files

| File | Purpose |
|---|---|
| `explore_mmlu.py` | One-off script to inspect the MMLU dataset structure |
| `test_transformerlens.py` | Sanity-check that TransformerLens loads and generates correctly |
| `correct_questions.json` | The 25 MMLU questions used as stimuli (model answers these correctly at baseline) |
| `mmlu_cache/` | Local cache of the downloaded MMLU dataset |
| `env/` | Python virtual environment |
