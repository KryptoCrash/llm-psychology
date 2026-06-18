# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_8_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 8
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.203125 |
| 30 | 1.046875 |
| 29 | 0.925781 |
| 28 | 0.847656 |
| 27 | 0.796875 |
| 26 | 0.750000 |
| 25 | 0.703125 |
| 24 | 0.664062 |
| 23 | 0.632812 |
| 22 | 0.601562 |

## Oracle True Layers

None