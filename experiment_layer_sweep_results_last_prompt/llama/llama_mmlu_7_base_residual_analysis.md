# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_7_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 7
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.125000 |
| 30 | 1.789062 |
| 29 | 1.601562 |
| 28 | 1.476562 |
| 27 | 1.375000 |
| 26 | 1.273438 |
| 25 | 1.171875 |
| 24 | 1.093750 |
| 23 | 1.015625 |
| 22 | 0.949219 |

## Oracle True Layers

None