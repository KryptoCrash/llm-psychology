# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_7_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 7
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.132812 |
| 30 | 0.984375 |
| 29 | 0.871094 |
| 28 | 0.800781 |
| 27 | 0.746094 |
| 26 | 0.703125 |
| 25 | 0.664062 |
| 24 | 0.621094 |
| 23 | 0.589844 |
| 22 | 0.558594 |

## Oracle True Layers

None