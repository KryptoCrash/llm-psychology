# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_6_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 6
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 38.750000 |
| 34 | 32.500000 |
| 33 | 27.000000 |
| 32 | 22.750000 |
| 31 | 19.375000 |
| 30 | 17.000000 |
| 29 | 14.562500 |
| 28 | 13.312500 |
| 27 | 11.750000 |
| 26 | 10.312500 |

## Oracle True Layers

None