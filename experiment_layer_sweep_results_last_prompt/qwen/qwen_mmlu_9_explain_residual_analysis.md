# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_9_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 9
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 44.500000 |
| 34 | 38.000000 |
| 33 | 31.750000 |
| 32 | 26.875000 |
| 31 | 22.750000 |
| 30 | 20.125000 |
| 29 | 17.375000 |
| 28 | 15.875000 |
| 27 | 13.937500 |
| 26 | 12.312500 |

## Oracle True Layers

None