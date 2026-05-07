# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_10_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 10
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 48.250000 |
| 34 | 41.000000 |
| 33 | 34.250000 |
| 32 | 29.250000 |
| 31 | 24.750000 |
| 30 | 21.875000 |
| 29 | 18.750000 |
| 28 | 16.875000 |
| 27 | 14.625000 |
| 26 | 12.875000 |

## Oracle True Layers

None