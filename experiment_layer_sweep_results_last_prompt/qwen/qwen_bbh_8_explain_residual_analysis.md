# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_8_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 8
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 34.750000 |
| 34 | 30.125000 |
| 33 | 25.875000 |
| 32 | 22.125000 |
| 31 | 18.875000 |
| 30 | 16.875000 |
| 29 | 14.250000 |
| 28 | 13.000000 |
| 27 | 11.375000 |
| 26 | 10.000000 |

## Oracle True Layers

None