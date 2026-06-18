# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_10_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 10
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 46.000000 |
| 34 | 39.250000 |
| 33 | 33.500000 |
| 32 | 28.500000 |
| 31 | 24.250000 |
| 30 | 21.250000 |
| 29 | 18.000000 |
| 28 | 16.125000 |
| 27 | 13.937500 |
| 26 | 12.125000 |

## Oracle True Layers

None