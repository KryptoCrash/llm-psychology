# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_9_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 9
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 36.250000 |
| 34 | 31.250000 |
| 33 | 26.750000 |
| 32 | 22.875000 |
| 31 | 19.625000 |
| 30 | 17.500000 |
| 29 | 14.937500 |
| 28 | 13.625000 |
| 27 | 11.937500 |
| 26 | 10.437500 |

## Oracle True Layers

None