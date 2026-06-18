# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_9_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 9
- prompt_style: explain
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 38.000000 |
| 34 | 32.500000 |
| 33 | 28.500000 |
| 32 | 24.125000 |
| 31 | 21.125000 |
| 30 | 18.125000 |
| 29 | 15.500000 |
| 28 | 13.937500 |
| 27 | 12.437500 |
| 26 | 10.875000 |

## Oracle True Layers

None