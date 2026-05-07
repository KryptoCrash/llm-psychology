# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_da_base.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: da
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 45.750000 |
| 34 | 39.250000 |
| 33 | 34.250000 |
| 32 | 28.750000 |
| 31 | 24.750000 |
| 30 | 21.875000 |
| 29 | 18.375000 |
| 28 | 15.937500 |
| 27 | 13.625000 |
| 26 | 11.875000 |

## Oracle True Layers

None