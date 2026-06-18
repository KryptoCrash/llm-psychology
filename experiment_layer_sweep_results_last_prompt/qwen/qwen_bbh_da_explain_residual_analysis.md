# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_da_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: da
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 26.125000 |
| 34 | 22.250000 |
| 33 | 19.000000 |
| 32 | 16.375000 |
| 31 | 14.000000 |
| 30 | 12.312500 |
| 29 | 10.625000 |
| 28 | 9.437500 |
| 27 | 8.062500 |
| 26 | 7.125000 |

## Oracle True Layers

None