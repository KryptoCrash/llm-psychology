# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_7_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 7
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 31.875000 |
| 34 | 27.250000 |
| 33 | 23.250000 |
| 32 | 19.750000 |
| 31 | 17.000000 |
| 30 | 15.250000 |
| 29 | 12.875000 |
| 28 | 11.750000 |
| 27 | 10.312500 |
| 26 | 9.000000 |

## Oracle True Layers

None