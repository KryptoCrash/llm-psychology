# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_qd_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 76.500000 |
| 34 | 61.500000 |
| 33 | 51.500000 |
| 32 | 43.000000 |
| 31 | 37.250000 |
| 30 | 31.500000 |
| 29 | 26.625000 |
| 28 | 23.500000 |
| 27 | 20.375000 |
| 26 | 17.875000 |

## Oracle True Layers

None