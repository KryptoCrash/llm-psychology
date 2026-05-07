# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_qd_base.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: qd
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 110.500000 |
| 34 | 93.000000 |
| 33 | 82.000000 |
| 32 | 71.500000 |
| 31 | 61.750000 |
| 30 | 54.000000 |
| 29 | 44.000000 |
| 28 | 39.250000 |
| 27 | 34.250000 |
| 26 | 30.000000 |

## Oracle True Layers

None