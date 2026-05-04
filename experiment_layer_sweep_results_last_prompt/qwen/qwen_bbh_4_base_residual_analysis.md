# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_4_base.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 4
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 37.500000 |
| 34 | 31.250000 |
| 33 | 27.625000 |
| 32 | 23.750000 |
| 31 | 20.750000 |
| 30 | 18.250000 |
| 29 | 14.937500 |
| 28 | 13.125000 |
| 27 | 11.187500 |
| 26 | 9.625000 |

## Oracle True Layers

None