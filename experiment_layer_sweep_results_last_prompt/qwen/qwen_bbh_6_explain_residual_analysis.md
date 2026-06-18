# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_6_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 6
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 31.000000 |
| 34 | 26.375000 |
| 33 | 22.625000 |
| 32 | 19.250000 |
| 31 | 16.500000 |
| 30 | 14.625000 |
| 29 | 12.312500 |
| 28 | 11.250000 |
| 27 | 9.937500 |
| 26 | 8.687500 |

## Oracle True Layers

None