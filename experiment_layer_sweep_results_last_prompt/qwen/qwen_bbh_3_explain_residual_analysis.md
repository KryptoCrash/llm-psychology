# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_3_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 3
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 15.062500 |
| 34 | 12.687500 |
| 33 | 11.062500 |
| 32 | 9.500000 |
| 31 | 8.125000 |
| 30 | 7.156250 |
| 29 | 6.062500 |
| 28 | 5.437500 |
| 27 | 4.718750 |
| 26 | 4.031250 |

## Oracle True Layers

None