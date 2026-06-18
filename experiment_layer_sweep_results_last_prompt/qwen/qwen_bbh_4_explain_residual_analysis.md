# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_4_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 4
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 24.500000 |
| 34 | 20.750000 |
| 33 | 17.750000 |
| 32 | 15.125000 |
| 31 | 13.062500 |
| 30 | 11.500000 |
| 29 | 9.687500 |
| 28 | 8.625000 |
| 27 | 7.656250 |
| 26 | 6.625000 |

## Oracle True Layers

None