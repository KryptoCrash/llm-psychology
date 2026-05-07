# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_5_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 5
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 31.125000 |
| 34 | 26.375000 |
| 33 | 22.500000 |
| 32 | 19.375000 |
| 31 | 16.500000 |
| 30 | 14.562500 |
| 29 | 12.187500 |
| 28 | 10.875000 |
| 27 | 9.500000 |
| 26 | 8.187500 |

## Oracle True Layers

None