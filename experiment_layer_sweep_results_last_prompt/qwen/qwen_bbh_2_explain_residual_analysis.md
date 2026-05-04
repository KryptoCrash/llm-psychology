# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_2_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: 2
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 4.156250 |
| 34 | 3.453125 |
| 33 | 2.968750 |
| 32 | 2.531250 |
| 31 | 2.171875 |
| 30 | 1.875000 |
| 29 | 1.585938 |
| 28 | 1.382812 |
| 27 | 1.195312 |
| 26 | 1.023438 |

## Oracle True Layers

None