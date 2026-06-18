# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_3_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 3
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 20.000000 |
| 34 | 16.375000 |
| 33 | 13.500000 |
| 32 | 11.375000 |
| 31 | 9.625000 |
| 30 | 8.437500 |
| 29 | 7.250000 |
| 28 | 6.562500 |
| 27 | 5.812500 |
| 26 | 5.156250 |

## Oracle True Layers

None