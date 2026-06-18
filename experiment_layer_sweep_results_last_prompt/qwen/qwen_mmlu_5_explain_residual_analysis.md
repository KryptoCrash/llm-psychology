# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_5_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 5
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 34.250000 |
| 34 | 28.625000 |
| 33 | 23.750000 |
| 32 | 20.000000 |
| 31 | 17.000000 |
| 30 | 14.937500 |
| 29 | 12.812500 |
| 28 | 11.750000 |
| 27 | 10.437500 |
| 26 | 9.125000 |

## Oracle True Layers

None