# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_8_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 8
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 41.250000 |
| 34 | 35.000000 |
| 33 | 29.250000 |
| 32 | 24.875000 |
| 31 | 21.250000 |
| 30 | 18.875000 |
| 29 | 16.250000 |
| 28 | 14.812500 |
| 27 | 13.062500 |
| 26 | 11.500000 |

## Oracle True Layers

None