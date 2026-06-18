# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_8_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 8
- prompt_style: explain
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 48.750000 |
| 34 | 41.250000 |
| 33 | 34.500000 |
| 32 | 29.125000 |
| 31 | 24.750000 |
| 30 | 21.875000 |
| 29 | 18.875000 |
| 28 | 17.125000 |
| 27 | 15.000000 |
| 26 | 13.187500 |

## Oracle True Layers

None