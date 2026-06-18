# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_7_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 7
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 40.250000 |
| 34 | 34.000000 |
| 33 | 28.500000 |
| 32 | 24.125000 |
| 31 | 20.750000 |
| 30 | 18.250000 |
| 29 | 15.687500 |
| 28 | 14.312500 |
| 27 | 12.562500 |
| 26 | 11.062500 |

## Oracle True Layers

None