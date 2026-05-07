# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_da_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: da
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 30.625000 |
| 34 | 25.250000 |
| 33 | 21.125000 |
| 32 | 18.125000 |
| 31 | 15.562500 |
| 30 | 13.750000 |
| 29 | 11.875000 |
| 28 | 10.625000 |
| 27 | 9.187500 |
| 26 | 8.187500 |

## Oracle True Layers

None