# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_4_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 4
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 31.500000 |
| 34 | 26.250000 |
| 33 | 22.000000 |
| 32 | 18.500000 |
| 31 | 15.625000 |
| 30 | 13.687500 |
| 29 | 11.625000 |
| 28 | 10.562500 |
| 27 | 9.437500 |
| 26 | 8.187500 |

## Oracle True Layers

None