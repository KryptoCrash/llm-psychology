# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_9_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 9
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 79.500000 |
| 34 | 68.500000 |
| 33 | 58.750000 |
| 32 | 50.000000 |
| 31 | 44.000000 |
| 30 | 39.000000 |
| 29 | 31.750000 |
| 28 | 27.625000 |
| 27 | 23.500000 |
| 26 | 20.625000 |

## Oracle True Layers

None