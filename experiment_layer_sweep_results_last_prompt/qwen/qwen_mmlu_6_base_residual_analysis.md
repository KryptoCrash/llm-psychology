# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_6_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 6
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 62.750000 |
| 34 | 54.000000 |
| 33 | 46.250000 |
| 32 | 39.750000 |
| 31 | 34.750000 |
| 30 | 30.500000 |
| 29 | 24.500000 |
| 28 | 21.625000 |
| 27 | 18.750000 |
| 26 | 16.375000 |

## Oracle True Layers

None