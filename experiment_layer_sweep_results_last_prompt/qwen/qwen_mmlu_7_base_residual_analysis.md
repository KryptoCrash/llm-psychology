# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_7_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 7
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 72.000000 |
| 34 | 62.250000 |
| 33 | 53.250000 |
| 32 | 45.500000 |
| 31 | 39.750000 |
| 30 | 35.000000 |
| 29 | 28.500000 |
| 28 | 25.125000 |
| 27 | 21.750000 |
| 26 | 19.000000 |

## Oracle True Layers

None