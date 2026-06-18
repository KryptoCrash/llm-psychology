# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_10_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 10
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 100.000000 |
| 34 | 85.000000 |
| 33 | 73.500000 |
| 32 | 62.500000 |
| 31 | 54.750000 |
| 30 | 49.250000 |
| 29 | 41.000000 |
| 28 | 35.500000 |
| 27 | 30.250000 |
| 26 | 26.375000 |

## Oracle True Layers

None