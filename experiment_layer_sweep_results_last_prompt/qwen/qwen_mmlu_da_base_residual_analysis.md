# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_da_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: da
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 58.000000 |
| 34 | 49.250000 |
| 33 | 43.000000 |
| 32 | 36.500000 |
| 31 | 31.750000 |
| 30 | 28.375000 |
| 29 | 24.000000 |
| 28 | 20.875000 |
| 27 | 17.875000 |
| 26 | 15.687500 |

## Oracle True Layers

None