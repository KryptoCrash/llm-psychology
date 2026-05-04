# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_4_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 4
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 52.500000 |
| 34 | 44.250000 |
| 33 | 39.000000 |
| 32 | 33.500000 |
| 31 | 29.375000 |
| 30 | 25.750000 |
| 29 | 20.875000 |
| 28 | 18.125000 |
| 27 | 15.875000 |
| 26 | 13.812500 |

## Oracle True Layers

None