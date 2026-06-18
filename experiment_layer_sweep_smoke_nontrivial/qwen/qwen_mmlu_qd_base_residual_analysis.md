# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_qd_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: qd
- prompt_style: base
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 124.000000 |
| 34 | 106.000000 |
| 33 | 91.500000 |
| 32 | 79.000000 |
| 31 | 68.000000 |
| 30 | 59.500000 |
| 29 | 48.500000 |
| 28 | 44.000000 |
| 27 | 38.000000 |
| 26 | 33.250000 |

## Oracle True Layers

None