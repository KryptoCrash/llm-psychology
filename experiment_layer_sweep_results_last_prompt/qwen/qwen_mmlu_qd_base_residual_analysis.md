# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_qd_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: qd
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 118.000000 |
| 34 | 99.000000 |
| 33 | 86.000000 |
| 32 | 75.000000 |
| 31 | 63.750000 |
| 30 | 55.500000 |
| 29 | 45.500000 |
| 28 | 41.000000 |
| 27 | 35.250000 |
| 26 | 31.000000 |

## Oracle True Layers

None