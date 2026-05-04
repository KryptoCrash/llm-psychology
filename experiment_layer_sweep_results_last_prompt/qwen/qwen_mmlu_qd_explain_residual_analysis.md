# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_qd_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 88.500000 |
| 34 | 71.500000 |
| 33 | 57.500000 |
| 32 | 51.000000 |
| 31 | 44.000000 |
| 30 | 37.250000 |
| 29 | 32.250000 |
| 28 | 28.125000 |
| 27 | 24.500000 |
| 26 | 21.750000 |

## Oracle True Layers

None