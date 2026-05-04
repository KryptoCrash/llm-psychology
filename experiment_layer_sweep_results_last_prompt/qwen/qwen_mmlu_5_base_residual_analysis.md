# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_5_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 5
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 60.750000 |
| 34 | 51.500000 |
| 33 | 45.250000 |
| 32 | 38.750000 |
| 31 | 34.000000 |
| 30 | 30.125000 |
| 29 | 24.250000 |
| 28 | 21.500000 |
| 27 | 18.500000 |
| 26 | 16.250000 |

## Oracle True Layers

None