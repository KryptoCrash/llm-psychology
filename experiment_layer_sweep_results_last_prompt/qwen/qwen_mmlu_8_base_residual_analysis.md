# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_8_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 8
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 78.000000 |
| 34 | 66.500000 |
| 33 | 57.250000 |
| 32 | 48.750000 |
| 31 | 42.750000 |
| 30 | 37.750000 |
| 29 | 30.750000 |
| 28 | 26.750000 |
| 27 | 23.000000 |
| 26 | 20.125000 |

## Oracle True Layers

None