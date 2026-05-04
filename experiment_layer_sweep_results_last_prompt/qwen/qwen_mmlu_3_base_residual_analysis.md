# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_3_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 3
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 32.500000 |
| 34 | 27.500000 |
| 33 | 24.000000 |
| 32 | 20.875000 |
| 31 | 17.875000 |
| 30 | 15.437500 |
| 29 | 12.687500 |
| 28 | 11.062500 |
| 27 | 9.500000 |
| 26 | 8.312500 |

## Oracle True Layers

None