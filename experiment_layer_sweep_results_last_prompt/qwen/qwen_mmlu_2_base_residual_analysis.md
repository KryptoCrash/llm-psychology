# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_2_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 2
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 14.375000 |
| 34 | 12.625000 |
| 33 | 10.500000 |
| 32 | 9.187500 |
| 31 | 7.968750 |
| 30 | 6.937500 |
| 29 | 5.750000 |
| 28 | 5.031250 |
| 27 | 4.468750 |
| 26 | 3.828125 |

## Oracle True Layers

None