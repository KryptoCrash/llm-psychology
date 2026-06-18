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
| 35 | 1.078125 |
| 34 | 0.574219 |
| 33 | 0.443359 |
| 32 | 0.345703 |
| 31 | 0.291016 |
| 30 | 0.238281 |
| 29 | 0.202148 |
| 28 | 0.175781 |
| 27 | 0.147461 |
| 26 | 0.125000 |

## Oracle True Layers

None