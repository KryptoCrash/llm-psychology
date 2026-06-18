# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_qd_base.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: qd
- prompt_style: base
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 0.384766 |
| 34 | 0.302734 |
| 33 | 0.267578 |
| 32 | 0.240234 |
| 31 | 0.227539 |
| 30 | 0.213867 |
| 29 | 0.154297 |
| 28 | 0.127930 |
| 27 | 0.119141 |
| 26 | 0.098145 |

## Oracle True Layers

None