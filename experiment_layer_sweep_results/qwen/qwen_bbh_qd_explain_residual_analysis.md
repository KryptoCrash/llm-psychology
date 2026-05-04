# Experiment Layer Sweep Analysis

- experiment: qwen_bbh_qd_explain.json
- model: Qwen/Qwen3-8B
- dataset: bbh
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 0.714844 |
| 34 | 0.621094 |
| 33 | 0.535156 |
| 32 | 0.482422 |
| 31 | 0.449219 |
| 30 | 0.402344 |
| 29 | 0.347656 |
| 28 | 0.267578 |
| 27 | 0.245117 |
| 26 | 0.222656 |

## Oracle True Layers

None