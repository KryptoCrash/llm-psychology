# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_qd_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 0.347656 |
| 34 | 0.243164 |
| 33 | 0.215820 |
| 32 | 0.200195 |
| 31 | 0.190430 |
| 30 | 0.178711 |
| 29 | 0.137695 |
| 28 | 0.116211 |
| 27 | 0.109375 |
| 26 | 0.089355 |

## Oracle True Layers

None