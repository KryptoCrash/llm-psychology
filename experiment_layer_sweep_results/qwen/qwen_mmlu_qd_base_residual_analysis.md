# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_qd_base.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: qd
- prompt_style: base
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 0.695312 |
| 34 | 0.486328 |
| 33 | 0.431641 |
| 32 | 0.400391 |
| 31 | 0.380859 |
| 30 | 0.357422 |
| 29 | 0.275391 |
| 28 | 0.232422 |
| 27 | 0.218750 |
| 26 | 0.178711 |

## Oracle True Layers

None