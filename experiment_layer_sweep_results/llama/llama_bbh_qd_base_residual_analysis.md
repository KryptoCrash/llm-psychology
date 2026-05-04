# Experiment Layer Sweep Analysis

- experiment: llama_bbh_qd_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: qd
- prompt_style: base
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.308594 |
| 30 | 0.248047 |
| 29 | 0.199219 |
| 28 | 0.170898 |
| 27 | 0.145508 |
| 26 | 0.122070 |
| 25 | 0.101074 |
| 24 | 0.085938 |
| 23 | 0.072754 |
| 22 | 0.060303 |

## Oracle True Layers

None