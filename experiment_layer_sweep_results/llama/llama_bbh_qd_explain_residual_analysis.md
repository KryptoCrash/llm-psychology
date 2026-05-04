# Experiment Layer Sweep Analysis

- experiment: llama_bbh_qd_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.096680 |
| 30 | 0.079590 |
| 29 | 0.063965 |
| 28 | 0.055420 |
| 27 | 0.046875 |
| 26 | 0.039062 |
| 25 | 0.032715 |
| 24 | 0.027710 |
| 23 | 0.023560 |
| 22 | 0.019531 |

## Oracle True Layers

None