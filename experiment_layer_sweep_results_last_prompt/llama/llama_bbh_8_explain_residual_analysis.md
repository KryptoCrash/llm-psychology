# Experiment Layer Sweep Analysis

- experiment: llama_bbh_8_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 8
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.046875 |
| 30 | 0.941406 |
| 29 | 0.847656 |
| 28 | 0.765625 |
| 27 | 0.710938 |
| 26 | 0.675781 |
| 25 | 0.636719 |
| 24 | 0.597656 |
| 23 | 0.570312 |
| 22 | 0.542969 |

## Oracle True Layers

None