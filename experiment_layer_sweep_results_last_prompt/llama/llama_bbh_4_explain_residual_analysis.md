# Experiment Layer Sweep Analysis

- experiment: llama_bbh_4_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 4
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.773438 |
| 30 | 0.675781 |
| 29 | 0.601562 |
| 28 | 0.550781 |
| 27 | 0.511719 |
| 26 | 0.482422 |
| 25 | 0.453125 |
| 24 | 0.427734 |
| 23 | 0.404297 |
| 22 | 0.382812 |

## Oracle True Layers

None