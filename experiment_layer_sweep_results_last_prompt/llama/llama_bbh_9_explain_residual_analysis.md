# Experiment Layer Sweep Analysis

- experiment: llama_bbh_9_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 9
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.062500 |
| 30 | 0.957031 |
| 29 | 0.855469 |
| 28 | 0.789062 |
| 27 | 0.734375 |
| 26 | 0.695312 |
| 25 | 0.656250 |
| 24 | 0.621094 |
| 23 | 0.597656 |
| 22 | 0.566406 |

## Oracle True Layers

None