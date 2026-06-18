# Experiment Layer Sweep Analysis

- experiment: llama_bbh_5_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 5
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.937500 |
| 30 | 0.824219 |
| 29 | 0.734375 |
| 28 | 0.664062 |
| 27 | 0.617188 |
| 26 | 0.582031 |
| 25 | 0.546875 |
| 24 | 0.515625 |
| 23 | 0.488281 |
| 22 | 0.462891 |

## Oracle True Layers

None