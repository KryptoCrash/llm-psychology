# Experiment Layer Sweep Analysis

- experiment: llama_bbh_da_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: da
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.679688 |
| 30 | 0.605469 |
| 29 | 0.546875 |
| 28 | 0.507812 |
| 27 | 0.478516 |
| 26 | 0.458984 |
| 25 | 0.433594 |
| 24 | 0.414062 |
| 23 | 0.396484 |
| 22 | 0.375000 |

## Oracle True Layers

None