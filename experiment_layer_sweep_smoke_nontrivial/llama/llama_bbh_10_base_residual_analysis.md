# Experiment Layer Sweep Analysis

- experiment: llama_bbh_10_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 10
- prompt_style: base
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.062500 |
| 30 | 1.789062 |
| 29 | 1.632812 |
| 28 | 1.476562 |
| 27 | 1.375000 |
| 26 | 1.257812 |
| 25 | 1.156250 |
| 24 | 1.078125 |
| 23 | 0.992188 |
| 22 | 0.921875 |

## Oracle True Layers

None