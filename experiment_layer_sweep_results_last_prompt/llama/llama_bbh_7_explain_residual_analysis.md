# Experiment Layer Sweep Analysis

- experiment: llama_bbh_7_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 7
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.007812 |
| 30 | 0.910156 |
| 29 | 0.816406 |
| 28 | 0.738281 |
| 27 | 0.679688 |
| 26 | 0.648438 |
| 25 | 0.609375 |
| 24 | 0.578125 |
| 23 | 0.546875 |
| 22 | 0.519531 |

## Oracle True Layers

None