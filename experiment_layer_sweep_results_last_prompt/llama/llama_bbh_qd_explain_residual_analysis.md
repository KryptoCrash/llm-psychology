# Experiment Layer Sweep Analysis

- experiment: llama_bbh_qd_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.781250 |
| 30 | 2.468750 |
| 29 | 2.234375 |
| 28 | 2.062500 |
| 27 | 1.937500 |
| 26 | 1.843750 |
| 25 | 1.726562 |
| 24 | 1.609375 |
| 23 | 1.515625 |
| 22 | 1.421875 |

## Oracle True Layers

None