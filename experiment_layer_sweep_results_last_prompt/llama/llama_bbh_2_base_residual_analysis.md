# Experiment Layer Sweep Analysis

- experiment: llama_bbh_2_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 2
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.351562 |
| 30 | 0.292969 |
| 29 | 0.261719 |
| 28 | 0.238281 |
| 27 | 0.220703 |
| 26 | 0.205078 |
| 25 | 0.190430 |
| 24 | 0.178711 |
| 23 | 0.165039 |
| 22 | 0.154297 |

## Oracle True Layers

None