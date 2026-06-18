# Experiment Layer Sweep Analysis

- experiment: llama_bbh_10_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 10
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.109375 |
| 30 | 1.781250 |
| 29 | 1.593750 |
| 28 | 1.445312 |
| 27 | 1.351562 |
| 26 | 1.242188 |
| 25 | 1.148438 |
| 24 | 1.070312 |
| 23 | 0.996094 |
| 22 | 0.933594 |

## Oracle True Layers

None