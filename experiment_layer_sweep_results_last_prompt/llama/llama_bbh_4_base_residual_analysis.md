# Experiment Layer Sweep Analysis

- experiment: llama_bbh_4_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 4
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.445312 |
| 30 | 1.226562 |
| 29 | 1.101562 |
| 28 | 0.984375 |
| 27 | 0.914062 |
| 26 | 0.832031 |
| 25 | 0.765625 |
| 24 | 0.707031 |
| 23 | 0.660156 |
| 22 | 0.621094 |

## Oracle True Layers

None