# Experiment Layer Sweep Analysis

- experiment: llama_bbh_5_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 5
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.593750 |
| 30 | 1.343750 |
| 29 | 1.203125 |
| 28 | 1.085938 |
| 27 | 1.007812 |
| 26 | 0.925781 |
| 25 | 0.851562 |
| 24 | 0.792969 |
| 23 | 0.734375 |
| 22 | 0.691406 |

## Oracle True Layers

None