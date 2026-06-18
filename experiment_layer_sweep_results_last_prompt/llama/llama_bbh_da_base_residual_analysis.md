# Experiment Layer Sweep Analysis

- experiment: llama_bbh_da_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: da
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.210938 |
| 30 | 1.039062 |
| 29 | 0.914062 |
| 28 | 0.824219 |
| 27 | 0.769531 |
| 26 | 0.710938 |
| 25 | 0.664062 |
| 24 | 0.621094 |
| 23 | 0.578125 |
| 22 | 0.546875 |

## Oracle True Layers

None