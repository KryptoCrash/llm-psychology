# Experiment Layer Sweep Analysis

- experiment: llama_bbh_3_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 3
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.921875 |
| 30 | 0.800781 |
| 29 | 0.722656 |
| 28 | 0.660156 |
| 27 | 0.613281 |
| 26 | 0.562500 |
| 25 | 0.519531 |
| 24 | 0.482422 |
| 23 | 0.449219 |
| 22 | 0.425781 |

## Oracle True Layers

None