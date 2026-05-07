# Experiment Layer Sweep Analysis

- experiment: llama_bbh_6_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 6
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.062500 |
| 30 | 0.933594 |
| 29 | 0.828125 |
| 28 | 0.750000 |
| 27 | 0.695312 |
| 26 | 0.652344 |
| 25 | 0.613281 |
| 24 | 0.578125 |
| 23 | 0.550781 |
| 22 | 0.519531 |

## Oracle True Layers

None