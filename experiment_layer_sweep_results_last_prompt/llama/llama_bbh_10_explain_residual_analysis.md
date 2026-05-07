# Experiment Layer Sweep Analysis

- experiment: llama_bbh_10_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 10
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.132812 |
| 30 | 1.007812 |
| 29 | 0.902344 |
| 28 | 0.828125 |
| 27 | 0.773438 |
| 26 | 0.738281 |
| 25 | 0.695312 |
| 24 | 0.652344 |
| 23 | 0.628906 |
| 22 | 0.593750 |

## Oracle True Layers

None