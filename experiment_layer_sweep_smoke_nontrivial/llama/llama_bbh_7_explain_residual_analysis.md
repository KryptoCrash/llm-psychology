# Experiment Layer Sweep Analysis

- experiment: llama_bbh_7_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 7
- prompt_style: explain
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.109375 |
| 30 | 0.941406 |
| 29 | 0.835938 |
| 28 | 0.777344 |
| 27 | 0.734375 |
| 26 | 0.679688 |
| 25 | 0.640625 |
| 24 | 0.601562 |
| 23 | 0.566406 |
| 22 | 0.531250 |

## Oracle True Layers

None