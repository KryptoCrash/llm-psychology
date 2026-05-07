# Experiment Layer Sweep Analysis

- experiment: llama_bbh_2_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 2
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.172852 |
| 30 | 0.149414 |
| 29 | 0.132812 |
| 28 | 0.120605 |
| 27 | 0.113281 |
| 26 | 0.108398 |
| 25 | 0.098633 |
| 24 | 0.091309 |
| 23 | 0.086914 |
| 22 | 0.081055 |

## Oracle True Layers

None