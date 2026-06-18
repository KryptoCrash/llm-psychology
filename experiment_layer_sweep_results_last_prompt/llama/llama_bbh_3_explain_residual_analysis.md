# Experiment Layer Sweep Analysis

- experiment: llama_bbh_3_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 3
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.589844 |
| 30 | 0.515625 |
| 29 | 0.451172 |
| 28 | 0.414062 |
| 27 | 0.388672 |
| 26 | 0.365234 |
| 25 | 0.339844 |
| 24 | 0.316406 |
| 23 | 0.298828 |
| 22 | 0.283203 |

## Oracle True Layers

None