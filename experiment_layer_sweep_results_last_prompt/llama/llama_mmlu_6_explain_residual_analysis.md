# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_6_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 6
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.179688 |
| 30 | 1.023438 |
| 29 | 0.914062 |
| 28 | 0.835938 |
| 27 | 0.785156 |
| 26 | 0.742188 |
| 25 | 0.699219 |
| 24 | 0.656250 |
| 23 | 0.621094 |
| 22 | 0.593750 |

## Oracle True Layers

None