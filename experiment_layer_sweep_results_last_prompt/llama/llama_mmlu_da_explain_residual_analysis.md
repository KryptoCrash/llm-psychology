# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_da_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: da
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.800781 |
| 30 | 0.699219 |
| 29 | 0.621094 |
| 28 | 0.574219 |
| 27 | 0.535156 |
| 26 | 0.503906 |
| 25 | 0.472656 |
| 24 | 0.447266 |
| 23 | 0.425781 |
| 22 | 0.404297 |

## Oracle True Layers

None