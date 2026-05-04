# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_5_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 5
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.023438 |
| 30 | 0.902344 |
| 29 | 0.792969 |
| 28 | 0.722656 |
| 27 | 0.679688 |
| 26 | 0.644531 |
| 25 | 0.609375 |
| 24 | 0.574219 |
| 23 | 0.542969 |
| 22 | 0.515625 |

## Oracle True Layers

None