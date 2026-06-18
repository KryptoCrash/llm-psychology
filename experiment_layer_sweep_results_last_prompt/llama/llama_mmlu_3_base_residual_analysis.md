# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_3_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 3
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.007812 |
| 30 | 0.859375 |
| 29 | 0.777344 |
| 28 | 0.714844 |
| 27 | 0.660156 |
| 26 | 0.605469 |
| 25 | 0.558594 |
| 24 | 0.515625 |
| 23 | 0.472656 |
| 22 | 0.437500 |

## Oracle True Layers

None