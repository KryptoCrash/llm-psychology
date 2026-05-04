# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_4_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 4
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.902344 |
| 30 | 0.796875 |
| 29 | 0.703125 |
| 28 | 0.648438 |
| 27 | 0.613281 |
| 26 | 0.578125 |
| 25 | 0.546875 |
| 24 | 0.515625 |
| 23 | 0.488281 |
| 22 | 0.460938 |

## Oracle True Layers

None