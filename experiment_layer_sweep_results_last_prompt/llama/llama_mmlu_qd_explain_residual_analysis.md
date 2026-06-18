# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_qd_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 3.625000 |
| 30 | 3.109375 |
| 29 | 2.765625 |
| 28 | 2.562500 |
| 27 | 2.421875 |
| 26 | 2.250000 |
| 25 | 2.109375 |
| 24 | 1.960938 |
| 23 | 1.875000 |
| 22 | 1.757812 |

## Oracle True Layers

None