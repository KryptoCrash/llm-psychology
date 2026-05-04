# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_10_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 10
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.218750 |
| 30 | 1.875000 |
| 29 | 1.679688 |
| 28 | 1.546875 |
| 27 | 1.437500 |
| 26 | 1.335938 |
| 25 | 1.226562 |
| 24 | 1.140625 |
| 23 | 1.054688 |
| 22 | 0.988281 |

## Oracle True Layers

None