# Experiment Layer Sweep Analysis

- experiment: llama_bbh_6_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 6
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.875000 |
| 30 | 1.593750 |
| 29 | 1.421875 |
| 28 | 1.273438 |
| 27 | 1.195312 |
| 26 | 1.085938 |
| 25 | 1.007812 |
| 24 | 0.929688 |
| 23 | 0.871094 |
| 22 | 0.816406 |

## Oracle True Layers

None