# Experiment Layer Sweep Analysis

- experiment: llama_bbh_8_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 8
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.203125 |
| 30 | 1.859375 |
| 29 | 1.664062 |
| 28 | 1.484375 |
| 27 | 1.390625 |
| 26 | 1.265625 |
| 25 | 1.164062 |
| 24 | 1.085938 |
| 23 | 1.015625 |
| 22 | 0.945312 |

## Oracle True Layers

None