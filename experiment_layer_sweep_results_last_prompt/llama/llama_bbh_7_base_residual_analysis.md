# Experiment Layer Sweep Analysis

- experiment: llama_bbh_7_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 7
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.109375 |
| 30 | 1.789062 |
| 29 | 1.585938 |
| 28 | 1.429688 |
| 27 | 1.328125 |
| 26 | 1.210938 |
| 25 | 1.117188 |
| 24 | 1.046875 |
| 23 | 0.976562 |
| 22 | 0.914062 |

## Oracle True Layers

None