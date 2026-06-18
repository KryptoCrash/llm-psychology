# Experiment Layer Sweep Analysis

- experiment: llama_bbh_9_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: 9
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.218750 |
| 30 | 1.898438 |
| 29 | 1.703125 |
| 28 | 1.531250 |
| 27 | 1.429688 |
| 26 | 1.312500 |
| 25 | 1.210938 |
| 24 | 1.117188 |
| 23 | 1.046875 |
| 22 | 0.972656 |

## Oracle True Layers

None