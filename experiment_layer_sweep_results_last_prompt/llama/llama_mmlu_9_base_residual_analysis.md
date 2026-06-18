# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_9_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 9
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.359375 |
| 30 | 1.992188 |
| 29 | 1.789062 |
| 28 | 1.632812 |
| 27 | 1.515625 |
| 26 | 1.398438 |
| 25 | 1.281250 |
| 24 | 1.195312 |
| 23 | 1.101562 |
| 22 | 1.031250 |

## Oracle True Layers

None