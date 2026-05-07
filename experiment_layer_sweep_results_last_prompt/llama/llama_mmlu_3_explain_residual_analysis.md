# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_3_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 3
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.691406 |
| 30 | 0.609375 |
| 29 | 0.535156 |
| 28 | 0.494141 |
| 27 | 0.468750 |
| 26 | 0.443359 |
| 25 | 0.416016 |
| 24 | 0.390625 |
| 23 | 0.369141 |
| 22 | 0.353516 |

## Oracle True Layers

None