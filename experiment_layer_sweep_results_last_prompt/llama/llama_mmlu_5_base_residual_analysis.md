# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_5_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 5
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.960938 |
| 30 | 1.648438 |
| 29 | 1.484375 |
| 28 | 1.375000 |
| 27 | 1.281250 |
| 26 | 1.179688 |
| 25 | 1.085938 |
| 24 | 1.007812 |
| 23 | 0.933594 |
| 22 | 0.871094 |

## Oracle True Layers

None