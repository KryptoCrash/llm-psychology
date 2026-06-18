# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_da_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: da
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.453125 |
| 30 | 1.226562 |
| 29 | 1.093750 |
| 28 | 1.007812 |
| 27 | 0.933594 |
| 26 | 0.851562 |
| 25 | 0.785156 |
| 24 | 0.734375 |
| 23 | 0.679688 |
| 22 | 0.636719 |

## Oracle True Layers

None