# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_9_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 9
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.250000 |
| 30 | 1.093750 |
| 29 | 0.968750 |
| 28 | 0.894531 |
| 27 | 0.839844 |
| 26 | 0.789062 |
| 25 | 0.738281 |
| 24 | 0.695312 |
| 23 | 0.664062 |
| 22 | 0.632812 |

## Oracle True Layers

None