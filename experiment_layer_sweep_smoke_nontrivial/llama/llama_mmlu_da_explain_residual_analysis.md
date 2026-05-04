# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_da_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: da
- prompt_style: explain
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.007812 |
| 30 | 0.871094 |
| 29 | 0.765625 |
| 28 | 0.707031 |
| 27 | 0.656250 |
| 26 | 0.617188 |
| 25 | 0.578125 |
| 24 | 0.546875 |
| 23 | 0.515625 |
| 22 | 0.488281 |

## Oracle True Layers

None