# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_4_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 4
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.765625 |
| 30 | 1.507812 |
| 29 | 1.359375 |
| 28 | 1.257812 |
| 27 | 1.164062 |
| 26 | 1.070312 |
| 25 | 0.984375 |
| 24 | 0.917969 |
| 23 | 0.855469 |
| 22 | 0.800781 |

## Oracle True Layers

None