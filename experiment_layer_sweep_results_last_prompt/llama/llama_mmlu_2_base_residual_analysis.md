# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_2_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 2
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.369141 |
| 30 | 0.306641 |
| 29 | 0.267578 |
| 28 | 0.243164 |
| 27 | 0.221680 |
| 26 | 0.204102 |
| 25 | 0.188477 |
| 24 | 0.172852 |
| 23 | 0.159180 |
| 22 | 0.145508 |

## Oracle True Layers

None