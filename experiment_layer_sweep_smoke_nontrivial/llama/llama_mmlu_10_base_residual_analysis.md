# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_10_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 10
- prompt_style: base
- sample_count: 10
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.453125 |
| 30 | 2.125000 |
| 29 | 1.921875 |
| 28 | 1.757812 |
| 27 | 1.625000 |
| 26 | 1.523438 |
| 25 | 1.398438 |
| 24 | 1.296875 |
| 23 | 1.203125 |
| 22 | 1.109375 |

## Oracle True Layers

None