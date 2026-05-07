# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_8_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 8
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 2.281250 |
| 30 | 1.914062 |
| 29 | 1.710938 |
| 28 | 1.570312 |
| 27 | 1.460938 |
| 26 | 1.351562 |
| 25 | 1.250000 |
| 24 | 1.164062 |
| 23 | 1.078125 |
| 22 | 1.000000 |

## Oracle True Layers

None