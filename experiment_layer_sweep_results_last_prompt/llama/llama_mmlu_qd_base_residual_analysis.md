# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_qd_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: qd
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 5.593750 |
| 30 | 4.812500 |
| 29 | 4.250000 |
| 28 | 3.781250 |
| 27 | 3.531250 |
| 26 | 3.281250 |
| 25 | 3.062500 |
| 24 | 2.843750 |
| 23 | 2.687500 |
| 22 | 2.546875 |

## Oracle True Layers

None