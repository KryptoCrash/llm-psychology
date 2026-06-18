# Experiment Layer Sweep Analysis

- experiment: llama_bbh_qd_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- mode: qd
- prompt_style: base
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 3.765625 |
| 30 | 3.203125 |
| 29 | 2.843750 |
| 28 | 2.500000 |
| 27 | 2.343750 |
| 26 | 2.218750 |
| 25 | 2.078125 |
| 24 | 1.937500 |
| 23 | 1.835938 |
| 22 | 1.734375 |

## Oracle True Layers

None