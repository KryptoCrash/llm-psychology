# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_qd_base.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: qd
- prompt_style: base
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.235352 |
| 30 | 0.192383 |
| 29 | 0.155273 |
| 28 | 0.134766 |
| 27 | 0.113770 |
| 26 | 0.095215 |
| 25 | 0.079590 |
| 24 | 0.067383 |
| 23 | 0.057373 |
| 22 | 0.047363 |

## Oracle True Layers

None