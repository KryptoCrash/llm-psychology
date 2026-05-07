# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_qd_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: qd
- prompt_style: explain
- sample_count: 100
- position: assistant_start_of_turn_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.166016 |
| 30 | 0.135742 |
| 29 | 0.109863 |
| 28 | 0.095215 |
| 27 | 0.080078 |
| 26 | 0.067383 |
| 25 | 0.056152 |
| 24 | 0.047607 |
| 23 | 0.040527 |
| 22 | 0.033447 |

## Oracle True Layers

None