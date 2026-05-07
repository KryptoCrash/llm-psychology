# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_2_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 2
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.139648 |
| 30 | 0.117676 |
| 29 | 0.103027 |
| 28 | 0.091797 |
| 27 | 0.084961 |
| 26 | 0.078125 |
| 25 | 0.072754 |
| 24 | 0.067871 |
| 23 | 0.063965 |
| 22 | 0.059814 |

## Oracle True Layers

None