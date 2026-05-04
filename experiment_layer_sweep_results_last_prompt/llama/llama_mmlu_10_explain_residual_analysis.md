# Experiment Layer Sweep Analysis

- experiment: llama_mmlu_10_explain.json
- model: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- mode: 10
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 1.289062 |
| 30 | 1.117188 |
| 29 | 0.992188 |
| 28 | 0.921875 |
| 27 | 0.863281 |
| 26 | 0.804688 |
| 25 | 0.746094 |
| 24 | 0.707031 |
| 23 | 0.675781 |
| 22 | 0.640625 |

## Oracle True Layers

None