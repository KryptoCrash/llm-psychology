# Experiment Layer Sweep Analysis

- experiment: qwen_mmlu_2_explain.json
- model: Qwen/Qwen3-8B
- dataset: mmlu
- mode: 2
- prompt_style: explain
- sample_count: 100
- position: last_prompt_token
- baseline: random_answers

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 5.937500 |
| 34 | 4.937500 |
| 33 | 4.218750 |
| 32 | 3.546875 |
| 31 | 3.078125 |
| 30 | 2.593750 |
| 29 | 2.156250 |
| 28 | 1.960938 |
| 27 | 1.726562 |
| 26 | 1.460938 |

## Oracle True Layers

None