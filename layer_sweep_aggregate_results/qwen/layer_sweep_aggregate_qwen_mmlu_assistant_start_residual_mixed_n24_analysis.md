# Aggregated Layer Sweep Analysis

- model: qwen
- model_name: Qwen/Qwen3-8B
- dataset: mmlu
- source_experiment_count: 24
- sample_count: 2400
- position: assistant_start
- activation_kind: residual

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 77.000000 |
| 34 | 53.500000 |
| 33 | 49.750000 |
| 32 | 46.500000 |
| 31 | 43.250000 |
| 30 | 40.250000 |
| 29 | 37.000000 |
| 28 | 33.750000 |
| 27 | 29.125000 |
| 26 | 26.000000 |

## Oracle True Layers

None

## Source Experiments

| file | dataset | mode | prompt_style | n | samples |
|---|---|---|---|---:|---:|
| qwen_mmlu_10_base_residual.json | mmlu | 10 | base | 10 | 100 |
| qwen_mmlu_10_explain_residual.json | mmlu | 10 | explain | 10 | 100 |
| qwen_mmlu_1_base_residual.json | mmlu | 1 | base | 1 | 100 |
| qwen_mmlu_1_explain_residual.json | mmlu | 1 | explain | 1 | 100 |
| qwen_mmlu_2_base_residual.json | mmlu | 2 | base | 2 | 100 |
| qwen_mmlu_2_explain_residual.json | mmlu | 2 | explain | 2 | 100 |
| qwen_mmlu_3_base_residual.json | mmlu | 3 | base | 3 | 100 |
| qwen_mmlu_3_explain_residual.json | mmlu | 3 | explain | 3 | 100 |
| qwen_mmlu_4_base_residual.json | mmlu | 4 | base | 4 | 100 |
| qwen_mmlu_4_explain_residual.json | mmlu | 4 | explain | 4 | 100 |
| qwen_mmlu_5_base_residual.json | mmlu | 5 | base | 5 | 100 |
| qwen_mmlu_5_explain_residual.json | mmlu | 5 | explain | 5 | 100 |
| qwen_mmlu_6_base_residual.json | mmlu | 6 | base | 6 | 100 |
| qwen_mmlu_6_explain_residual.json | mmlu | 6 | explain | 6 | 100 |
| qwen_mmlu_7_base_residual.json | mmlu | 7 | base | 7 | 100 |
| qwen_mmlu_7_explain_residual.json | mmlu | 7 | explain | 7 | 100 |
| qwen_mmlu_8_base_residual.json | mmlu | 8 | base | 8 | 100 |
| qwen_mmlu_8_explain_residual.json | mmlu | 8 | explain | 8 | 100 |
| qwen_mmlu_9_base_residual.json | mmlu | 9 | base | 9 | 100 |
| qwen_mmlu_9_explain_residual.json | mmlu | 9 | explain | 9 | 100 |
| qwen_mmlu_da_base_residual.json | mmlu | da | base | 10 | 100 |
| qwen_mmlu_da_explain_residual.json | mmlu | da | explain | 10 | 100 |
| qwen_mmlu_qd_base_residual.json | mmlu | qd | base | 10 | 100 |
| qwen_mmlu_qd_explain_residual.json | mmlu | qd | explain | 10 | 100 |