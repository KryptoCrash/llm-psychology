# Aggregated Layer Sweep Analysis

- model: qwen
- model_name: Qwen/Qwen3-8B
- dataset: bbh
- source_experiment_count: 24
- sample_count: 2400
- position: assistant_start
- activation_kind: residual

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 35 | 32.750000 |
| 34 | 19.875000 |
| 33 | 18.250000 |
| 32 | 16.875000 |
| 31 | 15.812500 |
| 30 | 14.687500 |
| 29 | 13.625000 |
| 28 | 12.312500 |
| 27 | 10.625000 |
| 26 | 9.687500 |

## Oracle True Layers

None

## Source Experiments

| file | dataset | mode | prompt_style | n | samples |
|---|---|---|---|---:|---:|
| qwen_bbh_10_base_residual.json | bbh | 10 | base | 10 | 100 |
| qwen_bbh_10_explain_residual.json | bbh | 10 | explain | 10 | 100 |
| qwen_bbh_1_base_residual.json | bbh | 1 | base | 1 | 100 |
| qwen_bbh_1_explain_residual.json | bbh | 1 | explain | 1 | 100 |
| qwen_bbh_2_base_residual.json | bbh | 2 | base | 2 | 100 |
| qwen_bbh_2_explain_residual.json | bbh | 2 | explain | 2 | 100 |
| qwen_bbh_3_base_residual.json | bbh | 3 | base | 3 | 100 |
| qwen_bbh_3_explain_residual.json | bbh | 3 | explain | 3 | 100 |
| qwen_bbh_4_base_residual.json | bbh | 4 | base | 4 | 100 |
| qwen_bbh_4_explain_residual.json | bbh | 4 | explain | 4 | 100 |
| qwen_bbh_5_base_residual.json | bbh | 5 | base | 5 | 100 |
| qwen_bbh_5_explain_residual.json | bbh | 5 | explain | 5 | 100 |
| qwen_bbh_6_base_residual.json | bbh | 6 | base | 6 | 100 |
| qwen_bbh_6_explain_residual.json | bbh | 6 | explain | 6 | 100 |
| qwen_bbh_7_base_residual.json | bbh | 7 | base | 7 | 100 |
| qwen_bbh_7_explain_residual.json | bbh | 7 | explain | 7 | 100 |
| qwen_bbh_8_base_residual.json | bbh | 8 | base | 8 | 100 |
| qwen_bbh_8_explain_residual.json | bbh | 8 | explain | 8 | 100 |
| qwen_bbh_9_base_residual.json | bbh | 9 | base | 9 | 100 |
| qwen_bbh_9_explain_residual.json | bbh | 9 | explain | 9 | 100 |
| qwen_bbh_da_base_residual.json | bbh | da | base | 10 | 100 |
| qwen_bbh_da_explain_residual.json | bbh | da | explain | 10 | 100 |
| qwen_bbh_qd_base_residual.json | bbh | qd | base | 10 | 100 |
| qwen_bbh_qd_explain_residual.json | bbh | qd | explain | 10 | 100 |