# Aggregated Layer Sweep Analysis

- model: llama
- model_name: meta-llama/Llama-3.1-8B-Instruct
- dataset: bbh
- source_experiment_count: 24
- sample_count: 2400
- position: last_prompt_token
- activation_kind: residual

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.777344 |
| 30 | 0.675781 |
| 29 | 0.605469 |
| 28 | 0.550781 |
| 27 | 0.515625 |
| 26 | 0.472656 |
| 25 | 0.445312 |
| 24 | 0.416016 |
| 23 | 0.390625 |
| 22 | 0.371094 |

## Oracle True Layers

None

## Source Experiments

| file | dataset | mode | prompt_style | n | samples |
|---|---|---|---|---:|---:|
| llama_bbh_10_base_residual.json | bbh | 10 | base | 10 | 100 |
| llama_bbh_10_explain_residual.json | bbh | 10 | explain | 10 | 100 |
| llama_bbh_1_base_residual.json | bbh | 1 | base | 1 | 100 |
| llama_bbh_1_explain_residual.json | bbh | 1 | explain | 1 | 100 |
| llama_bbh_2_base_residual.json | bbh | 2 | base | 2 | 100 |
| llama_bbh_2_explain_residual.json | bbh | 2 | explain | 2 | 100 |
| llama_bbh_3_base_residual.json | bbh | 3 | base | 3 | 100 |
| llama_bbh_3_explain_residual.json | bbh | 3 | explain | 3 | 100 |
| llama_bbh_4_base_residual.json | bbh | 4 | base | 4 | 100 |
| llama_bbh_4_explain_residual.json | bbh | 4 | explain | 4 | 100 |
| llama_bbh_5_base_residual.json | bbh | 5 | base | 5 | 100 |
| llama_bbh_5_explain_residual.json | bbh | 5 | explain | 5 | 100 |
| llama_bbh_6_base_residual.json | bbh | 6 | base | 6 | 100 |
| llama_bbh_6_explain_residual.json | bbh | 6 | explain | 6 | 100 |
| llama_bbh_7_base_residual.json | bbh | 7 | base | 7 | 100 |
| llama_bbh_7_explain_residual.json | bbh | 7 | explain | 7 | 100 |
| llama_bbh_8_base_residual.json | bbh | 8 | base | 8 | 100 |
| llama_bbh_8_explain_residual.json | bbh | 8 | explain | 8 | 100 |
| llama_bbh_9_base_residual.json | bbh | 9 | base | 9 | 100 |
| llama_bbh_9_explain_residual.json | bbh | 9 | explain | 9 | 100 |
| llama_bbh_da_base_residual.json | bbh | da | base | 10 | 100 |
| llama_bbh_da_explain_residual.json | bbh | da | explain | 10 | 100 |
| llama_bbh_qd_base_residual.json | bbh | qd | base | 10 | 100 |
| llama_bbh_qd_explain_residual.json | bbh | qd | explain | 10 | 100 |