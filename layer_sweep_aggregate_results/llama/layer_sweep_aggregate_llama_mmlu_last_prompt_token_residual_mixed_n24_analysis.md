# Aggregated Layer Sweep Analysis

- model: llama
- model_name: meta-llama/Llama-3.1-8B-Instruct
- dataset: mmlu
- source_experiment_count: 24
- sample_count: 2400
- position: last_prompt_token
- activation_kind: residual

## Largest Diffmean Norms

| layer | norm |
|---:|---:|
| 31 | 0.953125 |
| 30 | 0.816406 |
| 29 | 0.730469 |
| 28 | 0.671875 |
| 27 | 0.628906 |
| 26 | 0.578125 |
| 25 | 0.539062 |
| 24 | 0.503906 |
| 23 | 0.470703 |
| 22 | 0.443359 |

## Oracle True Layers

None

## Source Experiments

| file | dataset | mode | prompt_style | n | samples |
|---|---|---|---|---:|---:|
| llama_mmlu_10_base_residual.json | mmlu | 10 | base | 10 | 100 |
| llama_mmlu_10_explain_residual.json | mmlu | 10 | explain | 10 | 100 |
| llama_mmlu_1_base_residual.json | mmlu | 1 | base | 1 | 100 |
| llama_mmlu_1_explain_residual.json | mmlu | 1 | explain | 1 | 100 |
| llama_mmlu_2_base_residual.json | mmlu | 2 | base | 2 | 100 |
| llama_mmlu_2_explain_residual.json | mmlu | 2 | explain | 2 | 100 |
| llama_mmlu_3_base_residual.json | mmlu | 3 | base | 3 | 100 |
| llama_mmlu_3_explain_residual.json | mmlu | 3 | explain | 3 | 100 |
| llama_mmlu_4_base_residual.json | mmlu | 4 | base | 4 | 100 |
| llama_mmlu_4_explain_residual.json | mmlu | 4 | explain | 4 | 100 |
| llama_mmlu_5_base_residual.json | mmlu | 5 | base | 5 | 100 |
| llama_mmlu_5_explain_residual.json | mmlu | 5 | explain | 5 | 100 |
| llama_mmlu_6_base_residual.json | mmlu | 6 | base | 6 | 100 |
| llama_mmlu_6_explain_residual.json | mmlu | 6 | explain | 6 | 100 |
| llama_mmlu_7_base_residual.json | mmlu | 7 | base | 7 | 100 |
| llama_mmlu_7_explain_residual.json | mmlu | 7 | explain | 7 | 100 |
| llama_mmlu_8_base_residual.json | mmlu | 8 | base | 8 | 100 |
| llama_mmlu_8_explain_residual.json | mmlu | 8 | explain | 8 | 100 |
| llama_mmlu_9_base_residual.json | mmlu | 9 | base | 9 | 100 |
| llama_mmlu_9_explain_residual.json | mmlu | 9 | explain | 9 | 100 |
| llama_mmlu_da_base_residual.json | mmlu | da | base | 10 | 100 |
| llama_mmlu_da_explain_residual.json | mmlu | da | explain | 10 | 100 |
| llama_mmlu_qd_base_residual.json | mmlu | qd | base | 10 | 100 |
| llama_mmlu_qd_explain_residual.json | mmlu | qd | explain | 10 | 100 |