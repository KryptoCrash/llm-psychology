# Qwen Experiment Mean Activation Projection

- reference: `layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_10_base_residual_assistant_start_all_wrong_minus_random_answers_n10_100_diffmeans.pt`
- reference layer: 23
- reference norm: 26.583673
- projected activation: pure mean residual activation at `assistant_start`
- experiment count: 48

| rank | experiment | dataset | mode | prompt | projection | cosine | mean_norm | n |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | qwen_mmlu_8_base.json | mmlu | 8 | base | 9.412517 | 0.065172 | 144.426682 | 100 |
| 2 | qwen_mmlu_6_base.json | mmlu | 6 | base | 2.380827 | 0.016410 | 145.080978 | 100 |
| 3 | qwen_mmlu_7_base.json | mmlu | 7 | base | 2.358775 | 0.016355 | 144.222198 | 100 |
| 4 | qwen_mmlu_10_base.json | mmlu | 10 | base | 0.865288 | 0.006012 | 143.932800 | 100 |
| 5 | qwen_mmlu_9_base.json | mmlu | 9 | base | -0.645902 | -0.004481 | 144.130066 | 100 |
| 6 | qwen_mmlu_5_base.json | mmlu | 5 | base | -0.772645 | -0.005309 | 145.521774 | 100 |
| 7 | qwen_mmlu_4_base.json | mmlu | 4 | base | -3.253597 | -0.022008 | 147.833679 | 100 |
| 8 | qwen_mmlu_10_explain.json | mmlu | 10 | explain | -5.744494 | -0.039809 | 144.302689 | 100 |
| 9 | qwen_mmlu_8_explain.json | mmlu | 8 | explain | -6.646664 | -0.045891 | 144.834351 | 100 |
| 10 | qwen_mmlu_3_base.json | mmlu | 3 | base | -6.847384 | -0.045971 | 148.949387 | 100 |
| 11 | qwen_mmlu_6_explain.json | mmlu | 6 | explain | -8.072289 | -0.055517 | 145.402512 | 100 |
| 12 | qwen_bbh_qd_base.json | bbh | qd | base | -8.651935 | -0.058878 | 146.946838 | 100 |
| 13 | qwen_mmlu_qd_base.json | mmlu | qd | base | -9.043261 | -0.061705 | 146.556656 | 100 |
| 14 | qwen_mmlu_9_explain.json | mmlu | 9 | explain | -9.286931 | -0.064436 | 144.126801 | 100 |
| 15 | qwen_mmlu_da_base.json | mmlu | da | base | -9.523673 | -0.065714 | 144.925766 | 100 |
| 16 | qwen_mmlu_qd_explain.json | mmlu | qd | explain | -10.118949 | -0.069618 | 145.349564 | 100 |
| 17 | qwen_bbh_5_base.json | bbh | 5 | base | -10.494008 | -0.073831 | 142.135925 | 100 |
| 18 | qwen_bbh_6_base.json | bbh | 6 | base | -10.711159 | -0.075000 | 142.816254 | 100 |
| 19 | qwen_bbh_1_explain.json | bbh | 1 | explain | -10.788561 | -0.073461 | 146.861542 | 100 |
| 20 | qwen_bbh_4_base.json | bbh | 4 | base | -10.962867 | -0.076115 | 144.030884 | 100 |
| 21 | qwen_bbh_7_base.json | bbh | 7 | base | -11.259578 | -0.080198 | 140.397354 | 100 |
| 22 | qwen_bbh_1_base.json | bbh | 1 | base | -11.410757 | -0.077375 | 147.474365 | 100 |
| 23 | qwen_mmlu_5_explain.json | mmlu | 5 | explain | -11.788010 | -0.080942 | 145.635544 | 100 |
| 24 | qwen_bbh_9_base.json | bbh | 9 | base | -12.103592 | -0.084299 | 143.578568 | 100 |
| 25 | qwen_bbh_8_base.json | bbh | 8 | base | -13.151257 | -0.091835 | 143.205399 | 100 |
| 26 | qwen_bbh_7_explain.json | bbh | 7 | explain | -13.282789 | -0.093971 | 141.350281 | 100 |
| 27 | qwen_bbh_3_base.json | bbh | 3 | base | -13.414173 | -0.091898 | 145.967316 | 100 |
| 28 | qwen_mmlu_1_explain.json | mmlu | 1 | explain | -13.672791 | -0.092612 | 147.634583 | 100 |
| 29 | qwen_mmlu_1_base.json | mmlu | 1 | base | -13.916998 | -0.094539 | 147.209686 | 100 |
| 30 | qwen_bbh_6_explain.json | bbh | 6 | explain | -13.957725 | -0.097330 | 143.406036 | 100 |
| 31 | qwen_bbh_9_explain.json | bbh | 9 | explain | -13.965252 | -0.097532 | 143.185974 | 100 |
| 32 | qwen_mmlu_7_explain.json | mmlu | 7 | explain | -14.033230 | -0.097077 | 144.557419 | 100 |
| 33 | qwen_bbh_10_base.json | bbh | 10 | base | -14.050899 | -0.097731 | 143.770462 | 100 |
| 34 | qwen_bbh_8_explain.json | bbh | 8 | explain | -14.128426 | -0.098669 | 143.190598 | 100 |
| 35 | qwen_bbh_10_explain.json | bbh | 10 | explain | -14.274662 | -0.099153 | 143.965729 | 100 |
| 36 | qwen_bbh_qd_explain.json | bbh | qd | explain | -14.590561 | -0.098293 | 148.439285 | 100 |
| 37 | qwen_bbh_5_explain.json | bbh | 5 | explain | -14.607682 | -0.101919 | 143.326050 | 100 |
| 38 | qwen_mmlu_4_explain.json | mmlu | 4 | explain | -14.894738 | -0.100734 | 147.861832 | 100 |
| 39 | qwen_bbh_4_explain.json | bbh | 4 | explain | -15.286944 | -0.105457 | 144.959579 | 100 |
| 40 | qwen_mmlu_da_explain.json | mmlu | da | explain | -15.567509 | -0.107215 | 145.199173 | 100 |
| 41 | qwen_mmlu_3_explain.json | mmlu | 3 | explain | -16.321404 | -0.109128 | 149.562088 | 100 |
| 42 | qwen_bbh_2_base.json | bbh | 2 | base | -17.240328 | -0.115892 | 148.761703 | 100 |
| 43 | qwen_bbh_da_explain.json | bbh | da | explain | -17.335627 | -0.119459 | 145.117218 | 100 |
| 44 | qwen_bbh_da_base.json | bbh | da | base | -17.820234 | -0.122756 | 145.168259 | 100 |
| 45 | qwen_bbh_3_explain.json | bbh | 3 | explain | -17.839396 | -0.121291 | 147.079666 | 100 |
| 46 | qwen_bbh_2_explain.json | bbh | 2 | explain | -18.938168 | -0.127033 | 149.080429 | 100 |
| 47 | qwen_mmlu_2_base.json | mmlu | 2 | base | -20.113785 | -0.132880 | 151.367981 | 100 |
| 48 | qwen_mmlu_2_explain.json | mmlu | 2 | explain | -21.886387 | -0.146594 | 149.299210 | 100 |