# Qwen Experiment Mean Activation Projection

- reference: `layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt`
- reference layer: 23
- reference norm: 35.019440
- projected activation: pure mean residual activation at `assistant_start`
- experiment count: 48

| rank | experiment | dataset | mode | prompt | projection | cosine | mean_norm | n |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | qwen_mmlu_8_base.json | mmlu | 8 | base | 12.712745 | 0.088022 | 144.426682 | 100 |
| 2 | qwen_mmlu_6_base.json | mmlu | 6 | base | 3.648328 | 0.025147 | 145.080978 | 100 |
| 3 | qwen_mmlu_7_base.json | mmlu | 7 | base | 3.426757 | 0.023760 | 144.222198 | 100 |
| 4 | qwen_mmlu_5_base.json | mmlu | 5 | base | 0.085804 | 0.000590 | 145.521774 | 100 |
| 5 | qwen_mmlu_10_base.json | mmlu | 10 | base | -0.474979 | -0.003300 | 143.932800 | 100 |
| 6 | qwen_mmlu_9_base.json | mmlu | 9 | base | -0.838056 | -0.005815 | 144.130066 | 100 |
| 7 | qwen_mmlu_4_base.json | mmlu | 4 | base | -3.143245 | -0.021262 | 147.833679 | 100 |
| 8 | qwen_mmlu_8_explain.json | mmlu | 8 | explain | -7.033450 | -0.048562 | 144.834351 | 100 |
| 9 | qwen_mmlu_10_explain.json | mmlu | 10 | explain | -7.126652 | -0.049387 | 144.302689 | 100 |
| 10 | qwen_mmlu_3_base.json | mmlu | 3 | base | -8.010196 | -0.053778 | 148.949387 | 100 |
| 11 | qwen_bbh_qd_base.json | bbh | qd | base | -8.398162 | -0.057151 | 146.946838 | 100 |
| 12 | qwen_mmlu_6_explain.json | mmlu | 6 | explain | -8.845470 | -0.060834 | 145.402512 | 100 |
| 13 | qwen_mmlu_da_base.json | mmlu | da | base | -9.227221 | -0.063669 | 144.925766 | 100 |
| 14 | qwen_bbh_1_explain.json | bbh | 1 | explain | -9.387793 | -0.063923 | 146.861542 | 100 |
| 15 | qwen_bbh_5_base.json | bbh | 5 | base | -9.738829 | -0.068518 | 142.135925 | 100 |
| 16 | qwen_mmlu_qd_base.json | mmlu | qd | base | -9.986942 | -0.068144 | 146.556656 | 100 |
| 17 | qwen_bbh_6_base.json | bbh | 6 | base | -10.145071 | -0.071036 | 142.816254 | 100 |
| 18 | qwen_bbh_4_base.json | bbh | 4 | base | -10.246803 | -0.071143 | 144.030884 | 100 |
| 19 | qwen_bbh_1_base.json | bbh | 1 | base | -10.329516 | -0.070043 | 147.474365 | 100 |
| 20 | qwen_mmlu_9_explain.json | mmlu | 9 | explain | -11.143261 | -0.077316 | 144.126801 | 100 |
| 21 | qwen_mmlu_qd_explain.json | mmlu | qd | explain | -11.479313 | -0.078977 | 145.349564 | 100 |
| 22 | qwen_bbh_7_base.json | bbh | 7 | base | -11.646894 | -0.082957 | 140.397354 | 100 |
| 23 | qwen_bbh_9_base.json | bbh | 9 | base | -11.697704 | -0.081472 | 143.578568 | 100 |
| 24 | qwen_mmlu_5_explain.json | mmlu | 5 | explain | -12.815044 | -0.087994 | 145.635544 | 100 |
| 25 | qwen_bbh_8_base.json | bbh | 8 | base | -12.888247 | -0.089998 | 143.205399 | 100 |
| 26 | qwen_bbh_3_base.json | bbh | 3 | base | -13.094507 | -0.089708 | 145.967316 | 100 |
| 27 | qwen_bbh_7_explain.json | bbh | 7 | explain | -13.308381 | -0.094152 | 141.350281 | 100 |
| 28 | qwen_bbh_9_explain.json | bbh | 9 | explain | -13.569474 | -0.094768 | 143.185974 | 100 |
| 29 | qwen_bbh_6_explain.json | bbh | 6 | explain | -13.620403 | -0.094978 | 143.406036 | 100 |
| 30 | qwen_bbh_8_explain.json | bbh | 8 | explain | -13.809212 | -0.096439 | 143.190598 | 100 |
| 31 | qwen_bbh_10_explain.json | bbh | 10 | explain | -13.987718 | -0.097160 | 143.965729 | 100 |
| 32 | qwen_mmlu_1_explain.json | mmlu | 1 | explain | -14.014750 | -0.094929 | 147.634583 | 100 |
| 33 | qwen_bbh_10_base.json | bbh | 10 | base | -14.199415 | -0.098764 | 143.770462 | 100 |
| 34 | qwen_bbh_5_explain.json | bbh | 5 | explain | -14.380419 | -0.100334 | 143.326050 | 100 |
| 35 | qwen_bbh_qd_explain.json | bbh | qd | explain | -14.466408 | -0.097457 | 148.439285 | 100 |
| 36 | qwen_mmlu_1_base.json | mmlu | 1 | base | -14.527880 | -0.098688 | 147.209686 | 100 |
| 37 | qwen_bbh_4_explain.json | bbh | 4 | explain | -14.988424 | -0.103397 | 144.959579 | 100 |
| 38 | qwen_mmlu_7_explain.json | mmlu | 7 | explain | -15.849955 | -0.109645 | 144.557419 | 100 |
| 39 | qwen_mmlu_da_explain.json | mmlu | da | explain | -15.936293 | -0.109755 | 145.199173 | 100 |
| 40 | qwen_bbh_da_explain.json | bbh | da | explain | -16.199024 | -0.111627 | 145.117218 | 100 |
| 41 | qwen_bbh_da_base.json | bbh | da | base | -16.938705 | -0.116683 | 145.168259 | 100 |
| 42 | qwen_mmlu_4_explain.json | mmlu | 4 | explain | -17.106350 | -0.115691 | 147.861832 | 100 |
| 43 | qwen_bbh_2_base.json | bbh | 2 | base | -17.600716 | -0.118315 | 148.761703 | 100 |
| 44 | qwen_bbh_3_explain.json | bbh | 3 | explain | -17.673748 | -0.120164 | 147.079666 | 100 |
| 45 | qwen_mmlu_3_explain.json | mmlu | 3 | explain | -18.517475 | -0.123811 | 149.562088 | 100 |
| 46 | qwen_bbh_2_explain.json | bbh | 2 | explain | -18.893188 | -0.126732 | 149.080429 | 100 |
| 47 | qwen_mmlu_2_base.json | mmlu | 2 | base | -23.119492 | -0.152737 | 151.367981 | 100 |
| 48 | qwen_mmlu_2_explain.json | mmlu | 2 | explain | -24.864817 | -0.166544 | 149.299210 | 100 |