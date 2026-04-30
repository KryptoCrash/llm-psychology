# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_delta_all_wrong_minus_one_right_n8_100_diffmeans.pt`
- direction: `all_wrong_minus_one_right`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_delta_all_wrong_minus_one_right_n8_100_cosine_similarity.csv`

## Oracle True Layers

8, 22, 23, 24, 28

## Block Similarity

- true/true mean off-diagonal cosine: 0.04195520654320717
- false/false mean off-diagonal cosine: -0.0048272390849888325
- false/true mean cosine: 0.0022306228056550026

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 34 | 35 | -0.375503 | 3.102 |
| 8 | 9 | -0.268801 | 1.280 |
| 4 | 5 | -0.265762 | 1.559 |
| 9 | 10 | -0.265558 | 1.155 |
| 10 | 11 | -0.177836 | 1.027 |
| 12 | 13 | -0.177653 | 1.118 |
| 13 | 14 | -0.173978 | 1.479 |
| 11 | 12 | -0.156823 | 1.230 |
| 14 | 15 | -0.132952 | 1.291 |
| 15 | 16 | -0.101068 | 1.541 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | -0.018869 | False |
| 1 | 0.017100 | False |
| 2 | -0.009753 | False |
| 3 | -0.006111 | False |
| 4 | -0.017842 | False |
| 5 | -0.024246 | False |
| 6 | -0.019017 | False |
| 7 | -0.025874 | False |
| 8 | 0.429322 | True |
| 9 | -0.133557 | False |
| 10 | -0.022991 | False |
| 11 | -0.031896 | False |
| 12 | 0.001956 | False |
| 13 | -0.053087 | False |
| 14 | 0.018682 | False |
| 15 | -0.012994 | False |
| 16 | 0.029706 | False |
| 17 | 0.053757 | False |
| 18 | 0.053479 | False |
| 19 | 0.080140 | False |
| 20 | 0.032906 | False |
| 21 | -0.018166 | False |
| 22 | 0.484795 | True |
| 23 | 0.508583 | True |
| 24 | 0.499227 | True |
| 25 | 0.005289 | False |
| 26 | 0.081552 | False |
| 27 | 0.052325 | False |
| 28 | 0.494497 | True |
| 29 | 0.066553 | False |
| 30 | 0.034301 | False |
| 31 | 0.017098 | False |
| 32 | 0.011696 | False |
| 33 | 0.003174 | False |
| 34 | -0.049702 | False |
| 35 | 0.027473 | False |