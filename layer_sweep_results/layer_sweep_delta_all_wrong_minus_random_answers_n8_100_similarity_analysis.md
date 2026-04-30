# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_delta_all_wrong_minus_random_answers_n8_100_diffmeans.pt`
- direction: `all_wrong_minus_random_answers`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_delta_all_wrong_minus_random_answers_n8_100_cosine_similarity.csv`

## Oracle True Layers

8, 22, 23

## Block Similarity

- true/true mean off-diagonal cosine: 0.03513086587190628
- false/false mean off-diagonal cosine: -0.004487581551074982
- false/true mean cosine: 0.007056717295199633

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 8 | 9 | -0.269510 | 1.220 |
| 34 | 35 | -0.265747 | 2.787 |
| 9 | 10 | -0.250762 | 1.156 |
| 10 | 11 | -0.194085 | 0.911 |
| 13 | 14 | -0.159343 | 1.880 |
| 2 | 3 | -0.141981 | 1.376 |
| 11 | 12 | -0.140883 | 1.228 |
| 24 | 25 | -0.127044 | 0.839 |
| 3 | 4 | -0.123665 | 1.517 |
| 15 | 16 | -0.122905 | 1.418 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | -0.007809 | False |
| 1 | -0.026152 | False |
| 2 | 0.002985 | False |
| 3 | -0.020090 | False |
| 4 | 0.001038 | False |
| 5 | -0.052016 | False |
| 6 | -0.029441 | False |
| 7 | -0.010292 | False |
| 8 | 0.562763 | True |
| 9 | -0.147088 | False |
| 10 | -0.063123 | False |
| 11 | -0.035842 | False |
| 12 | -0.022406 | False |
| 13 | -0.058288 | False |
| 14 | -0.005881 | False |
| 15 | 0.021134 | False |
| 16 | 0.023578 | False |
| 17 | 0.062104 | False |
| 18 | 0.078715 | False |
| 19 | 0.057625 | False |
| 20 | 0.087067 | False |
| 21 | -0.013667 | False |
| 22 | 0.614749 | True |
| 23 | 0.614356 | True |
| 24 | 0.146055 | False |
| 25 | 0.016846 | False |
| 26 | 0.106732 | False |
| 27 | 0.074836 | False |
| 28 | 0.108659 | False |
| 29 | 0.065338 | False |
| 30 | 0.030925 | False |
| 31 | 0.023484 | False |
| 32 | 0.007601 | False |
| 33 | 0.022398 | False |
| 34 | -0.020209 | False |
| 35 | -0.034935 | False |