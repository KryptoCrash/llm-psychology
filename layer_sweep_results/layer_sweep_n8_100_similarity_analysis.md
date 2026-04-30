# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_n8_100_diffmeans.pt`
- direction: `all_wrong_minus_one_right`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_n8_100_cosine_similarity.csv`

## Oracle True Layers

24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35

## Block Similarity

- true/true mean off-diagonal cosine: 0.6683835983276367
- false/false mean off-diagonal cosine: 0.1488349884748459
- false/true mean cosine: 0.11498551815748215

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 34 | 35 | 0.360014 | 1.616 |
| 7 | 8 | 0.508129 | 1.765 |
| 3 | 4 | 0.536554 | 1.499 |
| 8 | 9 | 0.543048 | 1.275 |
| 4 | 5 | 0.555393 | 1.585 |
| 9 | 10 | 0.568496 | 1.163 |
| 13 | 14 | 0.573498 | 1.354 |
| 6 | 7 | 0.582480 | 1.608 |
| 15 | 16 | 0.596691 | 1.537 |
| 14 | 15 | 0.605505 | 1.339 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | -0.027012 | False |
| 1 | -0.017157 | False |
| 2 | -0.025835 | False |
| 3 | -0.010255 | False |
| 4 | 0.000958 | False |
| 5 | -0.018572 | False |
| 6 | -0.023335 | False |
| 7 | -0.026662 | False |
| 8 | 0.001973 | False |
| 9 | -0.020305 | False |
| 10 | -0.000752 | False |
| 11 | 0.005026 | False |
| 12 | 0.027779 | False |
| 13 | 0.025896 | False |
| 14 | 0.047008 | False |
| 15 | 0.045209 | False |
| 16 | 0.098666 | False |
| 17 | 0.160570 | False |
| 18 | 0.224749 | False |
| 19 | 0.310948 | False |
| 20 | 0.361742 | False |
| 21 | 0.409374 | False |
| 22 | 0.529009 | False |
| 23 | 0.647202 | False |
| 24 | 0.773579 | True |
| 25 | 0.829401 | True |
| 26 | 0.871510 | True |
| 27 | 0.895080 | True |
| 28 | 0.917649 | True |
| 29 | 0.928700 | True |
| 30 | 0.922891 | True |
| 31 | 0.905804 | True |
| 32 | 0.871935 | True |
| 33 | 0.827685 | True |
| 34 | 0.713235 | False |
| 35 | 0.449361 | True |