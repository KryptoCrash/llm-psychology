# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_random_deviant_minus_one_right_n8_100_diffmeans.pt`
- direction: `random_deviant_minus_one_right`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_random_deviant_minus_one_right_n8_100_cosine_similarity.csv`

## Oracle True Layers

15

## Block Similarity

- true/true mean off-diagonal cosine: None
- false/false mean off-diagonal cosine: 0.1712149828672409
- false/true mean cosine: 0.1056787297129631

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 13 | 14 | 0.494386 | 1.864 |
| 7 | 8 | 0.513889 | 1.579 |
| 8 | 9 | 0.556984 | 1.303 |
| 4 | 5 | 0.567335 | 1.220 |
| 9 | 10 | 0.581397 | 1.129 |
| 15 | 16 | 0.602038 | 1.273 |
| 18 | 19 | 0.618154 | 1.499 |
| 14 | 15 | 0.618713 | 1.360 |
| 11 | 12 | 0.625365 | 1.297 |
| 12 | 13 | 0.628805 | 1.307 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | 0.005488 | False |
| 1 | 0.015813 | False |
| 2 | 0.018582 | False |
| 3 | -0.008354 | False |
| 4 | -0.035949 | False |
| 5 | 0.005808 | False |
| 6 | 0.022991 | False |
| 7 | 0.011071 | False |
| 8 | 0.037948 | False |
| 9 | 0.009174 | False |
| 10 | 0.054329 | False |
| 11 | 0.090362 | False |
| 12 | 0.218272 | False |
| 13 | 0.312127 | False |
| 14 | 0.618713 | False |
| 15 | 1.000000 | True |
| 16 | 0.602037 | False |
| 17 | 0.472826 | False |
| 18 | 0.317868 | False |
| 19 | 0.175489 | False |
| 20 | 0.138322 | False |
| 21 | 0.113192 | False |
| 22 | 0.094965 | False |
| 23 | 0.062559 | False |
| 24 | 0.057371 | False |
| 25 | 0.055837 | False |
| 26 | 0.040692 | False |
| 27 | 0.038884 | False |
| 28 | 0.035360 | False |
| 29 | 0.035024 | False |
| 30 | 0.033394 | False |
| 31 | 0.023255 | False |
| 32 | 0.019934 | False |
| 33 | 0.008868 | False |
| 34 | 0.005194 | False |
| 35 | -0.008692 | False |