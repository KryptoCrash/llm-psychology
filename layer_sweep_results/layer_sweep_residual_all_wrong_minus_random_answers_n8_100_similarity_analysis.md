# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_residual_all_wrong_minus_random_answers_n8_100_diffmeans.pt`
- direction: `all_wrong_minus_random_answers`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_residual_all_wrong_minus_random_answers_n8_100_cosine_similarity.csv`

## Oracle True Layers

10, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33

## Block Similarity

- true/true mean off-diagonal cosine: 0.5284236669540405
- false/false mean off-diagonal cosine: 0.15585067868232727
- false/true mean cosine: 0.12852215766906738

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 13 | 14 | 0.474334 | 1.657 |
| 34 | 35 | 0.509402 | 1.497 |
| 7 | 8 | 0.562783 | 1.592 |
| 9 | 10 | 0.564725 | 1.103 |
| 8 | 9 | 0.583107 | 1.187 |
| 4 | 5 | 0.607981 | 1.414 |
| 14 | 15 | 0.623704 | 1.437 |
| 15 | 16 | 0.633467 | 1.435 |
| 12 | 13 | 0.635921 | 1.289 |
| 6 | 7 | 0.637139 | 1.423 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | -0.011025 | False |
| 1 | -0.018069 | False |
| 2 | -0.015756 | False |
| 3 | 0.001581 | False |
| 4 | 0.021333 | False |
| 5 | 0.019311 | False |
| 6 | 0.018664 | False |
| 7 | 0.033800 | False |
| 8 | 0.063056 | False |
| 9 | 0.090324 | False |
| 10 | 0.150711 | True |
| 11 | 0.149532 | False |
| 12 | 0.177363 | True |
| 13 | 0.099180 | False |
| 14 | 0.104010 | False |
| 15 | 0.099027 | False |
| 16 | 0.155365 | False |
| 17 | 0.209965 | False |
| 18 | 0.295518 | False |
| 19 | 0.382996 | False |
| 20 | 0.451837 | False |
| 21 | 0.518759 | False |
| 22 | 0.652816 | True |
| 23 | 0.776904 | True |
| 24 | 0.827784 | True |
| 25 | 0.868356 | True |
| 26 | 0.894538 | True |
| 27 | 0.903247 | True |
| 28 | 0.913652 | True |
| 29 | 0.908903 | True |
| 30 | 0.895998 | True |
| 31 | 0.883502 | True |
| 32 | 0.845600 | True |
| 33 | 0.796967 | True |
| 34 | 0.708679 | False |
| 35 | 0.393211 | False |