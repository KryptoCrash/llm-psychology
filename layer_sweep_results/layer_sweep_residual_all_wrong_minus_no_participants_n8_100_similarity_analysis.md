# Diffmean Layer Similarity Analysis

- source: `layer_sweep_results/layer_sweep_residual_all_wrong_minus_no_participants_n8_100_diffmeans.pt`
- direction: `all_wrong_minus_no_participants`
- sample_count: 100
- cosine matrix: `layer_sweep_results/layer_sweep_residual_all_wrong_minus_no_participants_n8_100_cosine_similarity.csv`

## Oracle True Layers

11, 34

## Block Similarity

- true/true mean off-diagonal cosine: 0.0908282995223999
- false/false mean off-diagonal cosine: 0.2706383764743805
- false/true mean cosine: 0.25650256872177124

## Lowest Adjacent Layer Cosines

| from | to | cosine | norm ratio |
|---:|---:|---:|---:|
| 34 | 35 | 0.415229 | 1.636 |
| 4 | 5 | 0.639893 | 1.391 |
| 3 | 4 | 0.646572 | 1.332 |
| 9 | 10 | 0.687169 | 1.201 |
| 8 | 9 | 0.699706 | 1.133 |
| 2 | 3 | 0.725491 | 1.319 |
| 7 | 8 | 0.744998 | 1.191 |
| 11 | 12 | 0.779901 | 1.132 |
| 1 | 2 | 0.780569 | 1.261 |
| 10 | 11 | 0.780664 | 1.104 |

## Cosine To Oracle-True Centroid

| layer | cosine | oracle true |
|---:|---:|:---:|
| 0 | 0.029258 | False |
| 1 | 0.032458 | False |
| 2 | 0.026062 | False |
| 3 | 0.028216 | False |
| 4 | 0.109831 | False |
| 5 | 0.175150 | False |
| 6 | 0.226515 | False |
| 7 | 0.260411 | False |
| 8 | 0.332018 | False |
| 9 | 0.413996 | False |
| 10 | 0.585749 | False |
| 11 | 0.738522 | True |
| 12 | 0.599248 | False |
| 13 | 0.512824 | False |
| 14 | 0.449309 | False |
| 15 | 0.362783 | False |
| 16 | 0.330661 | False |
| 17 | 0.320575 | False |
| 18 | 0.303927 | False |
| 19 | 0.294401 | False |
| 20 | 0.278655 | False |
| 21 | 0.283329 | False |
| 22 | 0.316158 | False |
| 23 | 0.321088 | False |
| 24 | 0.357636 | False |
| 25 | 0.398275 | False |
| 26 | 0.419570 | False |
| 27 | 0.432000 | False |
| 28 | 0.464891 | False |
| 29 | 0.497505 | False |
| 30 | 0.532537 | False |
| 31 | 0.568048 | False |
| 32 | 0.612534 | False |
| 33 | 0.656265 | False |
| 34 | 0.738522 | True |
| 35 | 0.276958 | False |