# Delta Oracle-True Centroid Contributions

- source: `layer_sweep_results/layer_sweep_delta_all_wrong_minus_one_right_n8_100_diffmeans.pt`
- oracle true layers: 8, 22, 23, 24, 28
- unnormalized mean true-unit-vector norm: 0.483285

## Oracle-True Delta Layers Ranked By Directional Centrality

| layer | cos to centroid | direct centroid contribution | leave-one-out delta | norm | raw projection |
|---:|---:|---:|---:|---:|---:|
| 23 | 0.508583 | 0.101717 | 0.08852345 | 7.517402 | 3.823220 |
| 24 | 0.499227 | 0.099845 | 0.08874357 | 10.059836 | 5.022145 |
| 28 | 0.494497 | 0.098899 | 0.08884513 | 13.033328 | 6.444943 |
| 22 | 0.484795 | 0.096959 | 0.08903325 | 6.091100 | 2.952936 |
| 8 | 0.429322 | 0.085864 | 0.08962035 | 0.202355 | 0.086875 |

## All Delta Layers Ranked By Raw Projection Onto Centroid

| layer | oracle true | raw projection | norm | cos to centroid |
|---:|:---:|---:|---:|---:|
| 28 | True | 6.444943 | 13.033328 | 0.494497 |
| 24 | True | 5.022145 | 10.059836 | 0.499227 |
| 23 | True | 3.823220 | 7.517402 | 0.508583 |
| 22 | True | 2.952936 | 6.091100 | 0.484795 |
| 35 | False | 2.381114 | 86.672127 | 0.027473 |
| 29 | False | 0.900633 | 13.532572 | 0.066553 |
| 26 | False | 0.781828 | 9.586814 | 0.081552 |
| 27 | False | 0.539591 | 10.312280 | 0.052325 |
| 30 | False | 0.447753 | 13.053714 | 0.034301 |
| 19 | False | 0.317149 | 3.957429 | 0.080140 |
| 31 | False | 0.244067 | 14.274852 | 0.017098 |
| 32 | False | 0.204573 | 17.490582 | 0.011696 |
| 20 | False | 0.128291 | 3.898752 | 0.032906 |
| 18 | False | 0.123234 | 2.304351 | 0.053479 |
| 8 | True | 0.086875 | 0.202355 | 0.429322 |