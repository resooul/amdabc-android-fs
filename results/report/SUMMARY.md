# Run summary (results)

Machine: 12th Gen Intel(R) Core(TM) i9-12900K, 24 threads, 63.7 GB RAM; Windows 10 (10.0.19045); Python 3.11.18.

Protocol: 5-fold outer CV; 15 optimizers x 30 seeds, 500 FE each; fitness 0.9 x (1 - balanced_accuracy) + 0.1 x |S|/D.

## Progress

| Stage | Done | Total | % |
|---|---|---|---|
| base | 660 | 660 | 100 |
| opt | 6750 | 6750 | 100 |
| subsets | 225 | 225 | 100 |
| eval | 1635 | 1635 | 100 |
| scenarios | 384 | 384 | 100 |

## Data

| Step | DVM | JVM | Hybrid |
|---|---|---|---|
| Rows (processed tables) | 13,799 | 13,799 | 13,799 |
| Unique rows used | 5,204 | 3,427 | 6,765 |
|   malware | 1,744 | 412 | 1,843 |
|   benign | 3,460 | 3,015 | 4,922 |
| Features used | 151 | 109 | 260 |

## Base learners, all features (accuracy %, mean of outer folds)

| Model | DVM | JVM | Hybrid | Mean |
|---|---|---|---|---|
| DL-MLP | 96.89 | 94.84 | 97.56 | 96.43 |
| MLP | 96.50 | 95.10 | 97.25 | 96.28 |
| LR | 96.31 | 94.75 | 97.40 | 96.15 |
| XGB | 96.33 | 94.63 | 97.09 | 96.02 |
| GBM | 96.12 | 94.57 | 96.87 | 95.85 |
| PatchTransformer | 95.60 | 94.78 | 96.85 | 95.74 |
| LDA | 96.18 | 93.81 | 97.01 | 95.67 |
| CAT | 95.12 | 94.63 | 96.29 | 95.35 |
| CNN1D | 95.58 | 93.17 | 96.19 | 94.98 |
| RF | 93.99 | 93.32 | 95.05 | 94.12 |
| BiLSTM | 90.47 | 93.58 | 92.15 | 92.07 |

## Optimizers (final fitness, lower is better)

Mean over folds x seeds; rank = Friedman mean rank; p = Wilcoxon vs AMDABC, Holm-adjusted; A12 > 0.5 means AMDABC tends to be better.

### ablation

| Algorithm | DVM | JVM | Hybrid |
|---|---|---|---|
| AMDABC | 0.0813 (rank 1.44, p -) | 0.1392 (rank 1.43, p -) | 0.0730 (rank 1.43, p -) |
| ABC | 0.1041 (rank 3.92, p 9.2e-26) | 0.1985 (rank 4.05, p 9.2e-26) | 0.0943 (rank 3.89, p 9.2e-26) |
| AMDABC1 | 0.1045 (rank 4.34, p 9.2e-26) | 0.1992 (rank 4.27, p 9.2e-26) | 0.0953 (rank 4.49, p 9.2e-26) |
| AMDABC2 | 0.0821 (rank 1.56, p 9.6e-03) | 0.1415 (rank 1.57, p 6.3e-02) | 0.0734 (rank 1.57, p 3.3e-01) |
| AMDABC3 | 0.1031 (rank 3.74, p 9.2e-26) | 0.1962 (rank 3.68, p 9.2e-26) | 0.0938 (rank 3.62, p 9.2e-26) |

Best mean rank: DVM AMDABC, Hybrid AMDABC, JVM AMDABC

### abc_variants

| Algorithm | DVM | JVM | Hybrid |
|---|---|---|---|
| AMDABC | 0.0813 (rank 1.01, p -) | 0.1392 (rank 1.02, p -) | 0.0730 (rank 1.00, p -) |
| ABC | 0.1041 (rank 5.59, p 1.4e-25) | 0.1985 (rank 5.46, p 1.4e-25) | 0.0943 (rank 5.49, p 1.4e-25) |
| GBABC | 0.1005 (rank 4.11, p 1.4e-25) | 0.1920 (rank 4.21, p 1.4e-25) | 0.0913 (rank 4.28, p 1.4e-25) |
| GBC | 0.0981 (rank 3.43, p 1.4e-25) | 0.1870 (rank 3.65, p 1.4e-25) | 0.0885 (rank 3.31, p 1.4e-25) |
| ABCX | 0.0933 (rank 2.13, p 1.4e-25) | 0.1759 (rank 2.35, p 1.4e-25) | 0.0842 (rank 2.19, p 1.4e-25) |
| DABC | 0.1040 (rank 5.60, p 1.4e-25) | 0.1991 (rank 5.39, p 1.4e-25) | 0.0944 (rank 5.56, p 1.4e-25) |
| ABCVSS | 0.1051 (rank 6.15, p 1.4e-25) | 0.2008 (rank 5.93, p 1.4e-25) | 0.0953 (rank 6.17, p 1.4e-25) |

Best mean rank: DVM AMDABC, Hybrid AMDABC, JVM AMDABC

### metaheuristics

| Algorithm | DVM | JVM | Hybrid |
|---|---|---|---|
| AMDABC | 0.0813 (rank 1.39, p -) | 0.1392 (rank 1.74, p -) | 0.0730 (rank 1.38, p -) |
| ABC | 0.1041 (rank 6.71, p 1.4e-25) | 0.1985 (rank 6.54, p 1.4e-25) | 0.0943 (rank 6.83, p 1.4e-25) |
| MPA | 0.0849 (rank 2.20, p 1.1e-11) | 0.1422 (rank 2.06, p 1.4e-01) | 0.0763 (rank 2.13, p 6.9e-13) |
| SSO | 0.0933 (rank 4.44, p 1.4e-25) | 0.1716 (rank 4.19, p 6.8e-24) | 0.0848 (rank 4.72, p 1.4e-25) |
| HHO | 0.1005 (rank 5.96, p 1.4e-25) | 0.1905 (rank 5.89, p 2.0e-25) | 0.0887 (rank 5.67, p 1.4e-25) |
| GTO | 0.0927 (rank 4.21, p 2.5e-25) | 0.1693 (rank 4.16, p 9.6e-22) | 0.0818 (rank 3.89, p 1.4e-24) |
| PSO | 0.0884 (rank 3.09, p 2.7e-23) | 0.1635 (rank 3.42, p 7.4e-23) | 0.0807 (rank 3.39, p 7.5e-25) |

Best mean rank: DVM AMDABC, Hybrid AMDABC, JVM AMDABC

## Wrapper subsets vs all features (abc_variants; mean of the 7 ML models)

| Dataset | FS Method | ACC | F1 | FS |
|---|---|---|---|---|
| DVM | - | 95.79 | 95.81 | 151.00 |
| DVM | ABC | 94.60 | 94.64 | 46.60 |
| DVM | ABCVSS | 94.51 | 94.55 | 44.80 |
| DVM | ABCX | 94.80 | 94.83 | 40.60 |
| DVM | DABC | 94.34 | 94.37 | 47.80 |
| DVM | GBABC | 94.79 | 94.82 | 48.60 |
| DVM | GBC | 94.78 | 94.82 | 45.20 |
| DVM | AMDABC | 94.61 | 94.64 | 30.60 |
| JVM | - | 94.40 | 94.01 | 109.00 |
| JVM | ABC | 94.38 | 94.03 | 47.80 |
| JVM | ABCVSS | 94.43 | 94.08 | 48.00 |
| JVM | ABCX | 94.46 | 94.12 | 40.00 |
| JVM | DABC | 94.41 | 94.04 | 47.40 |
| JVM | GBABC | 94.39 | 94.02 | 44.60 |
| JVM | GBC | 94.46 | 94.09 | 40.00 |
| JVM | AMDABC | 94.20 | 93.83 | 41.60 |
| Hybrid | - | 96.71 | 96.72 | 260.00 |
| Hybrid | ABC | 95.90 | 95.92 | 46.40 |
| Hybrid | ABCVSS | 95.80 | 95.82 | 46.80 |
| Hybrid | ABCX | 95.84 | 95.87 | 43.60 |
| Hybrid | DABC | 95.84 | 95.86 | 47.00 |
| Hybrid | GBABC | 95.86 | 95.88 | 44.60 |
| Hybrid | GBC | 95.93 | 95.96 | 46.00 |
| Hybrid | AMDABC | 95.95 | 95.97 | 32.80 |

## Wrapper subsets vs all features (metaheuristics; mean of the 7 ML models)

| Dataset | FS Method | ACC | F1 | FS |
|---|---|---|---|---|
| DVM | - | 95.79 | 95.81 | 151.00 |
| DVM | ABC | 94.60 | 94.64 | 46.60 |
| DVM | MPA | 94.57 | 94.60 | 35.20 |
| DVM | SSO | 94.80 | 94.83 | 38.80 |
| DVM | HHO | 94.49 | 94.52 | 42.40 |
| DVM | GTO | 94.51 | 94.54 | 37.40 |
| DVM | PSO | 94.93 | 94.96 | 40.00 |
| DVM | AMDABC | 94.61 | 94.64 | 30.60 |
| JVM | - | 94.40 | 94.01 | 109.00 |
| JVM | ABC | 94.38 | 94.03 | 47.80 |
| JVM | MPA | 94.39 | 94.04 | 33.60 |
| JVM | SSO | 94.36 | 94.03 | 42.00 |
| JVM | HHO | 94.25 | 93.89 | 45.60 |
| JVM | GTO | 94.37 | 94.00 | 40.20 |
| JVM | PSO | 94.24 | 93.85 | 44.80 |
| JVM | AMDABC | 94.20 | 93.83 | 41.60 |
| Hybrid | - | 96.71 | 96.72 | 260.00 |
| Hybrid | ABC | 95.90 | 95.92 | 46.40 |
| Hybrid | MPA | 95.87 | 95.90 | 39.40 |
| Hybrid | SSO | 95.96 | 95.99 | 43.60 |
| Hybrid | HHO | 95.79 | 95.81 | 44.20 |
| Hybrid | GTO | 95.79 | 95.81 | 41.00 |
| Hybrid | PSO | 95.77 | 95.80 | 37.60 |
| Hybrid | AMDABC | 95.95 | 95.97 | 32.80 |

## Efficiency, AMDABC subset vs all features (mean over models)

| Dataset | full_n_features | amdabc_n_features | fit_speedup | predict_speedup |
|---|---|---|---|---|
| DVM | 151.00 | 30.60 | 2.92 | 1.95 |
| Hybrid | 260.00 | 32.80 | 2.92 | 2.99 |
| JVM | 109.00 | 41.60 | 1.37 | 0.85 |

## Source-shift scenarios (LR, weighted F1 %)

CI = 95% bootstrap interval of (AMDABC - all features) in percentage points.

| scenario | Dataset | n_full | n_selected | full_F1 | amdabc_F1 | ci_low | ci_high |
|---|---|---|---|---|---|---|---|
| S1 | DVM | 151 | 47 | 94.30 | 90.97 | -4.53 | -2.19 |
| S1 | JVM | 109 | 29 | 91.74 | 88.68 | -4.69 | -1.41 |
| S1 | Hybrid | 260 | 47 | 95.03 | 92.82 | -3.25 | -1.20 |
| S2 | DVM | 151 | 34 | 96.35 | 95.52 | -1.72 | 0.05 |
| S2 | JVM | 109 | 38 | 95.63 | 94.51 | -2.16 | 0.01 |
| S2 | Hybrid | 260 | 34 | 97.03 | 95.81 | -1.97 | -0.50 |
| S3 | DVM | 151 | 33 | 96.98 | 96.34 | -1.18 | -0.09 |
| S3 | JVM | 109 | 33 | 95.53 | 94.73 | -1.26 | -0.38 |
| S3 | Hybrid | 260 | 34 | 98.15 | 97.50 | -1.06 | -0.28 |
| S4 | DVM | 151 | 40 | 94.43 | 91.33 | -3.85 | -2.32 |
| S4 | JVM | 109 | 49 | 94.09 | 92.51 | -2.48 | -0.73 |
| S4 | Hybrid | 260 | 31 | 95.45 | 91.47 | -4.72 | -3.24 |
