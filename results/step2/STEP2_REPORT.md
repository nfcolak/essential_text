# Step 2 Report: Transformer Fine-Tuning vs Baselines

Generated 2026-10-20 13:10 UTC from `results/step2/runs`.

## Protocol

- Both encoders (`bert_base`, `finbert_mlm`) use one shared fixed recipe (AdamW, linear warmup, same batch size, learning rate and epoch budget) on the Step 1 locked train split of each config.
- The epoch is selected by validation Macro-F1 only; the test split and SEntFiN are scored once with the selected checkpoint.
- Fine-tuned models run with several seeds; tables show mean ± sample sd over seeds. Baselines are deterministic and have the single run `seed42`.
- The 5-fold CV on `sentences_allagree` is scored on the fold validation part, the same part used for epoch selection (as in Step 1 CV), so it is an optimistic estimate and not a test score.
- `prosus_finbert` is scored only on SEntFiN 1.0 (it was fine-tuned on Financial PhraseBank, so scoring it there would leak).
- Paired tests: McNemar per seed (primary p = seed 42 pairing), Holm-Bonferroni over the pairs of one config and split, and a paired sentence bootstrap of the seed-averaged Macro-F1 difference (B = 10000, 95% percentile CI).
- Attribution: `delta_arch` = `bert_base` − `tfidf_best` and `delta_domain` = `finbert_mlm` − `bert_base`; `tfidf_best` is the TF-IDF model with the higher validation Macro-F1 in that config.

## In-domain test: Macro-F1 / accuracy (mean ± sd)

### sentences_allagree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.753 ± 0.000 | 0.820 ± 0.000 | 0.019 ± 0.000 | 0.141 ± 0.000 |
| loughran_mcdonald | 1 | 0.636 ± 0.000 | 0.735 ± 0.000 | 0.034 ± 0.000 | 0.264 ± 0.000 |
| tfidf_logreg | 1 | 0.869 ± 0.000 | 0.900 ± 0.000 | 0.010 ± 0.000 | 0.077 ± 0.000 |
| tfidf_svm | 1 | 0.870 ± 0.000 | 0.900 ± 0.000 | 0.005 ± 0.000 | 0.039 ± 0.000 |
| bert_base | 3 | 0.974 ± 0.013 | 0.980 ± 0.009 | 0.000 ± 0.000 | 0.011 ± 0.006 |
| finbert_mlm | 3 | 0.944 ± 0.005 | 0.959 ± 0.005 | 0.005 ± 0.005 | 0.035 ± 0.007 |

### sentences_75agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.646 ± 0.000 | 0.730 ± 0.000 | 0.022 ± 0.000 | 0.265 ± 0.000 |
| loughran_mcdonald | 1 | 0.597 ± 0.000 | 0.708 ± 0.000 | 0.031 ± 0.000 | 0.286 ± 0.000 |
| tfidf_logreg | 1 | 0.776 ± 0.000 | 0.820 ± 0.000 | 0.019 ± 0.000 | 0.047 ± 0.000 |
| tfidf_svm | 1 | 0.791 ± 0.000 | 0.844 ± 0.000 | 0.016 ± 0.000 | 0.107 ± 0.000 |
| bert_base | 3 | 0.906 ± 0.004 | 0.931 ± 0.004 | 0.020 ± 0.002 | 0.046 ± 0.010 |
| finbert_mlm | 3 | 0.894 ± 0.004 | 0.920 ± 0.001 | 0.016 ± 0.003 | 0.054 ± 0.011 |

### sentences_50agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.645 ± 0.000 | 0.731 ± 0.000 | 0.019 ± 0.000 | 0.205 ± 0.000 |
| loughran_mcdonald | 1 | 0.567 ± 0.000 | 0.664 ± 0.000 | 0.047 ± 0.000 | 0.331 ± 0.000 |
| tfidf_logreg | 1 | 0.762 ± 0.000 | 0.798 ± 0.000 | 0.026 ± 0.000 | 0.038 ± 0.000 |
| tfidf_svm | 1 | 0.768 ± 0.000 | 0.804 ± 0.000 | 0.014 ± 0.000 | 0.134 ± 0.000 |
| bert_base | 3 | 0.872 ± 0.007 | 0.881 ± 0.010 | 0.016 ± 0.013 | 0.074 ± 0.008 |
| finbert_mlm | 3 | 0.846 ± 0.014 | 0.867 ± 0.012 | 0.023 ± 0.005 | 0.054 ± 0.028 |

## SEntFiN 1.0 (external): Macro-F1 / accuracy (mean ± sd)

### sentences_allagree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.624 ± 0.000 | 0.634 ± 0.000 | 0.036 ± 0.000 | 0.333 ± 0.000 |
| loughran_mcdonald | 1 | 0.585 ± 0.000 | 0.592 ± 0.000 | 0.064 ± 0.000 | 0.405 ± 0.000 |
| tfidf_logreg | 1 | 0.451 ± 0.000 | 0.496 ± 0.000 | 0.020 ± 0.000 | 0.199 ± 0.000 |
| tfidf_svm | 1 | 0.450 ± 0.000 | 0.498 ± 0.000 | 0.011 ± 0.000 | 0.396 ± 0.000 |
| bert_base | 3 | 0.685 ± 0.021 | 0.684 ± 0.019 | 0.050 ± 0.008 | 0.280 ± 0.023 |
| finbert_mlm | 3 | 0.669 ± 0.028 | 0.670 ± 0.025 | 0.060 ± 0.012 | 0.293 ± 0.019 |

### sentences_75agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.625 ± 0.000 | 0.636 ± 0.000 | 0.036 ± 0.000 | 0.359 ± 0.000 |
| loughran_mcdonald | 1 | 0.585 ± 0.000 | 0.592 ± 0.000 | 0.064 ± 0.000 | 0.405 ± 0.000 |
| tfidf_logreg | 1 | 0.475 ± 0.000 | 0.513 ± 0.000 | 0.020 ± 0.000 | 0.189 ± 0.000 |
| tfidf_svm | 1 | 0.472 ± 0.000 | 0.514 ± 0.000 | 0.017 ± 0.000 | 0.392 ± 0.000 |
| bert_base | 3 | 0.676 ± 0.012 | 0.675 ± 0.010 | 0.042 ± 0.003 | 0.279 ± 0.011 |
| finbert_mlm | 3 | 0.653 ± 0.047 | 0.658 ± 0.039 | 0.036 ± 0.024 | 0.270 ± 0.042 |

### sentences_50agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| vader | 1 | 0.624 ± 0.000 | 0.634 ± 0.000 | 0.036 ± 0.000 | 0.311 ± 0.000 |
| loughran_mcdonald | 1 | 0.585 ± 0.000 | 0.592 ± 0.000 | 0.064 ± 0.000 | 0.405 ± 0.000 |
| tfidf_logreg | 1 | 0.475 ± 0.000 | 0.500 ± 0.000 | 0.043 ± 0.000 | 0.185 ± 0.000 |
| tfidf_svm | 1 | 0.469 ± 0.000 | 0.500 ± 0.000 | 0.037 ± 0.000 | 0.401 ± 0.000 |
| bert_base | 3 | 0.676 ± 0.018 | 0.676 ± 0.015 | 0.055 ± 0.016 | 0.253 ± 0.022 |
| finbert_mlm | 3 | 0.704 ± 0.010 | 0.700 ± 0.010 | 0.094 ± 0.005 | 0.172 ± 0.030 |

### prosus_finbert (zero-shot, SEntFiN only)

Macro-F1 0.727, accuracy 0.724.

## 5-fold CV on sentences_allagree (fold validation part)

| model | folds | Macro-F1 | accuracy |
|---|---|---|---|
| vader | 5 | 0.716 ± 0.041 | 0.800 ± 0.027 |
| loughran_mcdonald | 5 | 0.618 ± 0.030 | 0.732 ± 0.021 |
| tfidf_logreg | 5 | 0.806 ± 0.028 | 0.864 ± 0.015 |
| tfidf_svm | 5 | 0.816 ± 0.028 | 0.873 ± 0.014 |
| bert_base | 5 | 0.950 ± 0.013 | 0.962 ± 0.013 |
| finbert_mlm | 5 | 0.952 ± 0.013 | 0.967 ± 0.009 |

## Architecture vs domain-pretraining attribution

| config | split | tfidf_best | delta_arch Macro-F1 | delta_arch accuracy | delta_domain Macro-F1 | delta_domain accuracy |
|---|---|---|---|---|---|---|
| sentences_allagree | test | tfidf_svm | +0.104 [+0.064, +0.148], p<0.0001 | +0.081 [+0.051, +0.113], p<0.0001 | -0.029 [-0.050, -0.012], p<0.0001 | -0.022 [-0.036, -0.009], p<0.0001 |
| sentences_allagree | external | tfidf_svm | +0.235 [+0.224, +0.247], p<0.0001 | +0.186 [+0.175, +0.196], p<0.0001 | -0.016 [-0.024, -0.008], p<0.0001 | -0.014 [-0.021, -0.006], p<0.0001 |
| sentences_75agree | test | tfidf_svm | +0.116 [+0.074, +0.159], p<0.0001 | +0.088 [+0.059, +0.117], p<0.0001 | -0.012 [-0.036, +0.011], p=0.3042 | -0.011 [-0.027, +0.005], p=0.1802 |
| sentences_75agree | external | tfidf_svm | +0.203 [+0.192, +0.215], p<0.0001 | +0.162 [+0.152, +0.172], p<0.0001 | -0.023 [-0.030, -0.015], p<0.0001 | -0.017 [-0.024, -0.010], p<0.0001 |
| sentences_50agree | test | tfidf_svm | +0.104 [+0.069, +0.141], p<0.0001 | +0.077 [+0.048, +0.106], p<0.0001 | -0.026 [-0.051, -0.002], p=0.0366 | -0.014 [-0.033, +0.005], p=0.1396 |
| sentences_50agree | external | tfidf_svm | +0.207 [+0.195, +0.219], p<0.0001 | +0.176 [+0.165, +0.187], p<0.0001 | +0.028 [+0.019, +0.036], p<0.0001 | +0.025 [+0.017, +0.033], p<0.0001 |

## Paired comparisons: In-domain test

### sentences_allagree (N = 339)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 8.17e-05 | 0.000572 | 42: 8.17e-05 | +0.118 [+0.061, +0.178] |
| vader vs tfidf_logreg | 0.00212 | 0.0127 | 42: 0.00212 | -0.115 [-0.187, -0.042] |
| vader vs tfidf_svm | 0.00244 | 0.0127 | 42: 0.00244 | -0.117 [-0.192, -0.040] |
| vader vs bert_base | 1.66e-12 | 2.16e-11 | 42: 1.66e-12, 100: 9.03e-12, 2026: 1.06e-16 | -0.221 [-0.282, -0.164] |
| vader vs finbert_mlm | 2.99e-09 | 3.58e-08 | 42: 2.99e-09, 100: 2.92e-12, 2026: 3.5e-11 | -0.191 [-0.252, -0.133] |
| loughran_mcdonald vs tfidf_logreg | 8.31e-08 | 8.31e-07 | 42: 8.31e-08 | -0.233 [-0.322, -0.148] |
| loughran_mcdonald vs tfidf_svm | 4.48e-08 | 4.93e-07 | 42: 4.48e-08 | -0.234 [-0.322, -0.149] |
| loughran_mcdonald vs bert_base | 2.1e-20 | 3.16e-19 | 42: 2.1e-20, 100: 1.65e-19, 2026: 2.91e-25 | -0.338 [-0.409, -0.274] |
| loughran_mcdonald vs finbert_mlm | 3.92e-16 | 5.49e-15 | 42: 3.92e-16, 100: 5.08e-19, 2026: 5.08e-19 | -0.309 [-0.381, -0.242] |
| tfidf_logreg vs tfidf_svm | 1 | 1 | 42: 1 | -0.001 [-0.031, +0.029] |
| tfidf_logreg vs bert_base | 2.56e-06 | 2.3e-05 | 42: 2.56e-06, 100: 1.09e-05, 2026: 9.31e-10 | -0.105 [-0.149, -0.067] |
| tfidf_logreg vs finbert_mlm | 0.00393 | 0.0157 | 42: 0.00393, 100: 0.000324, 2026: 0.000508 | -0.076 [-0.121, -0.035] |
| tfidf_svm vs bert_base | 2.56e-06 | 2.3e-05 | 42: 2.56e-06, 100: 2.24e-05, 2026: 9.31e-10 | -0.104 [-0.148, -0.064] |
| tfidf_svm vs finbert_mlm | 0.0051 | 0.0157 | 42: 0.0051, 100: 0.000324, 2026: 0.000324 | -0.075 [-0.119, -0.034] |
| bert_base vs finbert_mlm | 0.0386 | 0.0771 | 42: 0.0386, 100: 0.424, 2026: 0.00195 | +0.029 [+0.012, +0.050] |

### sentences_75agree (N = 518)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 0.32 | 0.639 | 42: 0.32 | +0.049 [-0.005, +0.103] |
| vader vs tfidf_logreg | 0.000105 | 0.00042 | 42: 0.000105 | -0.130 [-0.193, -0.069] |
| vader vs tfidf_svm | 4.87e-07 | 3.41e-06 | 42: 4.87e-07 | -0.145 [-0.212, -0.081] |
| vader vs bert_base | 2.39e-22 | 3.35e-21 | 42: 2.39e-22, 100: 1.33e-22, 2026: 9.05e-21 | -0.261 [-0.315, -0.209] |
| vader vs finbert_mlm | 3.03e-21 | 3.63e-20 | 42: 3.03e-21, 100: 5.47e-21, 2026: 1.06e-21 | -0.248 [-0.301, -0.197] |
| loughran_mcdonald vs tfidf_logreg | 6.94e-06 | 3.47e-05 | 42: 6.94e-06 | -0.179 [-0.250, -0.112] |
| loughran_mcdonald vs tfidf_svm | 1.93e-08 | 1.54e-07 | 42: 1.93e-08 | -0.194 [-0.267, -0.123] |
| loughran_mcdonald vs bert_base | 9.6e-23 | 1.44e-21 | 42: 9.6e-23, 100: 5.46e-23, 2026: 5.18e-22 | -0.310 [-0.373, -0.250] |
| loughran_mcdonald vs finbert_mlm | 5.35e-22 | 6.95e-21 | 42: 5.35e-22, 100: 3.95e-22, 2026: 3.4e-23 | -0.297 [-0.357, -0.240] |
| tfidf_logreg vs tfidf_svm | 0.0169 | 0.0507 | 42: 0.0169 | -0.014 [-0.037, +0.008] |
| tfidf_logreg vs bert_base | 2.05e-11 | 2.25e-10 | 42: 2.05e-11, 100: 1.18e-11, 2026: 1.96e-09 | -0.130 [-0.174, -0.088] |
| tfidf_logreg vs finbert_mlm | 8.54e-09 | 7.69e-08 | 42: 8.54e-09, 100: 3.32e-08, 2026: 5.26e-09 | -0.118 [-0.162, -0.075] |
| tfidf_svm vs bert_base | 6.9e-09 | 6.9e-08 | 42: 6.9e-09, 100: 4.04e-09, 2026: 4.09e-07 | -0.116 [-0.159, -0.074] |
| tfidf_svm vs finbert_mlm | 1.65e-06 | 9.92e-06 | 42: 1.65e-06, 100: 5.27e-06, 2026: 1.65e-06 | -0.103 [-0.148, -0.062] |
| bert_base vs finbert_mlm | 0.327 | 0.639 | 42: 0.327, 100: 0.169, 2026: 0.701 | +0.012 [-0.011, +0.036] |

### sentences_50agree (N = 726)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 6.52e-05 | 0.000326 | 42: 6.52e-05 | +0.078 [+0.035, +0.121] |
| vader vs tfidf_logreg | 0.00101 | 0.00303 | 42: 0.00101 | -0.116 [-0.172, -0.062] |
| vader vs tfidf_svm | 0.000227 | 0.000907 | 42: 0.000227 | -0.123 [-0.179, -0.069] |
| vader vs bert_base | 2.27e-18 | 2.95e-17 | 42: 2.27e-18, 100: 5.91e-12, 2026: 3.73e-15 | -0.227 [-0.276, -0.181] |
| vader vs finbert_mlm | 2.82e-15 | 3.38e-14 | 42: 2.82e-15, 100: 1.21e-12, 2026: 2.11e-11 | -0.201 [-0.248, -0.155] |
| loughran_mcdonald vs tfidf_logreg | 3.05e-10 | 3.05e-09 | 42: 3.05e-10 | -0.194 [-0.250, -0.139] |
| loughran_mcdonald vs tfidf_svm | 3.59e-11 | 3.95e-10 | 42: 3.59e-11 | -0.201 [-0.257, -0.145] |
| loughran_mcdonald vs bert_base | 2e-28 | 2.8e-27 | 42: 2e-28, 100: 3.07e-22, 2026: 2.24e-25 | -0.305 [-0.354, -0.256] |
| loughran_mcdonald vs finbert_mlm | 3.07e-29 | 4.61e-28 | 42: 3.07e-29, 100: 1.14e-24, 2026: 1.52e-21 | -0.279 [-0.325, -0.234] |
| tfidf_logreg vs tfidf_svm | 0.359 | 0.719 | 42: 0.359 | -0.007 [-0.023, +0.009] |
| tfidf_logreg vs bert_base | 1.29e-09 | 1.16e-08 | 42: 1.29e-09, 100: 2.2e-05, 2026: 7.34e-08 | -0.110 [-0.147, -0.076] |
| tfidf_logreg vs finbert_mlm | 8.46e-08 | 5.92e-07 | 42: 8.46e-08, 100: 1.39e-05, 2026: 0.00029 | -0.084 [-0.120, -0.049] |
| tfidf_svm vs bert_base | 1.94e-08 | 1.55e-07 | 42: 1.94e-08, 100: 0.00015, 2026: 7.88e-07 | -0.104 [-0.141, -0.069] |
| tfidf_svm vs finbert_mlm | 1.14e-06 | 6.84e-06 | 42: 1.14e-06, 100: 9.65e-05, 2026: 0.00156 | -0.078 [-0.114, -0.042] |
| bert_base vs finbert_mlm | 0.389 | 0.719 | 42: 0.389, 100: 0.771, 2026: 0.0503 | +0.026 [+0.002, +0.051] |

## Paired comparisons: SEntFiN 1.0

### sentences_allagree (N = 9460)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 7.85e-22 | 3.92e-21 | 42: 7.85e-22 | +0.039 [+0.030, +0.048] |
| vader vs tfidf_logreg | 3.74e-158 | 5.61e-157 | 42: 3.74e-158 | +0.173 [+0.161, +0.185] |
| vader vs tfidf_svm | 3.47e-152 | 4.86e-151 | 42: 3.47e-152 | +0.175 [+0.163, +0.186] |
| vader vs bert_base | 3.62e-15 | 1.45e-14 | 42: 3.62e-15, 100: 8.71e-42, 2026: 1.81e-13 | -0.061 [-0.071, -0.051] |
| vader vs finbert_mlm | 1.82e-33 | 1.28e-32 | 42: 1.82e-33, 100: 0.0155, 2026: 8.53e-11 | -0.045 [-0.055, -0.035] |
| loughran_mcdonald vs tfidf_logreg | 2.86e-55 | 2.86e-54 | 42: 2.86e-55 | +0.134 [+0.120, +0.147] |
| loughran_mcdonald vs tfidf_svm | 6.18e-53 | 5.56e-52 | 42: 6.18e-53 | +0.135 [+0.121, +0.149] |
| loughran_mcdonald vs bert_base | 1.09e-52 | 8.75e-52 | 42: 1.09e-52, 100: 1.01e-88, 2026: 9.01e-49 | -0.100 [-0.111, -0.089] |
| loughran_mcdonald vs finbert_mlm | 4.79e-78 | 5.75e-77 | 42: 4.79e-78, 100: 1.89e-23, 2026: 3.25e-43 | -0.084 [-0.095, -0.073] |
| tfidf_logreg vs tfidf_svm | 0.284 | 0.284 | 42: 0.284 | +0.002 [-0.003, +0.006] |
| tfidf_logreg vs bert_base | 1.32e-214 | 2.24e-213 | 42: 1.32e-214, 100: 2.56e-265, 2026: 2.39e-215 | -0.234 [-0.245, -0.222] |
| tfidf_logreg vs finbert_mlm | 1.72e-260 | 3.27e-259 | 42: 1.72e-260, 100: 6.56e-184, 2026: 1.22e-225 | -0.218 [-0.229, -0.207] |
| tfidf_svm vs bert_base | 2.49e-210 | 3.99e-209 | 42: 2.49e-210, 100: 1.86e-257, 2026: 5.37e-211 | -0.235 [-0.247, -0.224] |
| tfidf_svm vs finbert_mlm | 1.22e-256 | 2.19e-255 | 42: 1.22e-256, 100: 2.91e-183, 2026: 2.08e-224 | -0.219 [-0.230, -0.208] |
| bert_base vs finbert_mlm | 4.78e-06 | 9.56e-06 | 42: 4.78e-06, 100: 5.96e-36, 2026: 0.388 | +0.016 [+0.008, +0.024] |
| prosus_finbert vs vader | 1.63e-56 | 1.79e-55 | 42: 1.63e-56 | +0.103 [+0.092, +0.114] |
| prosus_finbert vs loughran_mcdonald | 2.17e-105 | 2.82e-104 | 42: 2.17e-105 | +0.142 [+0.131, +0.154] |
| prosus_finbert vs tfidf_logreg | 1.5e-279 | 3.16e-278 | 42: 1.5e-279 | +0.276 [+0.264, +0.288] |
| prosus_finbert vs tfidf_svm | 4.93e-272 | 9.85e-271 | 42: 4.93e-272 | +0.278 [+0.265, +0.290] |
| prosus_finbert vs bert_base | 3.2e-26 | 1.92e-25 | 42: 3.2e-26, 100: 1.99e-05, 2026: 8.81e-29 | +0.042 [+0.034, +0.051] |
| prosus_finbert vs finbert_mlm | 6.12e-09 | 1.83e-08 | 42: 6.12e-09, 100: 6.49e-51, 2026: 1.47e-29 | +0.058 [+0.049, +0.068] |

### sentences_75agree (N = 9460)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 1.67e-23 | 8.37e-23 | 42: 1.67e-23 | +0.040 [+0.031, +0.049] |
| vader vs tfidf_logreg | 6.32e-127 | 8.85e-126 | 42: 6.32e-127 | +0.150 [+0.139, +0.161] |
| vader vs tfidf_svm | 4.96e-127 | 7.45e-126 | 42: 4.96e-127 | +0.153 [+0.142, +0.164] |
| vader vs bert_base | 1.3e-14 | 5.19e-14 | 42: 1.3e-14, 100: 4.64e-21, 2026: 6.66e-08 | -0.050 [-0.061, -0.040] |
| vader vs finbert_mlm | 5.81e-05 | 0.000116 | 42: 5.81e-05, 100: 6.2e-32, 2026: 0.00211 | -0.028 [-0.038, -0.017] |
| loughran_mcdonald vs tfidf_logreg | 3.24e-38 | 2.92e-37 | 42: 3.24e-38 | +0.110 [+0.097, +0.124] |
| loughran_mcdonald vs tfidf_svm | 1.47e-38 | 1.47e-37 | 42: 1.47e-38 | +0.113 [+0.099, +0.127] |
| loughran_mcdonald vs bert_base | 3.29e-52 | 3.61e-51 | 42: 3.29e-52, 100: 1.38e-62, 2026: 4.07e-38 | -0.091 [-0.102, -0.079] |
| loughran_mcdonald vs finbert_mlm | 6.65e-32 | 4.65e-31 | 42: 6.65e-32, 100: 1.6e-82, 2026: 3.19e-07 | -0.068 [-0.079, -0.057] |
| tfidf_logreg vs tfidf_svm | 0.864 | 0.864 | 42: 0.864 | +0.003 [-0.001, +0.007] |
| tfidf_logreg vs bert_base | 6.87e-190 | 1.24e-188 | 42: 6.87e-190, 100: 3.33e-209, 2026: 1.79e-165 | -0.201 [-0.212, -0.190] |
| tfidf_logreg vs finbert_mlm | 1.9e-155 | 3.04e-154 | 42: 1.9e-155, 100: 7.37e-235, 2026: 1.34e-92 | -0.178 [-0.189, -0.167] |
| tfidf_svm vs bert_base | 1.16e-192 | 2.2e-191 | 42: 1.16e-192, 100: 3.54e-211, 2026: 2.54e-167 | -0.203 [-0.215, -0.192] |
| tfidf_svm vs finbert_mlm | 6.24e-160 | 1.06e-158 | 42: 6.24e-160, 100: 1.48e-238, 2026: 9.51e-96 | -0.181 [-0.192, -0.170] |
| bert_base vs finbert_mlm | 7.63e-06 | 2.29e-05 | 42: 7.63e-06, 100: 0.00493, 2026: 4.59e-22 | +0.023 [+0.015, +0.030] |
| prosus_finbert vs vader | 1.9e-54 | 2.28e-53 | 42: 1.9e-54 | +0.102 [+0.091, +0.114] |
| prosus_finbert vs loughran_mcdonald | 2.17e-105 | 2.82e-104 | 42: 2.17e-105 | +0.142 [+0.131, +0.154] |
| prosus_finbert vs tfidf_logreg | 1.94e-248 | 4.08e-247 | 42: 1.94e-248 | +0.253 [+0.240, +0.265] |
| prosus_finbert vs tfidf_svm | 8.37e-247 | 1.67e-245 | 42: 8.37e-247 | +0.255 [+0.243, +0.268] |
| prosus_finbert vs bert_base | 5.81e-25 | 3.48e-24 | 42: 5.81e-25, 100: 6.09e-18, 2026: 2.34e-37 | +0.052 [+0.044, +0.060] |
| prosus_finbert vs finbert_mlm | 6.8e-38 | 5.44e-37 | 42: 6.8e-38, 100: 3.07e-08, 2026: 3.57e-78 | +0.075 [+0.065, +0.084] |

### sentences_50agree (N = 9460)

| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| vader vs loughran_mcdonald | 3.19e-21 | 1.6e-20 | 42: 3.19e-21 | +0.039 [+0.030, +0.048] |
| vader vs tfidf_logreg | 2.89e-136 | 4.04e-135 | 42: 2.89e-136 | +0.149 [+0.137, +0.160] |
| vader vs tfidf_svm | 8.77e-138 | 1.32e-136 | 42: 8.77e-138 | +0.155 [+0.143, +0.167] |
| vader vs bert_base | 1.14e-17 | 4.56e-17 | 42: 1.14e-17, 100: 2.19e-22, 2026: 7.87e-06 | -0.052 [-0.063, -0.041] |
| vader vs finbert_mlm | 2.73e-41 | 1.91e-40 | 42: 2.73e-41, 100: 9.72e-35, 2026: 5.71e-25 | -0.080 [-0.090, -0.069] |
| loughran_mcdonald vs tfidf_logreg | 6.32e-53 | 5.06e-52 | 42: 6.32e-53 | +0.110 [+0.096, +0.123] |
| loughran_mcdonald vs tfidf_svm | 2.74e-54 | 2.74e-53 | 42: 2.74e-54 | +0.116 [+0.102, +0.129] |
| loughran_mcdonald vs bert_base | 3.72e-54 | 3.35e-53 | 42: 3.72e-54, 100: 9.49e-61, 2026: 2.46e-29 | -0.091 [-0.103, -0.080] |
| loughran_mcdonald vs finbert_mlm | 3.63e-96 | 4.36e-95 | 42: 3.63e-96, 100: 1.09e-88, 2026: 1.03e-69 | -0.119 [-0.130, -0.108] |
| tfidf_logreg vs tfidf_svm | 0.959 | 0.959 | 42: 0.959 | +0.006 [+0.001, +0.010] |
| tfidf_logreg vs bert_base | 8.54e-204 | 1.37e-202 | 42: 8.54e-204, 100: 1.25e-207, 2026: 2.59e-154 | -0.201 [-0.213, -0.189] |
| tfidf_logreg vs finbert_mlm | 2.87e-262 | 5.16e-261 | 42: 2.87e-262, 100: 2.35e-252, 2026: 1.63e-232 | -0.229 [-0.240, -0.217] |
| tfidf_svm vs bert_base | 1.27e-205 | 2.16e-204 | 42: 1.27e-205, 100: 1.24e-206, 2026: 3.59e-155 | -0.207 [-0.219, -0.195] |
| tfidf_svm vs finbert_mlm | 4.63e-263 | 8.8e-262 | 42: 4.63e-263, 100: 7.12e-255, 2026: 9.48e-234 | -0.235 [-0.246, -0.223] |
| bert_base vs finbert_mlm | 7.49e-10 | 2.25e-09 | 42: 7.49e-10, 100: 0.00561, 2026: 8.63e-10 | -0.028 [-0.036, -0.019] |
| prosus_finbert vs vader | 2.67e-57 | 2.94e-56 | 42: 2.67e-57 | +0.103 [+0.092, +0.115] |
| prosus_finbert vs loughran_mcdonald | 2.17e-105 | 2.82e-104 | 42: 2.17e-105 | +0.142 [+0.131, +0.154] |
| prosus_finbert vs tfidf_logreg | 4.19e-275 | 8.79e-274 | 42: 4.19e-275 | +0.252 [+0.240, +0.265] |
| prosus_finbert vs tfidf_svm | 2.84e-273 | 5.68e-272 | 42: 2.84e-273 | +0.258 [+0.246, +0.271] |
| prosus_finbert vs bert_base | 4.59e-22 | 2.76e-21 | 42: 4.59e-22, 100: 5.71e-16, 2026: 8.09e-42 | +0.051 [+0.043, +0.059] |
| prosus_finbert vs finbert_mlm | 0.00259 | 0.00519 | 42: 0.00259, 100: 1.81e-06, 2026: 1.08e-12 | +0.024 [+0.015, +0.032] |

## Figures

- `step2_macro_f1_by_config.png`
- `step2_external_generalization.png`
- `step2_attribution.png`
- `step2_confusion_matrices_allagree.png`
