# Step 2 Report: Transformer Fine-Tuning vs Baselines

Generated 2026-10-20 14:27 UTC from `results/step2/runs`.

## Protocol

- Both encoders (`bert_base`, `finbert_mlm`) use one shared fixed recipe (AdamW, linear warmup, same batch size, learning rate and epoch budget) on the Step 1 locked train split of each config.
- The epoch is selected by validation Macro-F1 only; the test split and SEntFiN are scored once with the selected checkpoint.
- Fine-tuned models run with several seeds; tables show mean ± sample sd over seeds. Baselines are deterministic and have the single run `seed42` (sd shown as n/a).
- Baseline fitting: the official lexicons (VADER, Loughran-McDonald) use fixed published rules and nothing is tuned. The finance-adjusted lexicon variants tune their decision thresholds on the validation split of each config, and the TF-IDF models tune their hyperparameters on validation per config.
- The 5-fold CV on `sentences_allagree` is scored on the fold validation part, the same part used for epoch selection (as in Step 1 CV), so it is an optimistic estimate and not a test score.
- The three PhraseBank levels (allagree, 75agree, 50agree) are nested: a sentence in a stricter level is also in the looser ones. Models and test splits are never mixed across levels, so tables are compared within a config and across configs only descriptively.
- `prosus_finbert` (ProsusAI/finbert) is zero-shot and scored only on SEntFiN 1.0 (it was fine-tuned on Financial PhraseBank, so scoring it there would leak).
- Confidence intervals include seed variation: the bootstrap resamples sentences and, for the fine-tuned models, the trained seeds (two-level, seed+sentence). The earlier sentence-only interval, conditional on the three seeds, is stored as `conditional_on_seeds` in `attribution.json`.
- Bootstrap p-value (B = 10000): p = min(1, 2 * min(P(d <= 0), P(d >= 0))) over the resampled differences d, so the smallest non-zero value is 2/B; 0 is printed as p < 0.0002.
- McNemar is a per-seed diagnostic. There is no primary seed: each seed pairing forms its own family, Holm-Bonferroni is applied within each family (pairs of single-run models enter every family with their one run), and the report gives p_min, p_max and the number of seed families with Holm-adjusted p < 0.05. The seed+sentence bootstrap of the Macro-F1 difference is the headline test.
- Attribution: `delta_arch` = BERT − `tfidf_best` and `delta_domain` = FinBERT-MLM − BERT; `tfidf_best` is the TF-IDF model with the higher validation Macro-F1 in that config. Per-seed deltas are listed beside each interval.

## In-domain test: Macro-F1 / accuracy (mean ± sd)

### sentences_allagree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.500 (sd n/a) | 0.590 (sd n/a) | 0.077 (sd n/a) | 0.406 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.498 (sd n/a) | 0.652 (sd n/a) | 0.072 (sd n/a) | 0.274 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.753 (sd n/a) | 0.820 (sd n/a) | 0.019 (sd n/a) | 0.141 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.636 (sd n/a) | 0.735 (sd n/a) | 0.034 (sd n/a) | 0.264 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.869 (sd n/a) | 0.900 (sd n/a) | 0.010 (sd n/a) | 0.077 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.870 (sd n/a) | 0.900 (sd n/a) | 0.005 (sd n/a) | 0.039 (sd n/a) |
| bert-base-uncased | 3 | 0.974 ± 0.013 | 0.980 ± 0.009 | 0.000 ± 0.000 | 0.011 ± 0.006 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.944 ± 0.005 | 0.959 ± 0.005 | 0.005 ± 0.005 | 0.035 ± 0.007 |

### sentences_75agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.515 (sd n/a) | 0.577 (sd n/a) | 0.065 (sd n/a) | 0.418 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.518 (sd n/a) | 0.666 (sd n/a) | 0.084 (sd n/a) | 0.262 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.646 (sd n/a) | 0.730 (sd n/a) | 0.022 (sd n/a) | 0.265 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.597 (sd n/a) | 0.708 (sd n/a) | 0.031 (sd n/a) | 0.286 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.776 (sd n/a) | 0.820 (sd n/a) | 0.019 (sd n/a) | 0.047 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.791 (sd n/a) | 0.844 (sd n/a) | 0.016 (sd n/a) | 0.107 (sd n/a) |
| bert-base-uncased | 3 | 0.906 ± 0.004 | 0.931 ± 0.004 | 0.020 ± 0.002 | 0.046 ± 0.010 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.894 ± 0.004 | 0.920 ± 0.001 | 0.016 ± 0.003 | 0.054 ± 0.011 |

### sentences_50agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.491 (sd n/a) | 0.559 (sd n/a) | 0.058 (sd n/a) | 0.438 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.502 (sd n/a) | 0.614 (sd n/a) | 0.086 (sd n/a) | 0.319 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.645 (sd n/a) | 0.731 (sd n/a) | 0.019 (sd n/a) | 0.205 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.567 (sd n/a) | 0.664 (sd n/a) | 0.047 (sd n/a) | 0.331 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.762 (sd n/a) | 0.798 (sd n/a) | 0.026 (sd n/a) | 0.038 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.768 (sd n/a) | 0.804 (sd n/a) | 0.014 (sd n/a) | 0.134 (sd n/a) |
| bert-base-uncased | 3 | 0.872 ± 0.007 | 0.881 ± 0.010 | 0.016 ± 0.013 | 0.074 ± 0.008 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.846 ± 0.014 | 0.867 ± 0.012 | 0.023 ± 0.005 | 0.054 ± 0.028 |

## SEntFiN 1.0 (external): Macro-F1 / accuracy (mean ± sd)

### sentences_allagree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.525 (sd n/a) | 0.525 (sd n/a) | 0.098 (sd n/a) | 0.470 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.624 (sd n/a) | 0.634 (sd n/a) | 0.036 (sd n/a) | 0.333 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.585 (sd n/a) | 0.592 (sd n/a) | 0.064 (sd n/a) | 0.405 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.451 (sd n/a) | 0.496 (sd n/a) | 0.020 (sd n/a) | 0.199 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.450 (sd n/a) | 0.498 (sd n/a) | 0.011 (sd n/a) | 0.396 (sd n/a) |
| bert-base-uncased | 3 | 0.685 ± 0.021 | 0.684 ± 0.019 | 0.050 ± 0.008 | 0.280 ± 0.023 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.669 ± 0.028 | 0.670 ± 0.025 | 0.060 ± 0.012 | 0.293 ± 0.019 |

### sentences_75agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.525 (sd n/a) | 0.525 (sd n/a) | 0.098 (sd n/a) | 0.470 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.625 (sd n/a) | 0.636 (sd n/a) | 0.036 (sd n/a) | 0.359 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.585 (sd n/a) | 0.592 (sd n/a) | 0.064 (sd n/a) | 0.405 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.475 (sd n/a) | 0.513 (sd n/a) | 0.020 (sd n/a) | 0.189 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.472 (sd n/a) | 0.514 (sd n/a) | 0.017 (sd n/a) | 0.392 (sd n/a) |
| bert-base-uncased | 3 | 0.676 ± 0.012 | 0.675 ± 0.010 | 0.042 ± 0.003 | 0.279 ± 0.011 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.653 ± 0.047 | 0.658 ± 0.039 | 0.036 ± 0.024 | 0.270 ± 0.042 |

### sentences_50agree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.525 (sd n/a) | 0.525 (sd n/a) | 0.098 (sd n/a) | 0.470 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 1 | 0.624 (sd n/a) | 0.634 (sd n/a) | 0.036 (sd n/a) | 0.311 (sd n/a) |
| LM-style (finance-adjusted, custom) | 1 | 0.585 (sd n/a) | 0.592 (sd n/a) | 0.064 (sd n/a) | 0.405 (sd n/a) |
| TF-IDF + Logistic Regression | 1 | 0.475 (sd n/a) | 0.500 (sd n/a) | 0.043 (sd n/a) | 0.185 (sd n/a) |
| TF-IDF + Linear SVM | 1 | 0.469 (sd n/a) | 0.500 (sd n/a) | 0.037 (sd n/a) | 0.401 (sd n/a) |
| bert-base-uncased | 3 | 0.676 ± 0.018 | 0.676 ± 0.015 | 0.055 ± 0.016 | 0.253 ± 0.022 |
| FinBERT-MLM (finbert-pretrain) | 3 | 0.704 ± 0.010 | 0.700 ± 0.010 | 0.094 ± 0.005 | 0.172 ± 0.030 |

### ProsusAI/finbert (zero-shot) (SEntFiN only)

Macro-F1 0.727, accuracy 0.724.

## 5-fold CV on sentences_allagree (fold validation part)

| model | folds | Macro-F1 | accuracy |
|---|---|---|---|
| VADER (official lexicon) | 5 | 0.492 ± 0.038 | 0.568 ± 0.035 |
| Loughran-McDonald (official dictionary) | 5 | 0.480 ± 0.026 | 0.649 ± 0.019 |
| VADER-style (finance-adjusted, custom) | 5 | 0.716 ± 0.041 | 0.800 ± 0.027 |
| LM-style (finance-adjusted, custom) | 5 | 0.618 ± 0.030 | 0.732 ± 0.021 |
| TF-IDF + Logistic Regression | 5 | 0.806 ± 0.028 | 0.864 ± 0.015 |
| TF-IDF + Linear SVM | 5 | 0.816 ± 0.028 | 0.873 ± 0.014 |
| bert-base-uncased | 5 | 0.950 ± 0.013 | 0.962 ± 0.013 |
| FinBERT-MLM (finbert-pretrain) | 5 | 0.952 ± 0.013 | 0.967 ± 0.009 |

## Architecture vs domain-pretraining attribution

Differences with the 95% seed+sentence bootstrap CI.

| config | split | tfidf_best | delta_arch Macro-F1 | delta_arch accuracy | delta_domain Macro-F1 | delta_domain accuracy |
|---|---|---|---|---|---|---|
| sentences_allagree | test | TF-IDF + Linear SVM | +0.104 [+0.062, +0.150], p < 0.0002 | +0.081 [+0.049, +0.114], p < 0.0002 | -0.029 [-0.057, -0.006], p=0.0124 | -0.022 [-0.041, -0.004], p=0.0186 |
| sentences_allagree | external | TF-IDF + Linear SVM | +0.235 [+0.215, +0.259], p < 0.0002 | +0.186 [+0.168, +0.207], p < 0.0002 | -0.016 [-0.048, +0.016], p=0.3514 | -0.014 [-0.043, +0.015], p=0.3662 |
| sentences_75agree | test | TF-IDF + Linear SVM | +0.116 [+0.073, +0.160], p < 0.0002 | +0.088 [+0.058, +0.117], p < 0.0002 | -0.012 [-0.039, +0.014], p=0.3516 | -0.011 [-0.029, +0.006], p=0.2308 |
| sentences_75agree | external | TF-IDF + Linear SVM | +0.203 [+0.187, +0.219], p < 0.0002 | +0.162 [+0.148, +0.176], p < 0.0002 | -0.023 [-0.068, +0.022], p=0.3366 | -0.017 [-0.055, +0.020], p=0.3782 |
| sentences_50agree | test | TF-IDF + Linear SVM | +0.104 [+0.067, +0.143], p < 0.0002 | +0.077 [+0.045, +0.108], p < 0.0002 | -0.026 [-0.056, +0.005], p=0.0984 | -0.014 [-0.039, +0.012], p=0.2758 |
| sentences_50agree | external | TF-IDF + Linear SVM | +0.207 [+0.186, +0.227], p < 0.0002 | +0.176 [+0.158, +0.193], p < 0.0002 | +0.028 [+0.007, +0.050], p=0.0058 | +0.025 [+0.007, +0.044], p=0.0050 |

### Per-seed Macro-F1 deltas

| config | split | delta | per-seed deltas | mean ± sd | seeds with the mean's sign | sentence-only CI (seeds fixed) |
|---|---|---|---|---|---|---|
| sentences_allagree | test | delta_arch | +0.099, +0.094, +0.119 | +0.104 ± 0.013 | 3/3 | [+0.064, +0.148] |
| sentences_allagree | test | delta_domain | -0.030, -0.015, -0.044 | -0.029 ± 0.015 | 3/3 | [-0.050, -0.012] |
| sentences_allagree | external | delta_arch | +0.225, +0.259, +0.222 | +0.235 ± 0.021 | 3/3 | [+0.224, +0.247] |
| sentences_allagree | external | delta_domain | +0.023, -0.066, -0.004 | -0.016 ± 0.046 | 2/3 | [-0.024, -0.008] |
| sentences_75agree | test | delta_arch | +0.118, +0.118, +0.111 | +0.116 ± 0.004 | 3/3 | [+0.074, +0.159] |
| sentences_75agree | test | delta_domain | -0.011, -0.018, -0.008 | -0.012 ± 0.005 | 3/3 | [-0.036, +0.011] |
| sentences_75agree | external | delta_arch | +0.205, +0.215, +0.191 | +0.203 ± 0.012 | 3/3 | [+0.192, +0.215] |
| sentences_75agree | external | delta_domain | -0.024, +0.013, -0.057 | -0.023 ± 0.035 | 2/3 | [-0.030, -0.015] |
| sentences_50agree | test | delta_arch | +0.111, +0.099, +0.101 | +0.104 ± 0.007 | 3/3 | [+0.069, +0.141] |
| sentences_50agree | test | delta_domain | -0.018, -0.025, -0.035 | -0.026 ± 0.009 | 3/3 | [-0.051, -0.002] |
| sentences_50agree | external | delta_arch | +0.211, +0.223, +0.187 | +0.207 ± 0.018 | 3/3 | [+0.195, +0.219] |
| sentences_50agree | external | delta_domain | +0.033, +0.013, +0.037 | +0.028 ± 0.013 | 3/3 | [+0.019, +0.036] |

## Paired comparisons: In-domain test

### sentences_allagree (N = 339)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.119–0.119 | 0/3 | 42: 0.119 | +0.002 [-0.080, +0.084], p=0.9514 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 2.51e-14–2.51e-14 | 3/3 | 42: 2.51e-14 | -0.253 [-0.319, -0.189], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 8.18e-06–8.18e-06 | 3/3 | 42: 8.18e-06 | -0.135 [-0.198, -0.072], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 1.39e-18–1.39e-18 | 3/3 | 42: 1.39e-18 | -0.369 [-0.448, -0.283], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 3.8e-19–3.8e-19 | 3/3 | 42: 3.8e-19 | -0.370 [-0.451, -0.284], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 7.98e-40–6.21e-34 | 3/3 | 42: 4.4e-35, 100: 6.21e-34, 2026: 7.98e-40 | -0.474 [-0.538, -0.410], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 5.16e-32–8.6e-30 | 3/3 | 42: 8.6e-30, 100: 2.72e-31, 2026: 5.16e-32 | -0.444 [-0.510, -0.377], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 6.88e-09–6.88e-09 | 3/3 | 42: 6.88e-09 | -0.256 [-0.330, -0.182], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.00152–0.00152 | 3/3 | 42: 0.00152 | -0.138 [-0.212, -0.060], p=0.0004 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 1.25e-14–1.25e-14 | 3/3 | 42: 1.25e-14 | -0.371 [-0.457, -0.287], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 2.21e-14–2.21e-14 | 3/3 | 42: 2.21e-14 | -0.372 [-0.462, -0.282], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 4.81e-35–3.05e-27 | 3/3 | 42: 6.31e-31, 100: 3.05e-27, 2026: 4.81e-35 | -0.476 [-0.546, -0.410], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 7.73e-27–1.71e-23 | 3/3 | 42: 1.71e-23, 100: 7.73e-27, 2026: 3.98e-26 | -0.447 [-0.519, -0.376], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 8.17e-05–8.17e-05 | 3/3 | 42: 8.17e-05 | +0.118 [+0.061, +0.178], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.00212–0.00212 | 3/3 | 42: 0.00212 | -0.115 [-0.187, -0.042], p=0.0032 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 0.00244–0.00244 | 3/3 | 42: 0.00244 | -0.117 [-0.192, -0.040], p=0.0038 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 1.06e-16–9.03e-12 | 3/3 | 42: 1.66e-12, 100: 9.03e-12, 2026: 1.06e-16 | -0.221 [-0.283, -0.161], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.92e-12–2.99e-09 | 3/3 | 42: 2.99e-09, 100: 2.92e-12, 2026: 3.5e-11 | -0.191 [-0.253, -0.132], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 8.31e-08–8.31e-08 | 3/3 | 42: 8.31e-08 | -0.233 [-0.322, -0.148], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.48e-08–4.48e-08 | 3/3 | 42: 4.48e-08 | -0.234 [-0.322, -0.149], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 2.91e-25–1.65e-19 | 3/3 | 42: 2.1e-20, 100: 1.65e-19, 2026: 2.91e-25 | -0.338 [-0.410, -0.272], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 5.08e-19–3.92e-16 | 3/3 | 42: 3.92e-16, 100: 5.08e-19, 2026: 5.08e-19 | -0.309 [-0.381, -0.241], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 1–1 | 0/3 | 42: 1 | -0.001 [-0.031, +0.029], p=0.9368 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 9.31e-10–1.09e-05 | 3/3 | 42: 2.56e-06, 100: 1.09e-05, 2026: 9.31e-10 | -0.105 [-0.151, -0.064], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 0.000324–0.00393 | 3/3 | 42: 0.00393, 100: 0.000324, 2026: 0.000508 | -0.076 [-0.121, -0.034], p=0.0008 |
| TF-IDF + Linear SVM vs bert-base-uncased | 9.31e-10–2.24e-05 | 3/3 | 42: 2.56e-06, 100: 2.24e-05, 2026: 9.31e-10 | -0.104 [-0.150, -0.062], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 0.000324–0.0051 | 3/3 | 42: 0.0051, 100: 0.000324, 2026: 0.000324 | -0.075 [-0.120, -0.033], p=0.0004 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.00195–0.424 | 1/3 | 42: 0.0386, 100: 0.424, 2026: 0.00195 | +0.029 [+0.007, +0.057], p=0.0114 |

### sentences_75agree (N = 518)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.00331–0.00331 | 3/3 | 42: 0.00331 | -0.003 [-0.068, +0.064], p=0.9380 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 1.31e-07–1.31e-07 | 3/3 | 42: 1.31e-07 | -0.131 [-0.195, -0.067], p=0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.21e-06–2.21e-06 | 3/3 | 42: 2.21e-06 | -0.082 [-0.143, -0.021], p=0.0082 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 3.44e-16–3.44e-16 | 3/3 | 42: 3.44e-16 | -0.261 [-0.333, -0.190], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 1.18e-19–1.18e-19 | 3/3 | 42: 1.18e-19 | -0.276 [-0.349, -0.202], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 2.33e-39–8.61e-39 | 3/3 | 42: 4.23e-39, 100: 2.33e-39, 2026: 8.61e-39 | -0.391 [-0.455, -0.327], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.14e-36–7.71e-36 | 3/3 | 42: 7.71e-36, 100: 2.05e-36, 2026: 1.14e-36 | -0.379 [-0.440, -0.317], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 0.00724–0.00724 | 3/3 | 42: 0.00724 | -0.128 [-0.193, -0.063], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.0263–0.0263 | 0/3 | 42: 0.0263 | -0.079 [-0.137, -0.022], p=0.0070 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 8.78e-09–8.78e-09 | 3/3 | 42: 8.78e-09 | -0.259 [-0.330, -0.186], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 7.09e-12–7.09e-12 | 3/3 | 42: 7.09e-12 | -0.273 [-0.346, -0.199], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.84e-28–1.1e-26 | 3/3 | 42: 2e-27, 100: 1.84e-28, 2026: 1.1e-26 | -0.389 [-0.451, -0.327], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 6.96e-29–4.19e-27 | 3/3 | 42: 4.19e-27, 100: 1.06e-27, 2026: 6.96e-29 | -0.376 [-0.435, -0.318], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.32–0.32 | 0/3 | 42: 0.32 | +0.049 [-0.005, +0.103], p=0.0790 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.000105–0.000105 | 3/3 | 42: 0.000105 | -0.130 [-0.193, -0.069], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.87e-07–4.87e-07 | 3/3 | 42: 4.87e-07 | -0.145 [-0.212, -0.081], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 1.33e-22–9.05e-21 | 3/3 | 42: 2.39e-22, 100: 1.33e-22, 2026: 9.05e-21 | -0.261 [-0.316, -0.208], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.06e-21–5.47e-21 | 3/3 | 42: 3.03e-21, 100: 5.47e-21, 2026: 1.06e-21 | -0.248 [-0.302, -0.197], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.94e-06–6.94e-06 | 3/3 | 42: 6.94e-06 | -0.179 [-0.250, -0.112], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 1.93e-08–1.93e-08 | 3/3 | 42: 1.93e-08 | -0.194 [-0.267, -0.123], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 5.46e-23–5.18e-22 | 3/3 | 42: 9.6e-23, 100: 5.46e-23, 2026: 5.18e-22 | -0.310 [-0.374, -0.249], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.4e-23–5.35e-22 | 3/3 | 42: 5.35e-22, 100: 3.95e-22, 2026: 3.4e-23 | -0.297 [-0.358, -0.240], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.0169–0.0169 | 0/3 | 42: 0.0169 | -0.014 [-0.037, +0.008], p=0.2086 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.18e-11–1.96e-09 | 3/3 | 42: 2.05e-11, 100: 1.18e-11, 2026: 1.96e-09 | -0.130 [-0.175, -0.087], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 5.26e-09–3.32e-08 | 3/3 | 42: 8.54e-09, 100: 3.32e-08, 2026: 5.26e-09 | -0.118 [-0.163, -0.074], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 4.04e-09–4.09e-07 | 3/3 | 42: 6.9e-09, 100: 4.04e-09, 2026: 4.09e-07 | -0.116 [-0.160, -0.073], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.65e-06–5.27e-06 | 3/3 | 42: 1.65e-06, 100: 5.27e-06, 2026: 1.65e-06 | -0.103 [-0.149, -0.060], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.169–0.701 | 0/3 | 42: 0.327, 100: 0.169, 2026: 0.701 | +0.012 [-0.014, +0.039], p=0.3454 |

### sentences_50agree (N = 726)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.0337–0.0337 | 0/3 | 42: 0.0337 | -0.012 [-0.062, +0.040], p=0.6714 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 8.51e-13–8.51e-13 | 3/3 | 42: 8.51e-13 | -0.155 [-0.211, -0.097], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.12e-05–2.12e-05 | 3/3 | 42: 2.12e-05 | -0.077 [-0.130, -0.022], p=0.0042 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 8.61e-22–8.61e-22 | 3/3 | 42: 8.61e-22 | -0.271 [-0.331, -0.209], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 1.97e-22–1.97e-22 | 3/3 | 42: 1.97e-22 | -0.278 [-0.339, -0.214], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 1.28e-47–1.59e-43 | 3/3 | 42: 1.28e-47, 100: 1.59e-43, 2026: 4.83e-44 | -0.382 [-0.432, -0.331], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.65e-44–8.73e-37 | 3/3 | 42: 1.65e-44, 100: 1.47e-37, 2026: 8.73e-37 | -0.355 [-0.407, -0.303], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 1.46e-08–1.46e-08 | 3/3 | 42: 1.46e-08 | -0.143 [-0.197, -0.090], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.00186–0.00186 | 3/3 | 42: 0.00186 | -0.065 [-0.110, -0.020], p=0.0054 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 7.57e-16–7.57e-16 | 3/3 | 42: 7.57e-16 | -0.259 [-0.318, -0.199], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 1.93e-17–1.93e-17 | 3/3 | 42: 1.93e-17 | -0.266 [-0.323, -0.207], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.7e-38–2.67e-30 | 3/3 | 42: 1.7e-38, 100: 2.67e-30, 2026: 6.08e-34 | -0.370 [-0.420, -0.319], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 2.91e-37–1.11e-27 | 3/3 | 42: 2.91e-37, 100: 1.79e-31, 2026: 1.11e-27 | -0.344 [-0.395, -0.292], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 6.52e-05–6.52e-05 | 3/3 | 42: 6.52e-05 | +0.078 [+0.035, +0.121], p=0.0006 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.00101–0.00101 | 3/3 | 42: 0.00101 | -0.116 [-0.172, -0.062], p=0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 0.000227–0.000227 | 3/3 | 42: 0.000227 | -0.123 [-0.179, -0.069], p=0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 2.27e-18–5.91e-12 | 3/3 | 42: 2.27e-18, 100: 5.91e-12, 2026: 3.73e-15 | -0.227 [-0.276, -0.180], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.82e-15–2.11e-11 | 3/3 | 42: 2.82e-15, 100: 1.21e-12, 2026: 2.11e-11 | -0.201 [-0.250, -0.153], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.05e-10–3.05e-10 | 3/3 | 42: 3.05e-10 | -0.194 [-0.250, -0.139], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 3.59e-11–3.59e-11 | 3/3 | 42: 3.59e-11 | -0.201 [-0.257, -0.145], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 2e-28–3.07e-22 | 3/3 | 42: 2e-28, 100: 3.07e-22, 2026: 2.24e-25 | -0.305 [-0.355, -0.254], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.07e-29–1.52e-21 | 3/3 | 42: 3.07e-29, 100: 1.14e-24, 2026: 1.52e-21 | -0.279 [-0.327, -0.232], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.359–0.359 | 0/3 | 42: 0.359 | -0.007 [-0.023, +0.009], p=0.4026 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.29e-09–2.2e-05 | 3/3 | 42: 1.29e-09, 100: 2.2e-05, 2026: 7.34e-08 | -0.110 [-0.148, -0.074], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 8.46e-08–0.00029 | 3/3 | 42: 8.46e-08, 100: 1.39e-05, 2026: 0.00029 | -0.084 [-0.124, -0.046], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.94e-08–0.00015 | 3/3 | 42: 1.94e-08, 100: 0.00015, 2026: 7.88e-07 | -0.104 [-0.143, -0.067], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.14e-06–0.00156 | 3/3 | 42: 1.14e-06, 100: 9.65e-05, 2026: 0.00156 | -0.078 [-0.119, -0.040], p=0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.0503–0.771 | 0/3 | 42: 0.389, 100: 0.771, 2026: 0.0503 | +0.026 [-0.005, +0.056], p=0.0964 |

## Paired comparisons: SEntFiN 1.0

### sentences_allagree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | 3/3 | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 4.42e-70–4.42e-70 | 3/3 | 42: 4.42e-70 | -0.099 [-0.112, -0.087], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | 3/3 | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 3.24e-05–3.24e-05 | 3/3 | 42: 3.24e-05 | +0.074 [+0.059, +0.089], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.000122–0.000122 | 3/3 | 42: 0.000122 | +0.075 [+0.061, +0.090], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 9.65e-172–1.97e-114 | 3/3 | 42: 9.76e-120, 100: 9.65e-172, 2026: 1.97e-114 | -0.160 [-0.184, -0.139], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.71e-155–3.17e-77 | 3/3 | 42: 1.71e-155, 100: 3.17e-77, 2026: 7.55e-106 | -0.144 [-0.173, -0.116], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 2.58e-125–2.58e-125 | 3/3 | 42: 2.58e-125 | -0.158 [-0.170, -0.146], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | 3/3 | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.508–0.508 | 0/3 | 42: 0.508 | +0.015 [-0.000, +0.030], p=0.0534 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.762–0.762 | 0/3 | 42: 0.762 | +0.016 [+0.002, +0.031], p=0.0296 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 4.78e-242–3.57e-193 | 3/3 | 42: 5.51e-197, 100: 4.78e-242, 2026: 3.57e-193 | -0.219 [-0.243, -0.198], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 8.23e-228–8e-144 | 3/3 | 42: 8.23e-228, 100: 8e-144, 2026: 2.42e-182 | -0.203 [-0.231, -0.175], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 7.85e-22–7.85e-22 | 3/3 | 42: 7.85e-22 | +0.039 [+0.030, +0.048], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.74e-158–3.74e-158 | 3/3 | 42: 3.74e-158 | +0.173 [+0.161, +0.185], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 3.47e-152–3.47e-152 | 3/3 | 42: 3.47e-152 | +0.175 [+0.163, +0.186], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 8.71e-42–1.81e-13 | 3/3 | 42: 3.62e-15, 100: 8.71e-42, 2026: 1.81e-13 | -0.061 [-0.084, -0.041], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.82e-33–0.0155 | 2/3 | 42: 1.82e-33, 100: 0.0155, 2026: 8.53e-11 | -0.045 [-0.073, -0.018], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 2.86e-55–2.86e-55 | 3/3 | 42: 2.86e-55 | +0.134 [+0.120, +0.147], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 6.18e-53–6.18e-53 | 3/3 | 42: 6.18e-53 | +0.135 [+0.121, +0.149], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 1.01e-88–9.01e-49 | 3/3 | 42: 1.09e-52, 100: 1.01e-88, 2026: 9.01e-49 | -0.100 [-0.123, -0.080], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 4.79e-78–1.89e-23 | 3/3 | 42: 4.79e-78, 100: 1.89e-23, 2026: 3.25e-43 | -0.084 [-0.113, -0.057], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.284–0.284 | 0/3 | 42: 0.284 | +0.002 [-0.003, +0.006], p=0.4728 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 2.56e-265–1.32e-214 | 3/3 | 42: 1.32e-214, 100: 2.56e-265, 2026: 2.39e-215 | -0.234 [-0.257, -0.214], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 1.72e-260–6.56e-184 | 3/3 | 42: 1.72e-260, 100: 6.56e-184, 2026: 1.22e-225 | -0.218 [-0.246, -0.190], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.86e-257–2.49e-210 | 3/3 | 42: 2.49e-210, 100: 1.86e-257, 2026: 5.37e-211 | -0.235 [-0.259, -0.215], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.22e-256–2.91e-183 | 3/3 | 42: 1.22e-256, 100: 2.91e-183, 2026: 2.08e-224 | -0.219 [-0.247, -0.192], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 5.96e-36–0.388 | 2/3 | 42: 4.78e-06, 100: 5.96e-36, 2026: 0.388 | +0.016 [-0.016, +0.049], p=0.3484 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | 3/3 | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | 3/3 | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 1.63e-56–1.63e-56 | 3/3 | 42: 1.63e-56 | +0.103 [+0.092, +0.114], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | 3/3 | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 1.5e-279–1.5e-279 | 3/3 | 42: 1.5e-279 | +0.276 [+0.264, +0.288], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 4.93e-272–4.93e-272 | 3/3 | 42: 4.93e-272 | +0.278 [+0.265, +0.290], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 8.81e-29–1.99e-05 | 3/3 | 42: 3.2e-26, 100: 1.99e-05, 2026: 8.81e-29 | +0.042 [+0.020, +0.061], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 6.49e-51–6.12e-09 | 3/3 | 42: 6.12e-09, 100: 6.49e-51, 2026: 1.47e-29 | +0.058 [+0.031, +0.085], p < 0.0002 |

### sentences_75agree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | 3/3 | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 2.25e-72–2.25e-72 | 3/3 | 42: 2.25e-72 | -0.100 [-0.112, -0.088], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | 3/3 | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 0.0885–0.0885 | 0/3 | 42: 0.0885 | +0.050 [+0.035, +0.065], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.0995–0.0995 | 0/3 | 42: 0.0995 | +0.053 [+0.038, +0.068], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 1.98e-130–8.07e-96 | 3/3 | 42: 6.97e-117, 100: 1.98e-130, 2026: 8.07e-96 | -0.151 [-0.167, -0.134], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 2.01e-157–4.27e-48 | 3/3 | 42: 7.96e-92, 100: 2.01e-157, 2026: 4.27e-48 | -0.128 [-0.173, -0.083], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 9.62e-129–9.62e-129 | 3/3 | 42: 9.62e-129 | -0.159 [-0.171, -0.147], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | 3/3 | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.0344–0.0344 | 0/3 | 42: 0.0344 | -0.009 [-0.023, +0.006], p=0.2366 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.0255–0.0255 | 0/3 | 42: 0.0255 | -0.006 [-0.020, +0.008], p=0.4176 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 5.48e-214–3.51e-168 | 3/3 | 42: 2.5e-194, 100: 5.48e-214, 2026: 3.51e-168 | -0.209 [-0.225, -0.193], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 3.19e-249–1.25e-104 | 3/3 | 42: 1.56e-163, 100: 3.19e-249, 2026: 1.25e-104 | -0.187 [-0.231, -0.141], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 1.67e-23–1.67e-23 | 3/3 | 42: 1.67e-23 | +0.040 [+0.031, +0.049], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.32e-127–6.32e-127 | 3/3 | 42: 6.32e-127 | +0.150 [+0.139, +0.161], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.96e-127–4.96e-127 | 3/3 | 42: 4.96e-127 | +0.153 [+0.142, +0.164], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 4.64e-21–6.66e-08 | 3/3 | 42: 1.3e-14, 100: 4.64e-21, 2026: 6.66e-08 | -0.050 [-0.066, -0.035], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 6.2e-32–0.00211 | 3/3 | 42: 5.81e-05, 100: 6.2e-32, 2026: 0.00211 | -0.028 [-0.072, +0.016], p=0.2432 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.24e-38–3.24e-38 | 3/3 | 42: 3.24e-38 | +0.110 [+0.097, +0.124], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 1.47e-38–1.47e-38 | 3/3 | 42: 1.47e-38 | +0.113 [+0.099, +0.127], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 1.38e-62–4.07e-38 | 3/3 | 42: 3.29e-52, 100: 1.38e-62, 2026: 4.07e-38 | -0.091 [-0.107, -0.074], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.6e-82–3.19e-07 | 3/3 | 42: 6.65e-32, 100: 1.6e-82, 2026: 3.19e-07 | -0.068 [-0.113, -0.023], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.864–0.864 | 0/3 | 42: 0.864 | +0.003 [-0.001, +0.007], p=0.1878 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 3.33e-209–1.79e-165 | 3/3 | 42: 6.87e-190, 100: 3.33e-209, 2026: 1.79e-165 | -0.201 [-0.217, -0.185], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 7.37e-235–1.34e-92 | 3/3 | 42: 1.9e-155, 100: 7.37e-235, 2026: 1.34e-92 | -0.178 [-0.222, -0.133], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 3.54e-211–2.54e-167 | 3/3 | 42: 1.16e-192, 100: 3.54e-211, 2026: 2.54e-167 | -0.203 [-0.219, -0.187], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.48e-238–9.51e-96 | 3/3 | 42: 6.24e-160, 100: 1.48e-238, 2026: 9.51e-96 | -0.181 [-0.225, -0.136], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 4.59e-22–0.00493 | 3/3 | 42: 7.63e-06, 100: 0.00493, 2026: 4.59e-22 | +0.023 [-0.023, +0.068], p=0.3502 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | 3/3 | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | 3/3 | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 1.9e-54–1.9e-54 | 3/3 | 42: 1.9e-54 | +0.102 [+0.091, +0.114], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | 3/3 | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 1.94e-248–1.94e-248 | 3/3 | 42: 1.94e-248 | +0.253 [+0.240, +0.265], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 8.37e-247–8.37e-247 | 3/3 | 42: 8.37e-247 | +0.255 [+0.243, +0.268], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 2.34e-37–6.09e-18 | 3/3 | 42: 5.81e-25, 100: 6.09e-18, 2026: 2.34e-37 | +0.052 [+0.038, +0.066], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 3.57e-78–3.07e-08 | 3/3 | 42: 6.8e-38, 100: 3.07e-08, 2026: 3.57e-78 | +0.075 [+0.030, +0.120], p < 0.0002 |

### sentences_50agree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | 3/3 | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 3.39e-69–3.39e-69 | 3/3 | 42: 3.39e-69 | -0.099 [-0.111, -0.087], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | 3/3 | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 0.000187–0.000187 | 3/3 | 42: 0.000187 | +0.050 [+0.036, +0.064], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.000199–0.000199 | 3/3 | 42: 0.000199 | +0.056 [+0.042, +0.070], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 4.58e-140–7.04e-92 | 3/3 | 42: 1.26e-126, 100: 4.58e-140, 2026: 7.04e-92 | -0.151 [-0.171, -0.130], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.9e-182–5.64e-150 | 3/3 | 42: 1.9e-182, 100: 5.43e-168, 2026: 5.64e-150 | -0.179 [-0.194, -0.164], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 3.37e-124–3.37e-124 | 3/3 | 42: 3.37e-124 | -0.158 [-0.170, -0.146], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | 3/3 | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.944–0.944 | 0/3 | 42: 0.944 | -0.009 [-0.023, +0.005], p=0.2142 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.971–0.971 | 0/3 | 42: 0.971 | -0.003 [-0.017, +0.011], p=0.6722 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.11e-201–2.81e-142 | 3/3 | 42: 1.11e-201, 100: 8.48e-198, 2026: 2.81e-142 | -0.210 [-0.230, -0.189], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 7.98e-264–5.34e-227 | 3/3 | 42: 2.11e-255, 100: 7.98e-264, 2026: 5.34e-227 | -0.238 [-0.253, -0.223], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 3.19e-21–3.19e-21 | 3/3 | 42: 3.19e-21 | +0.039 [+0.030, +0.048], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 2.89e-136–2.89e-136 | 3/3 | 42: 2.89e-136 | +0.149 [+0.137, +0.160], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 8.77e-138–8.77e-138 | 3/3 | 42: 8.77e-138 | +0.155 [+0.143, +0.167], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 2.19e-22–7.87e-06 | 3/3 | 42: 1.14e-17, 100: 2.19e-22, 2026: 7.87e-06 | -0.052 [-0.072, -0.031], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.73e-41–5.71e-25 | 3/3 | 42: 2.73e-41, 100: 9.72e-35, 2026: 5.71e-25 | -0.080 [-0.094, -0.066], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.32e-53–6.32e-53 | 3/3 | 42: 6.32e-53 | +0.110 [+0.096, +0.123], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 2.74e-54–2.74e-54 | 3/3 | 42: 2.74e-54 | +0.116 [+0.102, +0.129], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 9.49e-61–2.46e-29 | 3/3 | 42: 3.72e-54, 100: 9.49e-61, 2026: 2.46e-29 | -0.091 [-0.111, -0.070], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.63e-96–1.03e-69 | 3/3 | 42: 3.63e-96, 100: 1.09e-88, 2026: 1.03e-69 | -0.119 [-0.133, -0.104], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.959–0.959 | 0/3 | 42: 0.959 | +0.006 [+0.001, +0.010], p=0.0064 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.25e-207–2.59e-154 | 3/3 | 42: 8.54e-204, 100: 1.25e-207, 2026: 2.59e-154 | -0.201 [-0.221, -0.180], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 2.87e-262–1.63e-232 | 3/3 | 42: 2.87e-262, 100: 2.35e-252, 2026: 1.63e-232 | -0.229 [-0.244, -0.214], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.24e-206–3.59e-155 | 3/3 | 42: 1.27e-205, 100: 1.24e-206, 2026: 3.59e-155 | -0.207 [-0.227, -0.186], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 4.63e-263–9.48e-234 | 3/3 | 42: 4.63e-263, 100: 7.12e-255, 2026: 9.48e-234 | -0.235 [-0.250, -0.220], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 7.49e-10–0.00561 | 3/3 | 42: 7.49e-10, 100: 0.00561, 2026: 8.63e-10 | -0.028 [-0.049, -0.007], p=0.0042 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | 3/3 | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | 3/3 | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 2.67e-57–2.67e-57 | 3/3 | 42: 2.67e-57 | +0.103 [+0.092, +0.115], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | 3/3 | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 4.19e-275–4.19e-275 | 3/3 | 42: 4.19e-275 | +0.252 [+0.240, +0.265], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 2.84e-273–2.84e-273 | 3/3 | 42: 2.84e-273 | +0.258 [+0.246, +0.271], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 8.09e-42–5.71e-16 | 3/3 | 42: 4.59e-22, 100: 5.71e-16, 2026: 8.09e-42 | +0.051 [+0.034, +0.071], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 1.08e-12–0.00259 | 3/3 | 42: 0.00259, 100: 1.81e-06, 2026: 1.08e-12 | +0.024 [+0.011, +0.037], p=0.0004 |

## Figures

- `step2_macro_f1_by_config.png`
- `step2_external_generalization.png`
- `step2_attribution.png`
- `step2_confusion_matrices_allagree.png`
