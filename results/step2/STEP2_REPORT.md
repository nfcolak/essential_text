# Step 2 Report: Transformer Fine-Tuning vs Baselines

Generated 2026-10-20 22:14 UTC from `results/step2/runs`.

## Protocol

- Both encoders (`bert_base`, `finbert_mlm`) use one shared fixed recipe (AdamW, linear warmup, same batch size, learning rate and epoch budget) on the Step 1 locked train split of each config.
- The epoch is selected by validation Macro-F1 only; the test split and SEntFiN are scored once with the selected checkpoint.
- Fine-tuned models run with several seeds; tables show mean ± sample sd over seeds. Baselines are single runs with seed 42 (`seed42`; sd shown as n/a): the lexicons are deterministic, the TF-IDF models are deterministic given that seed.
- Baseline fitting: the plain official lexicons (VADER, Loughran-McDonald) use fixed published rules and nothing is tuned. The `official, val-tuned` variants keep the official lexicons but tune their decision thresholds on the validation split of each config (the LM variant also applies 3-token negation handling), as specified in the proposal. The finance-adjusted lexicon variants tune their thresholds the same way, and the TF-IDF models tune C in {0.01, 0.1, 1, 10} and the class weight on validation per config. For Loughran-McDonald the discrete score (P - N) / (P + N + 1) makes every threshold pair in (0, 0.25) give the same predictions, so the sweep kept the default +/-0.05 in every config and fold: the val-tuned LM variant differs from the plain one by negation handling only.
- TF-IDF + Linear SVM is `LinearSVC` wrapped in Platt (sigmoid) calibration fitted on the training features; its predictions are the argmax of the calibrated one-vs-rest probabilities and its ECE is train-calibrated, so it is not directly comparable with the ECE of the other models.
- The 5-fold CV on `sentences_allagree` is scored on the fold validation part, the same part used for epoch selection (as in Step 1 CV), so it is an optimistic estimate and not a test score.
- The three PhraseBank levels (allagree, 75agree, 50agree) are nested: a sentence in a stricter level is also in the looser ones. Models and test splits are never mixed across levels, so tables are compared within a config and across configs only descriptively.
- `prosus_finbert` (ProsusAI/finbert) is zero-shot and scored only on SEntFiN 1.0 (it was fine-tuned on Financial PhraseBank, so scoring it there would leak).
- Confidence intervals include seed variation: the bootstrap resamples sentences and, for the fine-tuned models, the trained seeds (two-level, seed+sentence). The earlier sentence-only interval, conditional on the three seeds, is stored as `conditional_on_seeds` in `attribution.json`.
- Bootstrap p-value (B = 10000): p = min(1, 2 * min(P(d <= 0), P(d >= 0))) over the resampled differences d, so the smallest non-zero value is 2/B; 0 is printed as p < 0.0002.
- McNemar is a per-seed diagnostic. There is no primary seed: each seed pairing forms its own family, Holm-Bonferroni is applied within each family (pairs of single-run models enter every family with their one run), and the report gives p_min, p_max and the number of seed families with Holm-adjusted p < 0.05. A pair of two single-run models (lexicons, TF-IDF) is one test, not a replication; its family count is labelled `single run`. McNemar p-values are the exact binomial test on the discordant pairs; the continuity-corrected chi-squared is stored alongside. The seed+sentence bootstrap of the Macro-F1 difference is the headline test.
- `runs/journal.jsonl` is an append-only event log: it also records the six runs that hit Apple MPS memory errors and were rerun, one deliberately stopped run and the later baseline refits (TF-IDF after the C grid gained 0.01, the val-tuned LM after a tokenisation fix; predictions unchanged). The `result.json` files are authoritative.
- RQ1 boundary: R_neu is the recall of gold-neutral sentences and P_neg the precision of the negative class (seed means; deltas use the seed+sentence bootstrap). The McNemar test in that section is restricted to gold-neutral sentences (correct = predicted neutral), with Holm within each seed family over the pairs of one config and split.
- Per-class precision, recall and F1 are the seed mean ± sd of the values stored in each `result.json` (`metrics.<split>.per_class`).
- Attribution: `delta_arch` = BERT − `tfidf_best` and `delta_domain` = FinBERT-MLM − BERT; `tfidf_best` is the TF-IDF model with the higher validation Macro-F1 in that config. Per-seed deltas are listed beside each interval.

## In-domain test: Macro-F1 / accuracy (mean ± sd)

### sentences_allagree

| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |
|---|---|---|---|---|---|
| VADER (official lexicon) | 1 | 0.500 (sd n/a) | 0.590 (sd n/a) | 0.077 (sd n/a) | 0.406 (sd n/a) |
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.507 (sd n/a) | 0.631 (sd n/a) | 0.038 (sd n/a) | 0.288 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.498 (sd n/a) | 0.652 (sd n/a) | 0.072 (sd n/a) | 0.274 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.505 (sd n/a) | 0.652 (sd n/a) | 0.038 (sd n/a) | 0.344 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.542 (sd n/a) | 0.629 (sd n/a) | 0.065 (sd n/a) | 0.303 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.518 (sd n/a) | 0.666 (sd n/a) | 0.084 (sd n/a) | 0.262 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.520 (sd n/a) | 0.666 (sd n/a) | 0.068 (sd n/a) | 0.330 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.506 (sd n/a) | 0.588 (sd n/a) | 0.058 (sd n/a) | 0.310 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.502 (sd n/a) | 0.614 (sd n/a) | 0.086 (sd n/a) | 0.319 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.503 (sd n/a) | 0.614 (sd n/a) | 0.079 (sd n/a) | 0.382 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.481 (sd n/a) | 0.500 (sd n/a) | 0.056 (sd n/a) | 0.420 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.469 (sd n/a) | 0.502 (sd n/a) | 0.087 (sd n/a) | 0.494 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.528 (sd n/a) | 0.531 (sd n/a) | 0.100 (sd n/a) | 0.298 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.469 (sd n/a) | 0.502 (sd n/a) | 0.087 (sd n/a) | 0.494 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 1 | 0.525 (sd n/a) | 0.529 (sd n/a) | 0.098 (sd n/a) | 0.369 (sd n/a) |
| Loughran-McDonald (official dictionary) | 1 | 0.466 (sd n/a) | 0.500 (sd n/a) | 0.090 (sd n/a) | 0.433 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1 | 0.469 (sd n/a) | 0.502 (sd n/a) | 0.087 (sd n/a) | 0.494 (sd n/a) |
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
| VADER (official lexicon, val-tuned thresholds) | 5 | 0.529 ± 0.039 | 0.643 ± 0.033 |
| Loughran-McDonald (official dictionary) | 5 | 0.480 ± 0.026 | 0.649 ± 0.019 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 5 | 0.483 ± 0.026 | 0.650 ± 0.020 |
| VADER-style (finance-adjusted, custom) | 5 | 0.716 ± 0.041 | 0.800 ± 0.027 |
| LM-style (finance-adjusted, custom) | 5 | 0.618 ± 0.030 | 0.732 ± 0.021 |
| TF-IDF + Logistic Regression | 5 | 0.806 ± 0.028 | 0.864 ± 0.015 |
| TF-IDF + Linear SVM | 5 | 0.816 ± 0.028 | 0.873 ± 0.014 |
| bert-base-uncased | 5 | 0.950 ± 0.013 | 0.962 ± 0.013 |
| FinBERT-MLM (finbert-pretrain) | 5 | 0.952 ± 0.013 | 0.967 ± 0.009 |

## Per-class precision / recall / F1 (sentences_allagree)

### In-domain test

| model | negative P | negative R | negative F1 | neutral P | neutral R | neutral F1 | positive P | positive R | positive F1 |
|---|---|---|---|---|---|---|---|---|---|
| VADER (official lexicon) | 0.345 (sd n/a) | 0.217 (sd n/a) | 0.267 (sd n/a) | 0.831 (sd n/a) | 0.591 (sd n/a) | 0.691 (sd n/a) | 0.414 (sd n/a) | 0.788 (sd n/a) | 0.543 (sd n/a) |
| VADER (official lexicon, val-tuned thresholds) | 0.421 (sd n/a) | 0.174 (sd n/a) | 0.246 (sd n/a) | 0.745 (sd n/a) | 0.731 (sd n/a) | 0.738 (sd n/a) | 0.466 (sd n/a) | 0.635 (sd n/a) | 0.537 (sd n/a) |
| Loughran-McDonald (official dictionary) | 0.475 (sd n/a) | 0.413 (sd n/a) | 0.442 (sd n/a) | 0.675 (sd n/a) | 0.899 (sd n/a) | 0.771 (sd n/a) | 0.682 (sd n/a) | 0.176 (sd n/a) | 0.280 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.576 (sd n/a) | 0.413 (sd n/a) | 0.481 (sd n/a) | 0.675 (sd n/a) | 0.899 (sd n/a) | 0.771 (sd n/a) | 0.517 (sd n/a) | 0.176 (sd n/a) | 0.263 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 0.771 (sd n/a) | 0.587 (sd n/a) | 0.667 (sd n/a) | 0.877 (sd n/a) | 0.923 (sd n/a) | 0.899 (sd n/a) | 0.694 (sd n/a) | 0.694 (sd n/a) | 0.694 (sd n/a) |
| LM-style (finance-adjusted, custom) | 0.643 (sd n/a) | 0.391 (sd n/a) | 0.486 (sd n/a) | 0.773 (sd n/a) | 0.885 (sd n/a) | 0.825 (sd n/a) | 0.644 (sd n/a) | 0.553 (sd n/a) | 0.595 (sd n/a) |
| TF-IDF + Logistic Regression | 0.830 (sd n/a) | 0.848 (sd n/a) | 0.839 (sd n/a) | 0.934 (sd n/a) | 0.952 (sd n/a) | 0.943 (sd n/a) | 0.850 (sd n/a) | 0.800 (sd n/a) | 0.824 (sd n/a) |
| TF-IDF + Linear SVM | 0.923 (sd n/a) | 0.783 (sd n/a) | 0.847 (sd n/a) | 0.922 (sd n/a) | 0.962 (sd n/a) | 0.941 (sd n/a) | 0.831 (sd n/a) | 0.812 (sd n/a) | 0.821 (sd n/a) |
| bert-base-uncased | 1.000 ± 0.000 | 0.942 ± 0.033 | 0.970 ± 0.018 | 0.989 ± 0.007 | 0.992 ± 0.010 | 0.990 ± 0.004 | 0.951 ± 0.038 | 0.973 ± 0.018 | 0.961 ± 0.018 |
| FinBERT-MLM (finbert-pretrain) | 0.955 ± 0.022 | 0.906 ± 0.013 | 0.929 ± 0.013 | 0.981 ± 0.008 | 0.978 ± 0.015 | 0.979 ± 0.005 | 0.910 ± 0.019 | 0.941 ± 0.020 | 0.925 ± 0.006 |

### SEntFiN 1.0 (allagree-trained)

| model | negative P | negative R | negative F1 | neutral P | neutral R | neutral F1 | positive P | positive R | positive F1 |
|---|---|---|---|---|---|---|---|---|---|
| VADER (official lexicon) | 0.656 (sd n/a) | 0.438 (sd n/a) | 0.525 (sd n/a) | 0.481 (sd n/a) | 0.590 (sd n/a) | 0.530 (sd n/a) | 0.511 (sd n/a) | 0.529 (sd n/a) | 0.520 (sd n/a) |
| VADER (official lexicon, val-tuned thresholds) | 0.704 (sd n/a) | 0.284 (sd n/a) | 0.405 (sd n/a) | 0.429 (sd n/a) | 0.729 (sd n/a) | 0.540 (sd n/a) | 0.573 (sd n/a) | 0.441 (sd n/a) | 0.499 (sd n/a) |
| Loughran-McDonald (official dictionary) | 0.687 (sd n/a) | 0.397 (sd n/a) | 0.503 (sd n/a) | 0.424 (sd n/a) | 0.871 (sd n/a) | 0.571 (sd n/a) | 0.761 (sd n/a) | 0.206 (sd n/a) | 0.324 (sd n/a) |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.696 (sd n/a) | 0.402 (sd n/a) | 0.509 (sd n/a) | 0.425 (sd n/a) | 0.871 (sd n/a) | 0.571 (sd n/a) | 0.767 (sd n/a) | 0.207 (sd n/a) | 0.327 (sd n/a) |
| VADER-style (finance-adjusted, custom) | 0.841 (sd n/a) | 0.438 (sd n/a) | 0.576 (sd n/a) | 0.533 (sd n/a) | 0.864 (sd n/a) | 0.659 (sd n/a) | 0.742 (sd n/a) | 0.559 (sd n/a) | 0.638 (sd n/a) |
| LM-style (finance-adjusted, custom) | 0.799 (sd n/a) | 0.470 (sd n/a) | 0.592 (sd n/a) | 0.490 (sd n/a) | 0.846 (sd n/a) | 0.620 (sd n/a) | 0.729 (sd n/a) | 0.432 (sd n/a) | 0.543 (sd n/a) |
| TF-IDF + Logistic Regression | 0.783 (sd n/a) | 0.223 (sd n/a) | 0.347 (sd n/a) | 0.432 (sd n/a) | 0.880 (sd n/a) | 0.580 (sd n/a) | 0.624 (sd n/a) | 0.326 (sd n/a) | 0.428 (sd n/a) |
| TF-IDF + Linear SVM | 0.824 (sd n/a) | 0.226 (sd n/a) | 0.354 (sd n/a) | 0.430 (sd n/a) | 0.913 (sd n/a) | 0.585 (sd n/a) | 0.668 (sd n/a) | 0.296 (sd n/a) | 0.410 (sd n/a) |
| bert-base-uncased | 0.855 ± 0.008 | 0.615 ± 0.032 | 0.715 ± 0.019 | 0.576 ± 0.029 | 0.858 ± 0.036 | 0.688 ± 0.008 | 0.781 ± 0.021 | 0.563 ± 0.065 | 0.652 ± 0.035 |
| FinBERT-MLM (finbert-pretrain) | 0.848 ± 0.010 | 0.561 ± 0.046 | 0.674 ± 0.031 | 0.548 ± 0.031 | 0.879 ± 0.043 | 0.674 ± 0.011 | 0.843 ± 0.040 | 0.546 ± 0.078 | 0.659 ± 0.044 |

## RQ1: neutral/negative boundary

R_neu = recall of gold-neutral sentences, P_neg = precision of the negative class (seed means); Δ = A − B with the 95% seed+sentence bootstrap CI. McNemar covers gold-neutral sentences only.

### In-domain test: sentences_allagree (N = 339)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.591 / 0.899 | 0.345 / 0.475 | -0.308 [-0.390, -0.225], p < 0.0002 | -0.130 [-0.321, +0.056], p=0.1738 | 2.07e-11–2.07e-11 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.731 / 0.899 | 0.421 / 0.576 | -0.168 [-0.244, -0.093], p < 0.0002 | -0.155 [-0.386, +0.071], p=0.1738 | 2.93e-05–2.93e-05 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.923 / 0.885 | 0.771 / 0.643 | +0.038 [+0.000, +0.080], p=0.0680 | +0.129 [-0.029, +0.292], p=0.1150 | 0.0963–0.0963 | single run (0/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.992 / 0.978 | 1.000 / 0.955 | +0.014 [-0.008, +0.040], p=0.2386 | +0.045 [+0.000, +0.116], p=0.1722 | 0.0156–0.688 | 1/3 |

### In-domain test: sentences_75agree (N = 518)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.556 / 0.891 | 0.426 / 0.394 | -0.335 [-0.397, -0.274], p < 0.0002 | +0.032 [-0.111, +0.177], p=0.6746 | 1.67e-21–1.67e-21 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.699 / 0.891 | 0.429 / 0.426 | -0.193 [-0.252, -0.135], p < 0.0002 | +0.002 [-0.139, +0.145], p=0.9880 | 1.02e-09–1.02e-09 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.885 / 0.885 | 0.673 / 0.595 | +0.000 [-0.039, +0.038], p=1.0000 | +0.077 [-0.058, +0.215], p=0.2580 | 1–1 | single run (0/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.953 / 0.964 | 0.865 / 0.883 | -0.010 [-0.030, +0.008], p=0.2924 | -0.018 [-0.084, +0.049], p=0.5916 | 0.00781–1 | 1/3 |

### In-domain test: sentences_50agree (N = 726)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.512 / 0.821 | 0.393 / 0.402 | -0.309 [-0.366, -0.252], p < 0.0002 | -0.009 [-0.129, +0.115], p=0.8810 | 1.56e-22–1.56e-22 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.612 / 0.821 | 0.393 / 0.416 | -0.209 [-0.264, -0.154], p < 0.0002 | -0.023 [-0.144, +0.102], p=0.7176 | 1.35e-12–1.35e-12 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.879 / 0.823 | 0.639 / 0.486 | +0.056 [+0.021, +0.092], p=0.0018 | +0.154 [+0.058, +0.252], p=0.0014 | 0.00369–0.00369 | single run (3/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.881 / 0.881 | 0.906 / 0.827 | -0.001 [-0.056, +0.049], p=0.9996 | +0.079 [-0.012, +0.169], p=0.0876 | 0.00263–0.337 | 2/3 |

### SEntFiN 1.0: sentences_allagree (N = 9460)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.590 / 0.871 | 0.656 / 0.687 | -0.281 [-0.298, -0.263], p < 0.0002 | -0.031 [-0.055, -0.009], p=0.0064 | 6.36e-197–6.36e-197 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.729 / 0.871 | 0.704 / 0.696 | -0.142 [-0.158, -0.126], p < 0.0002 | +0.008 [-0.017, +0.033], p=0.5004 | 5.12e-65–5.12e-65 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.864 / 0.846 | 0.841 / 0.799 | +0.018 [+0.007, +0.029], p=0.0016 | +0.042 [+0.024, +0.060], p < 0.0002 | 0.00123–0.00123 | single run (3/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.858 / 0.879 | 0.855 / 0.848 | -0.021 [-0.071, +0.031], p=0.4894 | +0.006 [-0.012, +0.025], p=0.5246 | 4.26e-47–1.27e-05 | 3/3 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 0.752 / 0.858 | 0.757 / 0.855 | -0.106 [-0.136, -0.068], p < 0.0002 | -0.097 [-0.114, -0.081], p < 0.0002 | 1.1e-78–6.35e-20 | 3/3 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 0.752 / 0.879 | 0.757 / 0.848 | -0.127 [-0.161, -0.081], p < 0.0002 | -0.091 [-0.110, -0.073], p < 0.0002 | 5.3e-104–4.19e-27 | 3/3 |

### SEntFiN 1.0: sentences_75agree (N = 9460)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.590 / 0.871 | 0.656 / 0.687 | -0.281 [-0.298, -0.263], p < 0.0002 | -0.031 [-0.055, -0.009], p=0.0064 | 6.36e-197–6.36e-197 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.685 / 0.871 | 0.657 / 0.696 | -0.186 [-0.202, -0.169], p < 0.0002 | -0.039 [-0.062, -0.017], p=0.0002 | 1.82e-104–1.82e-104 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.864 / 0.846 | 0.841 / 0.799 | +0.018 [+0.007, +0.029], p=0.0016 | +0.042 [+0.024, +0.060], p < 0.0002 | 0.00123–0.00123 | single run (3/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.888 / 0.913 | 0.861 / 0.884 | -0.025 [-0.066, +0.014], p=0.2420 | -0.023 [-0.073, +0.022], p=0.3766 | 1.97e-29–0.0946 | 2/3 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 0.752 / 0.888 | 0.757 / 0.861 | -0.136 [-0.151, -0.121], p < 0.0002 | -0.104 [-0.129, -0.075], p < 0.0002 | 5.01e-97–2.92e-80 | 3/3 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 0.752 / 0.913 | 0.757 / 0.884 | -0.161 [-0.201, -0.121], p < 0.0002 | -0.126 [-0.171, -0.089], p < 0.0002 | 7.7e-175–2.11e-64 | 3/3 |

### SEntFiN 1.0: sentences_50agree (N = 9460)

| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | neutral-only McNemar p range | families Holm p < 0.05 |
|---|---|---|---|---|---|---|
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.590 / 0.871 | 0.656 / 0.687 | -0.281 [-0.298, -0.263], p < 0.0002 | -0.031 [-0.055, -0.009], p=0.0064 | 6.36e-197–6.36e-197 | single run (3/3 families) |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.687 / 0.871 | 0.656 / 0.696 | -0.184 [-0.200, -0.168], p < 0.0002 | -0.040 [-0.062, -0.017], p=0.0004 | 1.39e-102–1.39e-102 | single run (3/3 families) |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.864 / 0.846 | 0.841 / 0.799 | +0.018 [+0.008, +0.029], p=0.0016 | +0.042 [+0.024, +0.060], p < 0.0002 | 0.00104–0.00104 | single run (3/3 families) |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.829 / 0.810 | 0.829 / 0.781 | +0.019 [-0.023, +0.062], p=0.4044 | +0.048 [+0.023, +0.076], p < 0.0002 | 5.86e-30–0.00413 | 3/3 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 0.752 / 0.829 | 0.757 / 0.829 | -0.077 [-0.115, -0.039], p < 0.0002 | -0.072 [-0.095, -0.050], p < 0.0002 | 1.99e-64–1.25e-07 | 3/3 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 0.752 / 0.810 | 0.757 / 0.781 | -0.058 [-0.082, -0.035], p < 0.0002 | -0.024 [-0.043, -0.003], p=0.0252 | 2e-27–1.86e-07 | 3/3 |

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
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 0.0488–0.0488 | single run (0/3 families) | 42: 0.0488 | -0.007 [-0.043, +0.032], p=0.6950 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.119–0.119 | single run (0/3 families) | 42: 0.119 | +0.002 [-0.080, +0.084], p=0.9514 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.119–0.119 | single run (0/3 families) | 42: 0.119 | -0.005 [-0.087, +0.077], p=0.9068 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 2.51e-14–2.51e-14 | single run (3/3 families) | 42: 2.51e-14 | -0.253 [-0.319, -0.189], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 8.18e-06–8.18e-06 | single run (3/3 families) | 42: 8.18e-06 | -0.135 [-0.198, -0.072], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 1.39e-18–1.39e-18 | single run (3/3 families) | 42: 1.39e-18 | -0.369 [-0.448, -0.283], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 3.8e-19–3.8e-19 | single run (3/3 families) | 42: 3.8e-19 | -0.370 [-0.451, -0.284], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 7.98e-40–6.21e-34 | 3/3 | 42: 4.4e-35, 100: 6.21e-34, 2026: 7.98e-40 | -0.474 [-0.538, -0.410], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 5.16e-32–8.6e-30 | 3/3 | 42: 8.6e-30, 100: 2.72e-31, 2026: 5.16e-32 | -0.444 [-0.510, -0.377], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 0.592–0.592 | single run (0/3 families) | 42: 0.592 | +0.009 [-0.068, +0.084], p=0.8086 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.592–0.592 | single run (0/3 families) | 42: 0.592 | +0.002 [-0.074, +0.076], p=0.9536 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 3.18e-12–3.18e-12 | single run (3/3 families) | 42: 3.18e-12 | -0.246 [-0.313, -0.182], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.000224–0.000224 | single run (3/3 families) | 42: 0.000224 | -0.128 [-0.191, -0.066], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 6.09e-16–6.09e-16 | single run (3/3 families) | 42: 6.09e-16 | -0.361 [-0.446, -0.273], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 3.32e-16–3.32e-16 | single run (3/3 families) | 42: 3.32e-16 | -0.363 [-0.448, -0.272], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 3.76e-37–5.98e-30 | 3/3 | 42: 5.55e-33, 100: 5.98e-30, 2026: 3.76e-37 | -0.467 [-0.534, -0.398], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 1.65e-30–5.74e-28 | 3/3 | 42: 5.74e-28, 100: 1.28e-29, 2026: 1.65e-30 | -0.437 [-0.506, -0.367], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1–1 | single run (0/3 families) | 42: 1 | -0.007 [-0.016, -0.002], p=0.0012 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 6.88e-09–6.88e-09 | single run (3/3 families) | 42: 6.88e-09 | -0.256 [-0.330, -0.182], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.00152–0.00152 | single run (3/3 families) | 42: 0.00152 | -0.138 [-0.212, -0.060], p=0.0004 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 1.25e-14–1.25e-14 | single run (3/3 families) | 42: 1.25e-14 | -0.371 [-0.457, -0.287], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 2.21e-14–2.21e-14 | single run (3/3 families) | 42: 2.21e-14 | -0.372 [-0.462, -0.282], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 4.81e-35–3.05e-27 | 3/3 | 42: 6.31e-31, 100: 3.05e-27, 2026: 4.81e-35 | -0.476 [-0.546, -0.410], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 7.73e-27–1.71e-23 | 3/3 | 42: 1.71e-23, 100: 7.73e-27, 2026: 3.98e-26 | -0.447 [-0.519, -0.376], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 6.88e-09–6.88e-09 | single run (3/3 families) | 42: 6.88e-09 | -0.248 [-0.322, -0.175], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.00152–0.00152 | single run (3/3 families) | 42: 0.00152 | -0.130 [-0.205, -0.054], p=0.0006 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 1.25e-14–1.25e-14 | single run (3/3 families) | 42: 1.25e-14 | -0.364 [-0.450, -0.279], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 2.21e-14–2.21e-14 | single run (3/3 families) | 42: 2.21e-14 | -0.365 [-0.455, -0.275], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 4.81e-35–3.05e-27 | 3/3 | 42: 6.31e-31, 100: 3.05e-27, 2026: 4.81e-35 | -0.469 [-0.539, -0.402], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 7.73e-27–1.71e-23 | 3/3 | 42: 1.71e-23, 100: 7.73e-27, 2026: 3.98e-26 | -0.439 [-0.512, -0.368], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 8.17e-05–8.17e-05 | single run (3/3 families) | 42: 8.17e-05 | +0.118 [+0.061, +0.178], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.00212–0.00212 | single run (3/3 families) | 42: 0.00212 | -0.115 [-0.187, -0.042], p=0.0032 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 0.00244–0.00244 | single run (3/3 families) | 42: 0.00244 | -0.117 [-0.192, -0.040], p=0.0038 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 1.06e-16–9.03e-12 | 3/3 | 42: 1.66e-12, 100: 9.03e-12, 2026: 1.06e-16 | -0.221 [-0.283, -0.161], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.92e-12–2.99e-09 | 3/3 | 42: 2.99e-09, 100: 2.92e-12, 2026: 3.5e-11 | -0.191 [-0.253, -0.132], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 8.31e-08–8.31e-08 | single run (3/3 families) | 42: 8.31e-08 | -0.233 [-0.322, -0.148], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.48e-08–4.48e-08 | single run (3/3 families) | 42: 4.48e-08 | -0.234 [-0.322, -0.149], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 2.91e-25–1.65e-19 | 3/3 | 42: 2.1e-20, 100: 1.65e-19, 2026: 2.91e-25 | -0.338 [-0.410, -0.272], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 5.08e-19–3.92e-16 | 3/3 | 42: 3.92e-16, 100: 5.08e-19, 2026: 5.08e-19 | -0.309 [-0.381, -0.241], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 1–1 | single run (0/3 families) | 42: 1 | -0.001 [-0.031, +0.029], p=0.9368 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 9.31e-10–1.09e-05 | 3/3 | 42: 2.56e-06, 100: 1.09e-05, 2026: 9.31e-10 | -0.105 [-0.151, -0.064], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 0.000324–0.00393 | 3/3 | 42: 0.00393, 100: 0.000324, 2026: 0.000508 | -0.076 [-0.121, -0.034], p=0.0008 |
| TF-IDF + Linear SVM vs bert-base-uncased | 9.31e-10–2.24e-05 | 3/3 | 42: 2.56e-06, 100: 2.24e-05, 2026: 9.31e-10 | -0.104 [-0.150, -0.062], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 0.000324–0.0051 | 3/3 | 42: 0.0051, 100: 0.000324, 2026: 0.000324 | -0.075 [-0.120, -0.033], p=0.0004 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.00195–0.424 | 1/3 | 42: 0.0386, 100: 0.424, 2026: 0.00195 | +0.029 [+0.007, +0.057], p=0.0114 |

### sentences_75agree (N = 518)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 0.00131–0.00131 | single run (3/3 families) | 42: 0.00131 | -0.027 [-0.053, -0.002], p=0.0296 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.00331–0.00331 | single run (3/3 families) | 42: 0.00331 | -0.003 [-0.068, +0.064], p=0.9380 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.00331–0.00331 | single run (3/3 families) | 42: 0.00331 | -0.005 [-0.071, +0.062], p=0.8860 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 1.31e-07–1.31e-07 | single run (3/3 families) | 42: 1.31e-07 | -0.131 [-0.195, -0.067], p=0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.21e-06–2.21e-06 | single run (3/3 families) | 42: 2.21e-06 | -0.082 [-0.143, -0.021], p=0.0082 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 3.44e-16–3.44e-16 | single run (3/3 families) | 42: 3.44e-16 | -0.261 [-0.333, -0.190], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 1.18e-19–1.18e-19 | single run (3/3 families) | 42: 1.18e-19 | -0.276 [-0.349, -0.202], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 2.33e-39–8.61e-39 | 3/3 | 42: 4.23e-39, 100: 2.33e-39, 2026: 8.61e-39 | -0.391 [-0.455, -0.327], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.14e-36–7.71e-36 | 3/3 | 42: 7.71e-36, 100: 2.05e-36, 2026: 1.14e-36 | -0.379 [-0.440, -0.317], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 0.183–0.183 | single run (0/3 families) | 42: 0.183 | +0.025 [-0.038, +0.088], p=0.4428 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.183–0.183 | single run (0/3 families) | 42: 0.183 | +0.023 [-0.040, +0.086], p=0.4824 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 8.15e-05–8.15e-05 | single run (3/3 families) | 42: 8.15e-05 | -0.103 [-0.165, -0.041], p=0.0014 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.00106–0.00106 | single run (3/3 families) | 42: 0.00106 | -0.054 [-0.114, +0.004], p=0.0714 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 1.27e-11–1.27e-11 | single run (3/3 families) | 42: 1.27e-11 | -0.234 [-0.308, -0.160], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 1.16e-14–1.16e-14 | single run (3/3 families) | 42: 1.16e-14 | -0.248 [-0.324, -0.173], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 1.95e-32–2e-31 | 3/3 | 42: 8.95e-32, 100: 1.95e-32, 2026: 2e-31 | -0.364 [-0.429, -0.300], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 4.41e-31–8.37e-29 | 3/3 | 42: 8.37e-29, 100: 1.22e-29, 2026: 4.41e-31 | -0.352 [-0.414, -0.289], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1–1 | single run (0/3 families) | 42: 1 | -0.002 [-0.005, -0.000], p=0.0184 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 0.00724–0.00724 | single run (0/3 families) | 42: 0.00724 | -0.128 [-0.193, -0.063], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.0263–0.0263 | single run (0/3 families) | 42: 0.0263 | -0.079 [-0.137, -0.022], p=0.0070 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 8.78e-09–8.78e-09 | single run (3/3 families) | 42: 8.78e-09 | -0.259 [-0.330, -0.186], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 7.09e-12–7.09e-12 | single run (3/3 families) | 42: 7.09e-12 | -0.273 [-0.346, -0.199], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.84e-28–1.1e-26 | 3/3 | 42: 2e-27, 100: 1.84e-28, 2026: 1.1e-26 | -0.389 [-0.451, -0.327], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 6.96e-29–4.19e-27 | 3/3 | 42: 4.19e-27, 100: 1.06e-27, 2026: 6.96e-29 | -0.376 [-0.435, -0.318], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 0.00724–0.00724 | single run (0/3 families) | 42: 0.00724 | -0.126 [-0.191, -0.061], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.0263–0.0263 | single run (0/3 families) | 42: 0.0263 | -0.077 [-0.135, -0.020], p=0.0086 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 8.78e-09–8.78e-09 | single run (3/3 families) | 42: 8.78e-09 | -0.257 [-0.328, -0.184], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 7.09e-12–7.09e-12 | single run (3/3 families) | 42: 7.09e-12 | -0.271 [-0.344, -0.197], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 1.84e-28–1.1e-26 | 3/3 | 42: 2e-27, 100: 1.84e-28, 2026: 1.1e-26 | -0.387 [-0.450, -0.325], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 6.96e-29–4.19e-27 | 3/3 | 42: 4.19e-27, 100: 1.06e-27, 2026: 6.96e-29 | -0.374 [-0.434, -0.315], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 0.32–0.32 | single run (0/3 families) | 42: 0.32 | +0.049 [-0.005, +0.103], p=0.0790 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.000105–0.000105 | single run (3/3 families) | 42: 0.000105 | -0.130 [-0.193, -0.069], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.87e-07–4.87e-07 | single run (3/3 families) | 42: 4.87e-07 | -0.145 [-0.212, -0.081], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 1.33e-22–9.05e-21 | 3/3 | 42: 2.39e-22, 100: 1.33e-22, 2026: 9.05e-21 | -0.261 [-0.316, -0.208], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.06e-21–5.47e-21 | 3/3 | 42: 3.03e-21, 100: 5.47e-21, 2026: 1.06e-21 | -0.248 [-0.302, -0.197], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.94e-06–6.94e-06 | single run (3/3 families) | 42: 6.94e-06 | -0.179 [-0.250, -0.112], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 1.93e-08–1.93e-08 | single run (3/3 families) | 42: 1.93e-08 | -0.194 [-0.267, -0.123], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 5.46e-23–5.18e-22 | 3/3 | 42: 9.6e-23, 100: 5.46e-23, 2026: 5.18e-22 | -0.310 [-0.374, -0.249], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.4e-23–5.35e-22 | 3/3 | 42: 5.35e-22, 100: 3.95e-22, 2026: 3.4e-23 | -0.297 [-0.358, -0.240], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.0169–0.0169 | single run (0/3 families) | 42: 0.0169 | -0.014 [-0.037, +0.008], p=0.2086 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.18e-11–1.96e-09 | 3/3 | 42: 2.05e-11, 100: 1.18e-11, 2026: 1.96e-09 | -0.130 [-0.175, -0.087], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 5.26e-09–3.32e-08 | 3/3 | 42: 8.54e-09, 100: 3.32e-08, 2026: 5.26e-09 | -0.118 [-0.163, -0.074], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 4.04e-09–4.09e-07 | 3/3 | 42: 6.9e-09, 100: 4.04e-09, 2026: 4.09e-07 | -0.116 [-0.160, -0.073], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.65e-06–5.27e-06 | 3/3 | 42: 1.65e-06, 100: 5.27e-06, 2026: 1.65e-06 | -0.103 [-0.149, -0.060], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.169–0.701 | 0/3 | 42: 0.327, 100: 0.169, 2026: 0.701 | +0.012 [-0.014, +0.039], p=0.3454 |

### sentences_50agree (N = 726)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 0.0125–0.0125 | single run (0/3 families) | 42: 0.0125 | -0.016 [-0.031, +0.000], p=0.0574 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 0.0337–0.0337 | single run (0/3 families) | 42: 0.0337 | -0.012 [-0.062, +0.040], p=0.6714 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.0337–0.0337 | single run (0/3 families) | 42: 0.0337 | -0.013 [-0.063, +0.039], p=0.6428 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 8.51e-13–8.51e-13 | single run (3/3 families) | 42: 8.51e-13 | -0.155 [-0.211, -0.097], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.12e-05–2.12e-05 | single run (3/3 families) | 42: 2.12e-05 | -0.077 [-0.130, -0.022], p=0.0042 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 8.61e-22–8.61e-22 | single run (3/3 families) | 42: 8.61e-22 | -0.271 [-0.331, -0.209], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 1.97e-22–1.97e-22 | single run (3/3 families) | 42: 1.97e-22 | -0.278 [-0.339, -0.214], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 1.28e-47–1.59e-43 | 3/3 | 42: 1.28e-47, 100: 1.59e-43, 2026: 4.83e-44 | -0.382 [-0.432, -0.331], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.65e-44–8.73e-37 | 3/3 | 42: 1.65e-44, 100: 1.47e-37, 2026: 8.73e-37 | -0.355 [-0.407, -0.303], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 0.285–0.285 | single run (0/3 families) | 42: 0.285 | +0.004 [-0.045, +0.054], p=0.8832 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.285–0.285 | single run (0/3 families) | 42: 0.285 | +0.003 [-0.046, +0.052], p=0.9158 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.87e-10–1.87e-10 | single run (3/3 families) | 42: 1.87e-10 | -0.139 [-0.196, -0.082], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.000875–0.000875 | single run (3/3 families) | 42: 0.000875 | -0.061 [-0.113, -0.008], p=0.0216 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 1.12e-18–1.12e-18 | single run (3/3 families) | 42: 1.12e-18 | -0.255 [-0.315, -0.194], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 1.08e-19–1.08e-19 | single run (3/3 families) | 42: 1.08e-19 | -0.262 [-0.323, -0.200], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 1.07e-41–3.4e-36 | 3/3 | 42: 1.07e-41, 100: 3.4e-36, 2026: 2.28e-38 | -0.366 [-0.417, -0.315], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 2.02e-39–2.56e-32 | 3/3 | 42: 2.02e-39, 100: 1.26e-33, 2026: 2.56e-32 | -0.340 [-0.392, -0.288], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 1–1 | single run (0/3 families) | 42: 1 | -0.001 [-0.003, +0.000], p=0.0974 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 1.46e-08–1.46e-08 | single run (3/3 families) | 42: 1.46e-08 | -0.143 [-0.197, -0.090], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 0.00186–0.00186 | single run (3/3 families) | 42: 0.00186 | -0.065 [-0.110, -0.020], p=0.0054 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 7.57e-16–7.57e-16 | single run (3/3 families) | 42: 7.57e-16 | -0.259 [-0.318, -0.199], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 1.93e-17–1.93e-17 | single run (3/3 families) | 42: 1.93e-17 | -0.266 [-0.323, -0.207], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.7e-38–2.67e-30 | 3/3 | 42: 1.7e-38, 100: 2.67e-30, 2026: 6.08e-34 | -0.370 [-0.420, -0.319], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 2.91e-37–1.11e-27 | 3/3 | 42: 2.91e-37, 100: 1.79e-31, 2026: 1.11e-27 | -0.344 [-0.395, -0.292], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.46e-08–1.46e-08 | single run (3/3 families) | 42: 1.46e-08 | -0.142 [-0.196, -0.089], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 0.00186–0.00186 | single run (3/3 families) | 42: 0.00186 | -0.064 [-0.109, -0.018], p=0.0066 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 7.57e-16–7.57e-16 | single run (3/3 families) | 42: 7.57e-16 | -0.258 [-0.317, -0.198], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 1.93e-17–1.93e-17 | single run (3/3 families) | 42: 1.93e-17 | -0.265 [-0.323, -0.205], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 1.7e-38–2.67e-30 | 3/3 | 42: 1.7e-38, 100: 2.67e-30, 2026: 6.08e-34 | -0.369 [-0.419, -0.318], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 2.91e-37–1.11e-27 | 3/3 | 42: 2.91e-37, 100: 1.79e-31, 2026: 1.11e-27 | -0.343 [-0.394, -0.291], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 6.52e-05–6.52e-05 | single run (3/3 families) | 42: 6.52e-05 | +0.078 [+0.035, +0.121], p=0.0006 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 0.00101–0.00101 | single run (3/3 families) | 42: 0.00101 | -0.116 [-0.172, -0.062], p=0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 0.000227–0.000227 | single run (3/3 families) | 42: 0.000227 | -0.123 [-0.179, -0.069], p=0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 2.27e-18–5.91e-12 | 3/3 | 42: 2.27e-18, 100: 5.91e-12, 2026: 3.73e-15 | -0.227 [-0.276, -0.180], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.82e-15–2.11e-11 | 3/3 | 42: 2.82e-15, 100: 1.21e-12, 2026: 2.11e-11 | -0.201 [-0.250, -0.153], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.05e-10–3.05e-10 | single run (3/3 families) | 42: 3.05e-10 | -0.194 [-0.250, -0.139], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 3.59e-11–3.59e-11 | single run (3/3 families) | 42: 3.59e-11 | -0.201 [-0.257, -0.145], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 2e-28–3.07e-22 | 3/3 | 42: 2e-28, 100: 3.07e-22, 2026: 2.24e-25 | -0.305 [-0.355, -0.254], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.07e-29–1.52e-21 | 3/3 | 42: 3.07e-29, 100: 1.14e-24, 2026: 1.52e-21 | -0.279 [-0.327, -0.232], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.359–0.359 | single run (0/3 families) | 42: 0.359 | -0.007 [-0.023, +0.009], p=0.4026 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.29e-09–2.2e-05 | 3/3 | 42: 1.29e-09, 100: 2.2e-05, 2026: 7.34e-08 | -0.110 [-0.148, -0.074], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 8.46e-08–0.00029 | 3/3 | 42: 8.46e-08, 100: 1.39e-05, 2026: 0.00029 | -0.084 [-0.124, -0.046], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.94e-08–0.00015 | 3/3 | 42: 1.94e-08, 100: 0.00015, 2026: 7.88e-07 | -0.104 [-0.143, -0.067], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.14e-06–0.00156 | 3/3 | 42: 1.14e-06, 100: 9.65e-05, 2026: 0.00156 | -0.078 [-0.119, -0.040], p=0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 0.0503–0.771 | 0/3 | 42: 0.389, 100: 0.771, 2026: 0.0503 | +0.026 [-0.005, +0.056], p=0.0964 |

## Paired comparisons: SEntFiN 1.0

### sentences_allagree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 7.29e-12–7.29e-12 | single run (3/3 families) | 42: 7.29e-12 | +0.044 [+0.036, +0.051], p < 0.0002 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | single run (3/3 families) | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 8.22e-05–8.22e-05 | single run (3/3 families) | 42: 8.22e-05 | +0.056 [+0.045, +0.067], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 4.42e-70–4.42e-70 | single run (3/3 families) | 42: 4.42e-70 | -0.099 [-0.112, -0.087], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | single run (3/3 families) | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 3.24e-05–3.24e-05 | single run (3/3 families) | 42: 3.24e-05 | +0.074 [+0.059, +0.089], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.000122–0.000122 | single run (3/3 families) | 42: 0.000122 | +0.075 [+0.061, +0.090], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 9.65e-172–1.97e-114 | 3/3 | 42: 9.76e-120, 100: 9.65e-172, 2026: 1.97e-114 | -0.160 [-0.184, -0.139], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.71e-155–3.17e-77 | 3/3 | 42: 1.71e-155, 100: 3.17e-77, 2026: 7.55e-106 | -0.144 [-0.173, -0.116], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 1–1 | single run (0/3 families) | 42: 1 | +0.015 [+0.004, +0.026], p=0.0060 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.745–0.745 | single run (0/3 families) | 42: 0.745 | +0.013 [+0.002, +0.023], p=0.0242 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 3.66e-121–3.66e-121 | single run (3/3 families) | 42: 3.66e-121 | -0.143 [-0.155, -0.131], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 7.79e-73–7.79e-73 | single run (3/3 families) | 42: 7.79e-73 | -0.104 [-0.114, -0.093], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 0.515–0.515 | single run (0/3 families) | 42: 0.515 | +0.030 [+0.015, +0.044], p=0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 0.763–0.763 | single run (0/3 families) | 42: 0.763 | +0.032 [+0.017, +0.046], p=0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 5.49e-227–1.88e-168 | 3/3 | 42: 1.89e-175, 100: 5.49e-227, 2026: 1.88e-168 | -0.204 [-0.227, -0.183], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 3.3e-208–4.39e-124 | 3/3 | 42: 3.3e-208, 100: 4.39e-124, 2026: 3.83e-158 | -0.188 [-0.217, -0.160], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.00143–0.00143 | single run (3/3 families) | 42: 0.00143 | -0.003 [-0.004, -0.001], p=0.0004 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 2.58e-125–2.58e-125 | single run (3/3 families) | 42: 2.58e-125 | -0.158 [-0.170, -0.146], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | single run (3/3 families) | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.508–0.508 | single run (0/3 families) | 42: 0.508 | +0.015 [-0.000, +0.030], p=0.0534 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.762–0.762 | single run (0/3 families) | 42: 0.762 | +0.016 [+0.002, +0.031], p=0.0296 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 4.78e-242–3.57e-193 | 3/3 | 42: 5.51e-197, 100: 4.78e-242, 2026: 3.57e-193 | -0.219 [-0.243, -0.198], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 8.23e-228–8e-144 | 3/3 | 42: 8.23e-228, 100: 8e-144, 2026: 2.42e-182 | -0.203 [-0.231, -0.175], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.01e-121–1.01e-121 | single run (3/3 families) | 42: 1.01e-121 | -0.155 [-0.168, -0.143], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 3.44e-93–3.44e-93 | single run (3/3 families) | 42: 3.44e-93 | -0.116 [-0.126, -0.107], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 0.331–0.331 | single run (0/3 families) | 42: 0.331 | +0.018 [+0.003, +0.032], p=0.0198 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 0.534–0.534 | single run (0/3 families) | 42: 0.534 | +0.019 [+0.004, +0.034], p=0.0110 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 1.23e-236–2.41e-188 | 3/3 | 42: 1.7e-191, 100: 1.23e-236, 2026: 2.41e-188 | -0.216 [-0.240, -0.196], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 1.68e-222–3.02e-139 | 3/3 | 42: 1.68e-222, 100: 3.02e-139, 2026: 5.4e-177 | -0.200 [-0.228, -0.172], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 7.85e-22–7.85e-22 | single run (3/3 families) | 42: 7.85e-22 | +0.039 [+0.030, +0.048], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.74e-158–3.74e-158 | single run (3/3 families) | 42: 3.74e-158 | +0.173 [+0.161, +0.185], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 3.47e-152–3.47e-152 | single run (3/3 families) | 42: 3.47e-152 | +0.175 [+0.163, +0.186], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 8.71e-42–1.81e-13 | 3/3 | 42: 3.62e-15, 100: 8.71e-42, 2026: 1.81e-13 | -0.061 [-0.084, -0.041], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.82e-33–0.0155 | 2/3 | 42: 1.82e-33, 100: 0.0155, 2026: 8.53e-11 | -0.045 [-0.073, -0.018], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 2.86e-55–2.86e-55 | single run (3/3 families) | 42: 2.86e-55 | +0.134 [+0.120, +0.147], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 6.18e-53–6.18e-53 | single run (3/3 families) | 42: 6.18e-53 | +0.135 [+0.121, +0.149], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 1.01e-88–9.01e-49 | 3/3 | 42: 1.09e-52, 100: 1.01e-88, 2026: 9.01e-49 | -0.100 [-0.123, -0.080], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 4.79e-78–1.89e-23 | 3/3 | 42: 4.79e-78, 100: 1.89e-23, 2026: 3.25e-43 | -0.084 [-0.113, -0.057], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.284–0.284 | single run (0/3 families) | 42: 0.284 | +0.002 [-0.003, +0.006], p=0.4728 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 2.56e-265–1.32e-214 | 3/3 | 42: 1.32e-214, 100: 2.56e-265, 2026: 2.39e-215 | -0.234 [-0.257, -0.214], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 1.72e-260–6.56e-184 | 3/3 | 42: 1.72e-260, 100: 6.56e-184, 2026: 1.22e-225 | -0.218 [-0.246, -0.190], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.86e-257–2.49e-210 | 3/3 | 42: 2.49e-210, 100: 1.86e-257, 2026: 5.37e-211 | -0.235 [-0.259, -0.215], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.22e-256–2.91e-183 | 3/3 | 42: 1.22e-256, 100: 2.91e-183, 2026: 2.08e-224 | -0.219 [-0.247, -0.192], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 5.96e-36–0.388 | 2/3 | 42: 4.78e-06, 100: 5.96e-36, 2026: 0.388 | +0.016 [-0.016, +0.049], p=0.3484 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | single run (3/3 families) | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon, val-tuned thresholds) | 1.55e-244–1.55e-244 | single run (3/3 families) | 42: 1.55e-244 | +0.246 [+0.233, +0.259], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | single run (3/3 families) | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 4.72e-247–4.72e-247 | single run (3/3 families) | 42: 4.72e-247 | +0.259 [+0.246, +0.272], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 1.63e-56–1.63e-56 | single run (3/3 families) | 42: 1.63e-56 | +0.103 [+0.092, +0.114], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | single run (3/3 families) | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 1.5e-279–1.5e-279 | single run (3/3 families) | 42: 1.5e-279 | +0.276 [+0.264, +0.288], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 4.93e-272–4.93e-272 | single run (3/3 families) | 42: 4.93e-272 | +0.278 [+0.265, +0.290], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 8.81e-29–1.99e-05 | 3/3 | 42: 3.2e-26, 100: 1.99e-05, 2026: 8.81e-29 | +0.042 [+0.020, +0.061], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 6.49e-51–6.12e-09 | 3/3 | 42: 6.12e-09, 100: 6.49e-51, 2026: 1.47e-29 | +0.058 [+0.031, +0.085], p < 0.0002 |

### sentences_75agree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 0.0503–0.0503 | single run (0/3 families) | 42: 0.0503 | -0.003 [-0.007, +0.002], p=0.2920 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | single run (3/3 families) | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 8.22e-05–8.22e-05 | single run (3/3 families) | 42: 8.22e-05 | +0.056 [+0.045, +0.067], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 2.25e-72–2.25e-72 | single run (3/3 families) | 42: 2.25e-72 | -0.100 [-0.112, -0.088], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | single run (3/3 families) | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 0.0885–0.0885 | single run (0/3 families) | 42: 0.0885 | +0.050 [+0.035, +0.065], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.0995–0.0995 | single run (0/3 families) | 42: 0.0995 | +0.053 [+0.038, +0.068], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 1.98e-130–8.07e-96 | 3/3 | 42: 6.97e-117, 100: 1.98e-130, 2026: 8.07e-96 | -0.151 [-0.167, -0.134], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 2.01e-157–4.27e-48 | 3/3 | 42: 7.96e-92, 100: 2.01e-157, 2026: 4.27e-48 | -0.128 [-0.173, -0.083], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 1.49e-08–1.49e-08 | single run (3/3 families) | 42: 1.49e-08 | +0.061 [+0.050, +0.072], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 9.84e-08–9.84e-08 | single run (3/3 families) | 42: 9.84e-08 | +0.059 [+0.048, +0.069], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.25e-72–1.25e-72 | single run (3/3 families) | 42: 1.25e-72 | -0.098 [-0.110, -0.085], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 5.82e-33–5.82e-33 | single run (3/3 families) | 42: 5.82e-33 | -0.058 [-0.068, -0.047], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 0.00915–0.00915 | single run (0/3 families) | 42: 0.00915 | +0.053 [+0.038, +0.067], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 0.0105–0.0105 | single run (0/3 families) | 42: 0.0105 | +0.055 [+0.041, +0.070], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 6.68e-133–1.37e-97 | 3/3 | 42: 7.47e-118, 100: 6.68e-133, 2026: 1.37e-97 | -0.148 [-0.165, -0.132], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 2.11e-159–1.66e-47 | 3/3 | 42: 9.6e-92, 100: 2.11e-159, 2026: 1.66e-47 | -0.125 [-0.170, -0.080], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.00143–0.00143 | single run (3/3 families) | 42: 0.00143 | -0.003 [-0.004, -0.001], p=0.0004 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 9.62e-129–9.62e-129 | single run (3/3 families) | 42: 9.62e-129 | -0.159 [-0.171, -0.147], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | single run (3/3 families) | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.0344–0.0344 | single run (0/3 families) | 42: 0.0344 | -0.009 [-0.023, +0.006], p=0.2366 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.0255–0.0255 | single run (0/3 families) | 42: 0.0255 | -0.006 [-0.020, +0.008], p=0.4176 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 5.48e-214–3.51e-168 | 3/3 | 42: 2.5e-194, 100: 5.48e-214, 2026: 3.51e-168 | -0.209 [-0.225, -0.193], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 3.19e-249–1.25e-104 | 3/3 | 42: 1.56e-163, 100: 3.19e-249, 2026: 1.25e-104 | -0.187 [-0.231, -0.141], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 4.31e-125–4.31e-125 | single run (3/3 families) | 42: 4.31e-125 | -0.156 [-0.169, -0.144], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 3.44e-93–3.44e-93 | single run (3/3 families) | 42: 3.44e-93 | -0.116 [-0.126, -0.107], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 0.0716–0.0716 | single run (0/3 families) | 42: 0.0716 | -0.006 [-0.021, +0.009], p=0.4162 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 0.0558–0.0558 | single run (0/3 families) | 42: 0.0558 | -0.003 [-0.018, +0.011], p=0.6540 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 1.42e-208–3.07e-163 | 3/3 | 42: 3.99e-189, 100: 1.42e-208, 2026: 3.07e-163 | -0.207 [-0.223, -0.190], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 1.21e-242–1.51e-100 | 3/3 | 42: 1.74e-158, 100: 1.21e-242, 2026: 1.51e-100 | -0.184 [-0.229, -0.139], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 1.67e-23–1.67e-23 | single run (3/3 families) | 42: 1.67e-23 | +0.040 [+0.031, +0.049], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.32e-127–6.32e-127 | single run (3/3 families) | 42: 6.32e-127 | +0.150 [+0.139, +0.161], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 4.96e-127–4.96e-127 | single run (3/3 families) | 42: 4.96e-127 | +0.153 [+0.142, +0.164], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 4.64e-21–6.66e-08 | 3/3 | 42: 1.3e-14, 100: 4.64e-21, 2026: 6.66e-08 | -0.050 [-0.066, -0.035], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 6.2e-32–0.00211 | 3/3 | 42: 5.81e-05, 100: 6.2e-32, 2026: 0.00211 | -0.028 [-0.072, +0.016], p=0.2432 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 3.24e-38–3.24e-38 | single run (3/3 families) | 42: 3.24e-38 | +0.110 [+0.097, +0.124], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 1.47e-38–1.47e-38 | single run (3/3 families) | 42: 1.47e-38 | +0.113 [+0.099, +0.127], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 1.38e-62–4.07e-38 | 3/3 | 42: 3.29e-52, 100: 1.38e-62, 2026: 4.07e-38 | -0.091 [-0.107, -0.074], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 1.6e-82–3.19e-07 | 3/3 | 42: 6.65e-32, 100: 1.6e-82, 2026: 3.19e-07 | -0.068 [-0.113, -0.023], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.864–0.864 | single run (0/3 families) | 42: 0.864 | +0.003 [-0.001, +0.007], p=0.1878 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 3.33e-209–1.79e-165 | 3/3 | 42: 6.87e-190, 100: 3.33e-209, 2026: 1.79e-165 | -0.201 [-0.217, -0.185], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 7.37e-235–1.34e-92 | 3/3 | 42: 1.9e-155, 100: 7.37e-235, 2026: 1.34e-92 | -0.178 [-0.222, -0.133], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 3.54e-211–2.54e-167 | 3/3 | 42: 1.16e-192, 100: 3.54e-211, 2026: 2.54e-167 | -0.203 [-0.219, -0.187], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 1.48e-238–9.51e-96 | 3/3 | 42: 6.24e-160, 100: 1.48e-238, 2026: 9.51e-96 | -0.181 [-0.225, -0.136], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 4.59e-22–0.00493 | 2/3 | 42: 7.63e-06, 100: 0.00493, 2026: 4.59e-22 | +0.023 [-0.023, +0.068], p=0.3502 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | single run (3/3 families) | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon, val-tuned thresholds) | 2.15e-192–2.15e-192 | single run (3/3 families) | 42: 2.15e-192 | +0.200 [+0.187, +0.213], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | single run (3/3 families) | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 4.72e-247–4.72e-247 | single run (3/3 families) | 42: 4.72e-247 | +0.259 [+0.246, +0.272], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 1.9e-54–1.9e-54 | single run (3/3 families) | 42: 1.9e-54 | +0.102 [+0.091, +0.114], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | single run (3/3 families) | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 1.94e-248–1.94e-248 | single run (3/3 families) | 42: 1.94e-248 | +0.253 [+0.240, +0.265], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 8.37e-247–8.37e-247 | single run (3/3 families) | 42: 8.37e-247 | +0.255 [+0.243, +0.268], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 2.34e-37–6.09e-18 | 3/3 | 42: 5.81e-25, 100: 6.09e-18, 2026: 2.34e-37 | +0.052 [+0.038, +0.066], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 3.57e-78–3.07e-08 | 3/3 | 42: 6.8e-38, 100: 3.07e-08, 2026: 3.57e-78 | +0.075 [+0.030, +0.120], p < 0.0002 |

### sentences_50agree (N = 9460)

| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | seed+sentence bootstrap ΔMacro-F1 [95% CI] |
|---|---|---|---|---|
| VADER (official lexicon) vs VADER (official lexicon, val-tuned thresholds) | 0.174–0.174 | single run (0/3 families) | 42: 0.174 | -0.000 [-0.005, +0.004], p=0.9006 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary) | 2.17e-05–2.17e-05 | single run (3/3 families) | 42: 2.17e-05 | +0.059 [+0.047, +0.070], p < 0.0002 |
| VADER (official lexicon) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 8.22e-05–8.22e-05 | single run (3/3 families) | 42: 8.22e-05 | +0.056 [+0.045, +0.067], p < 0.0002 |
| VADER (official lexicon) vs VADER-style (finance-adjusted, custom) | 3.39e-69–3.39e-69 | single run (3/3 families) | 42: 3.39e-69 | -0.099 [-0.111, -0.087], p < 0.0002 |
| VADER (official lexicon) vs LM-style (finance-adjusted, custom) | 2.98e-33–2.98e-33 | single run (3/3 families) | 42: 2.98e-33 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Logistic Regression | 0.000187–0.000187 | single run (3/3 families) | 42: 0.000187 | +0.050 [+0.036, +0.064], p < 0.0002 |
| VADER (official lexicon) vs TF-IDF + Linear SVM | 0.000199–0.000199 | single run (3/3 families) | 42: 0.000199 | +0.056 [+0.042, +0.070], p < 0.0002 |
| VADER (official lexicon) vs bert-base-uncased | 4.58e-140–7.04e-92 | 3/3 | 42: 1.26e-126, 100: 4.58e-140, 2026: 7.04e-92 | -0.151 [-0.171, -0.130], p < 0.0002 |
| VADER (official lexicon) vs FinBERT-MLM (finbert-pretrain) | 1.9e-182–5.64e-150 | 3/3 | 42: 1.9e-182, 100: 5.43e-168, 2026: 5.64e-150 | -0.179 [-0.194, -0.164], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary) | 9.4e-08–9.4e-08 | single run (3/3 families) | 42: 9.4e-08 | +0.059 [+0.048, +0.070], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 5.61e-07–5.61e-07 | single run (3/3 families) | 42: 5.61e-07 | +0.056 [+0.045, +0.067], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.41e-71–1.41e-71 | single run (3/3 families) | 42: 1.41e-71 | -0.099 [-0.111, -0.087], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 1.14e-34–1.14e-34 | single run (3/3 families) | 42: 1.14e-34 | -0.060 [-0.071, -0.049], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Logistic Regression | 6.39e-06–6.39e-06 | single run (3/3 families) | 42: 6.39e-06 | +0.050 [+0.036, +0.064], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs TF-IDF + Linear SVM | 6.35e-06–6.35e-06 | single run (3/3 families) | 42: 6.35e-06 | +0.056 [+0.042, +0.070], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs bert-base-uncased | 5.37e-141–6.31e-92 | 3/3 | 42: 4.94e-130, 100: 5.37e-141, 2026: 6.31e-92 | -0.151 [-0.171, -0.130], p < 0.0002 |
| VADER (official lexicon, val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 3.8e-183–2.03e-153 | 3/3 | 42: 3.8e-183, 100: 8.01e-174, 2026: 2.03e-153 | -0.179 [-0.193, -0.163], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.00143–0.00143 | single run (3/3 families) | 42: 0.00143 | -0.003 [-0.004, -0.001], p=0.0004 |
| Loughran-McDonald (official dictionary) vs VADER-style (finance-adjusted, custom) | 3.37e-124–3.37e-124 | single run (3/3 families) | 42: 3.37e-124 | -0.158 [-0.170, -0.146], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs LM-style (finance-adjusted, custom) | 3.98e-96–3.98e-96 | single run (3/3 families) | 42: 3.98e-96 | -0.119 [-0.129, -0.109], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Logistic Regression | 0.944–0.944 | single run (0/3 families) | 42: 0.944 | -0.009 [-0.023, +0.005], p=0.2142 |
| Loughran-McDonald (official dictionary) vs TF-IDF + Linear SVM | 0.971–0.971 | single run (0/3 families) | 42: 0.971 | -0.003 [-0.017, +0.011], p=0.6722 |
| Loughran-McDonald (official dictionary) vs bert-base-uncased | 1.11e-201–2.81e-142 | 3/3 | 42: 1.11e-201, 100: 8.48e-198, 2026: 2.81e-142 | -0.210 [-0.230, -0.189], p < 0.0002 |
| Loughran-McDonald (official dictionary) vs FinBERT-MLM (finbert-pretrain) | 7.98e-264–5.34e-227 | 3/3 | 42: 2.11e-255, 100: 7.98e-264, 2026: 5.34e-227 | -0.238 [-0.253, -0.223], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs VADER-style (finance-adjusted, custom) | 1.27e-120–1.27e-120 | single run (3/3 families) | 42: 1.27e-120 | -0.155 [-0.168, -0.143], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs LM-style (finance-adjusted, custom) | 3.44e-93–3.44e-93 | single run (3/3 families) | 42: 3.44e-93 | -0.116 [-0.126, -0.107], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Logistic Regression | 0.701–0.701 | single run (0/3 families) | 42: 0.701 | -0.006 [-0.020, +0.008], p=0.3880 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs TF-IDF + Linear SVM | 0.722–0.722 | single run (0/3 families) | 42: 0.722 | -0.000 [-0.014, +0.014], p=0.9732 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs bert-base-uncased | 3.15e-196–1.66e-138 | 3/3 | 42: 3.15e-196, 100: 2.48e-193, 2026: 1.66e-138 | -0.207 [-0.227, -0.186], p < 0.0002 |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) vs FinBERT-MLM (finbert-pretrain) | 1.54e-256–9.5e-221 | 3/3 | 42: 2.67e-249, 100: 1.54e-256, 2026: 9.5e-221 | -0.235 [-0.250, -0.220], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs LM-style (finance-adjusted, custom) | 3.19e-21–3.19e-21 | single run (3/3 families) | 42: 3.19e-21 | +0.039 [+0.030, +0.048], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 2.89e-136–2.89e-136 | single run (3/3 families) | 42: 2.89e-136 | +0.149 [+0.137, +0.160], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 8.77e-138–8.77e-138 | single run (3/3 families) | 42: 8.77e-138 | +0.155 [+0.143, +0.167], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs bert-base-uncased | 2.19e-22–7.87e-06 | 3/3 | 42: 1.14e-17, 100: 2.19e-22, 2026: 7.87e-06 | -0.052 [-0.072, -0.031], p < 0.0002 |
| VADER-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 2.73e-41–5.71e-25 | 3/3 | 42: 2.73e-41, 100: 9.72e-35, 2026: 5.71e-25 | -0.080 [-0.094, -0.066], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Logistic Regression | 6.32e-53–6.32e-53 | single run (3/3 families) | 42: 6.32e-53 | +0.110 [+0.096, +0.123], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs TF-IDF + Linear SVM | 2.74e-54–2.74e-54 | single run (3/3 families) | 42: 2.74e-54 | +0.116 [+0.102, +0.129], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs bert-base-uncased | 9.49e-61–2.46e-29 | 3/3 | 42: 3.72e-54, 100: 9.49e-61, 2026: 2.46e-29 | -0.091 [-0.111, -0.070], p < 0.0002 |
| LM-style (finance-adjusted, custom) vs FinBERT-MLM (finbert-pretrain) | 3.63e-96–1.03e-69 | 3/3 | 42: 3.63e-96, 100: 1.09e-88, 2026: 1.03e-69 | -0.119 [-0.133, -0.104], p < 0.0002 |
| TF-IDF + Logistic Regression vs TF-IDF + Linear SVM | 0.959–0.959 | single run (0/3 families) | 42: 0.959 | +0.006 [+0.001, +0.010], p=0.0064 |
| TF-IDF + Logistic Regression vs bert-base-uncased | 1.25e-207–2.59e-154 | 3/3 | 42: 8.54e-204, 100: 1.25e-207, 2026: 2.59e-154 | -0.201 [-0.221, -0.180], p < 0.0002 |
| TF-IDF + Logistic Regression vs FinBERT-MLM (finbert-pretrain) | 2.87e-262–1.63e-232 | 3/3 | 42: 2.87e-262, 100: 2.35e-252, 2026: 1.63e-232 | -0.229 [-0.244, -0.214], p < 0.0002 |
| TF-IDF + Linear SVM vs bert-base-uncased | 1.24e-206–3.59e-155 | 3/3 | 42: 1.27e-205, 100: 1.24e-206, 2026: 3.59e-155 | -0.207 [-0.227, -0.186], p < 0.0002 |
| TF-IDF + Linear SVM vs FinBERT-MLM (finbert-pretrain) | 4.63e-263–9.48e-234 | 3/3 | 42: 4.63e-263, 100: 7.12e-255, 2026: 9.48e-234 | -0.235 [-0.250, -0.220], p < 0.0002 |
| bert-base-uncased vs FinBERT-MLM (finbert-pretrain) | 7.49e-10–0.00561 | 3/3 | 42: 7.49e-10, 100: 0.00561, 2026: 8.63e-10 | -0.028 [-0.049, -0.007], p=0.0042 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon) | 6.84e-198–6.84e-198 | single run (3/3 families) | 42: 6.84e-198 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER (official lexicon, val-tuned thresholds) | 3.06e-195–3.06e-195 | single run (3/3 families) | 42: 3.06e-195 | +0.202 [+0.190, +0.215], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary) | 1.06e-252–1.06e-252 | single run (3/3 families) | 42: 1.06e-252 | +0.261 [+0.249, +0.274], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 4.72e-247–4.72e-247 | single run (3/3 families) | 42: 4.72e-247 | +0.259 [+0.246, +0.272], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs VADER-style (finance-adjusted, custom) | 2.67e-57–2.67e-57 | single run (3/3 families) | 42: 2.67e-57 | +0.103 [+0.092, +0.115], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs LM-style (finance-adjusted, custom) | 2.17e-105–2.17e-105 | single run (3/3 families) | 42: 2.17e-105 | +0.142 [+0.131, +0.154], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Logistic Regression | 4.19e-275–4.19e-275 | single run (3/3 families) | 42: 4.19e-275 | +0.252 [+0.240, +0.265], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs TF-IDF + Linear SVM | 2.84e-273–2.84e-273 | single run (3/3 families) | 42: 2.84e-273 | +0.258 [+0.246, +0.271], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs bert-base-uncased | 8.09e-42–5.71e-16 | 3/3 | 42: 4.59e-22, 100: 5.71e-16, 2026: 8.09e-42 | +0.051 [+0.034, +0.071], p < 0.0002 |
| ProsusAI/finbert (zero-shot) vs FinBERT-MLM (finbert-pretrain) | 1.08e-12–0.00259 | 3/3 | 42: 0.00259, 100: 1.81e-06, 2026: 1.08e-12 | +0.024 [+0.011, +0.037], p=0.0004 |

## Figures

- `step2_macro_f1_by_config.png`
- `step2_external_generalization.png`
- `step2_attribution.png`
- `step2_confusion_matrices_allagree.png`
