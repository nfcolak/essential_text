# Step 1 Baseline Evaluation Report

- **Execution Mode:** `full`
- **Dataset:** `financial_phrasebank`
- **Random Seed:** `42`
- **Bootstrap Samples:** `10000`
- **Generated At (UTC):** `2026-10-19T10:45:40.173402+00:00`

## 1. Dataset Partitioning & Leakage Audit

| Partition | Sample Count |
|---|---|
| Training (`train`) | 1,585 |
| Validation (`val`) | 340 |
| Test (`test`) | 339 |
| External Holdout (`SEntFiN 1.0` deduplicated) | 9,460 |

### Leakage Verification Results

| Verification Check | Status | Value | Constraint |
|---|---|---|---|
| ID Overlap Count | **PASSED** | `0` | Exactly 0 |
| SHA-256 Hash Overlap Count | **PASSED** | `0` | Exactly 0 |
| Near-Duplicate Cluster Overlap | **PASSED** | `0` | Exactly 0 |
| Max Cross-Split Token Jaccard | **PASSED** | `0.7857` | < 0.8000 |
| 5-Fold CV Partition Leakage | **PASSED** | `0` | Disjoint Folds |

## 2. Held-Out Test Set Performance (Tier 1 & Tier 2 Baselines)

| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ECE | RQ1 Neu→Neg Error Rate |
|---|---|---|---|---|---|---|---|
| **VADER** | 0.8201 | **0.7534** | 0.6667 | 0.8993 | 0.6941 | 0.1410 | **0.0192** |
| **Loughran-McDonald** | 0.7345 | **0.6355** | 0.4865 | 0.8251 | 0.5949 | 0.2635 | **0.0337** |
| **TF-IDF + Logistic Regression** | 0.8938 | **0.8567** | 0.8172 | 0.9431 | 0.8098 | 0.0798 | **0.0096** |
| **TF-IDF + Linear SVM** | 0.9056 | **0.8791** | 0.8605 | 0.9434 | 0.8333 | 0.0366 | **0.0048** |

## 3. 5-Fold Cross-Validation Performance (Training Partition)

| Model | Mean Accuracy (±Std) | Mean Macro-F1 (±Std) | Mean Neu→Neg Rate (±Std) |
|---|---|---|---|
| **VADER** | 0.8000 ± 0.0242 | **0.7165 ± 0.0371** | 0.0082 ± 0.0025 |
| **Loughran-McDonald** | 0.7319 ± 0.0187 | **0.6181 ± 0.0267** | 0.0215 ± 0.0104 |
| **TF-IDF + Logistic Regression** | 0.8593 ± 0.0240 | **0.8003 ± 0.0419** | 0.0123 ± 0.0106 |
| **TF-IDF + Linear SVM** | 0.8732 ± 0.0131 | **0.8181 ± 0.0276** | 0.0041 ± 0.0039 |

## 4. Deduplicated External Holdout Performance (SEntFiN 1.0)

| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ECE | Neu→Neg Error Rate |
|---|---|---|---|---|---|---|---|
| **VADER** | 0.6344 | **0.6242** | 0.5757 | 0.6589 | 0.6380 | 0.3335 | 0.0364 |
| **Loughran-McDonald** | 0.5921 | **0.5850** | 0.5917 | 0.6205 | 0.5429 | 0.4049 | 0.0637 |
| **TF-IDF + Logistic Regression** | 0.4921 | **0.4440** | 0.3304 | 0.5781 | 0.4235 | 0.2117 | 0.0208 |
| **TF-IDF + Linear SVM** | 0.4895 | **0.4353** | 0.3235 | 0.5796 | 0.4029 | 0.4093 | 0.0123 |

## 5. RQ1 Deep Dive: Neutral-to-Negative Misclassification Analysis

RQ1 investigates domain polarity mismatch: general sentiment lexicons (such as VADER) frequently misclassify neutral descriptive accounting terminology (*cost*, *liability*, *depreciation*, *debt*, *obligation*) as negative, creating false-negative errors. Below are the observed `neutral_to_negative_rate` metrics across models:

- **VADER:** 4 / 208 neutral sentences falsely predicted as negative (1.92%).
- **Loughran-McDonald:** 7 / 208 neutral sentences falsely predicted as negative (3.37%).
- **TF-IDF + Logistic Regression:** 2 / 208 neutral sentences falsely predicted as negative (0.96%).
- **TF-IDF + Linear SVM:** 1 / 208 neutral sentences falsely predicted as negative (0.48%).

## 6. Statistical Significance Comparisons (Held-Out Test Set)

| Comparison (Model A vs. Model B) | McNemar χ² | McNemar p-value | ΔMacro-F1 (95% CI) | Bootstrap p-value | Significant (p < 0.05) |
|---|---|---|---|---|---|
| **VADER vs Loughran-McDonald** | 14.792 | 0.0001 | +0.1178 [+0.0609, +0.1776] | 0.0000 | **Yes (p < 0.05)** |
| **VADER vs TF-IDF + Logistic Regression** | 7.890 | 0.0046 | -0.1034 [-0.1772, -0.0285] | 0.0070 | **Yes (p < 0.05)** |
| **VADER vs TF-IDF + Linear SVM** | 10.453 | 0.0011 | -0.1257 [-0.1999, -0.0510] | 0.0007 | **Yes (p < 0.05)** |
| **Loughran-McDonald vs TF-IDF + Logistic Regression** | 25.536 | 0.0000 | -0.2212 [-0.3098, -0.1345] | 0.0000 | **Yes (p < 0.05)** |
| **Loughran-McDonald vs TF-IDF + Linear SVM** | 29.536 | 0.0000 | -0.2436 [-0.3310, -0.1573] | 0.0000 | **Yes (p < 0.05)** |
| **TF-IDF + Logistic Regression vs TF-IDF + Linear SVM** | 0.900 | 0.3438 | -0.0223 [-0.0512, +0.0038] | 0.1064 | **No** |

## 7. Transition to Step 2 (Tier 3 Transformer Fine-Tuning)

With Step 1 complete, the locked SHA-256 split manifests in `data/splits/` will serve as the exact, identical train/val/test splits for fine-tuning `bert-base-uncased` and `yiyanghkust/finbert-pretrain` in Step 2. This guarantees zero test leakage and enables rigorous attribution of domain adaptation benefits.
