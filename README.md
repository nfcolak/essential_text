# Evaluating Domain Adaptation in Financial Sentiment Analysis

**FinBERT vs. Traditional NLP Baselines**  
*Authors: Qiyao Zheng, Benjamin Donick, Necati Furkan Colak*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nfcolak/essential_text/blob/main/notebooks/step1_data_and_baselines_colab.ipynb)

## Overview

This repository contains the reproducible evaluation pipeline, leakage-free data splits, dataset provenance audit, and experiment code for our study on **domain adaptation in 3-class financial sentiment classification** (*positive*, *neutral*, *negative*).

### Research Questions
1. **RQ1 (Domain Polarity Mismatch):** To what extent do general-purpose sentiment methods misclassify financial language, particularly by falsely assigning *negative* polarity to *neutral* descriptive accounting statements (e.g., *cost*, *liability*, *debt*, *depreciation*)?
2. **RQ2 (Controlled Architecture vs. Domain Adaptation):** Under a strictly leakage-free evaluation protocol, how much classification improvement stems from domain-adaptive pretraining (`yiyanghkust/finbert-pretrain`) versus general-domain `bert-base-uncased` and traditional machine-learning baselines?
3. **RQ3 (Linguistic Phenomenon Attribution):** Which linguistic phenomena—financial terminology, negation, hedging, numerical comparisons, or context-dependent expressions—drive model divergence?

---

## Step 1 Experimental Results (Leakage-Free Splits & Tier 1/2 Baselines)

Full machine-readable artifacts are available in [`results/step1/`](results/step1/):
- **Detailed Report:** [`results/step1/STEP1_REPORT.md`](results/step1/STEP1_REPORT.md)
- **Summary Metrics & Calibration:** [`results/step1/summary_metrics.json`](results/step1/summary_metrics.json)
- **Statistical Significance Tests:** [`results/step1/statistical_comparisons.json`](results/step1/statistical_comparisons.json)
- **Dataset Provenance & Audit Verdict:** [`data/dataset_audit_verdict.json`](data/dataset_audit_verdict.json)

### 1. Dataset Partitioning & Leakage Audit (`seed=42`)

All datasets are downloaded from pinned upstream commits with SHA-256 verification (`FinancialPhraseBank-v1.0.zip` @ `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964` and `SEntFiN.csv` @ `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`). Near-duplicate sentences (token Jaccard $\ge 0.80$) are grouped via connected-component clustering and locked into identical partitions prior to splitting.

| Partition / Subset | Total Rows | Train (70%) | Val (15%) | Test (15%) | Cross-Split ID/Hash/Cluster Overlap | Max Cross-Split Token Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| **Financial PhraseBank (`sentences_allagree`)** | 2,264 | 1,585 | 340 | 339 | `0` (**PASSED**) | `0.7857 < 0.80` (**PASSED**) |
| **Financial PhraseBank (`sentences_75agree`)** | 3,453 | 2,417 | 518 | 518 | `0` (**PASSED**) | `0.7895 < 0.80` (**PASSED**) |
| **Financial PhraseBank (`sentences_50agree`)** | 4,846 | 3,392 | 728 | 726 | `0` (**PASSED**) | `0.7917 < 0.80` (**PASSED**) |
| **External Holdout (`SEntFiN 1.0` unambiguous unique)** | 9,460 | — | — | 9,460 | `0` (**PASSED**) | `< 0.80` vs. PhraseBank (**PASSED**) |

### 2. Held-Out Test Set Performance (`Financial PhraseBank sentences_allagree`, $N = 339$)

| Model | Tier | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | ECE | RQ1 Neu→Neg Error Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **VADER** | Tier 1 (Lexicon) | 0.8201 | **0.7534** | 0.6667 | 0.8993 | 0.6941 | 0.1410 | 1.92% (4 / 208) |
| **Loughran–McDonald** | Tier 1 (Lexicon) | 0.7345 | **0.6355** | 0.4865 | 0.8251 | 0.5949 | 0.2635 | 3.37% (7 / 208) |
| **TF-IDF + Logistic Regression** | Tier 2 (Linear ML) | 0.8938 | **0.8567** | 0.8172 | 0.9431 | 0.8098 | 0.0798 | 0.96% (2 / 208) |
| **TF-IDF + Linear SVM** | Tier 2 (Linear ML) | **0.9056** | **0.8791** | **0.8605** | **0.9434** | **0.8333** | **0.0366** | **0.48% (1 / 208)** |

### 3. 5-Fold Cross-Validation Performance (Training Partition, $N = 1,585$)

| Model | Mean Accuracy ($\pm$ Std) | Mean Macro-F1 ($\pm$ Std) | Mean Neu→Neg Rate ($\pm$ Std) |
|---|---:|---:|---:|
| **VADER** | $0.8000 \pm 0.0242$ | **$0.7165 \pm 0.0371$** | $0.0082 \pm 0.0025$ |
| **Loughran–McDonald** | $0.7319 \pm 0.0187$ | **$0.6181 \pm 0.0267$** | $0.0215 \pm 0.0104$ |
| **TF-IDF + Logistic Regression** | $0.8593 \pm 0.0240$ | **$0.8003 \pm 0.0419$** | $0.0123 \pm 0.0106$ |
| **TF-IDF + Linear SVM** | **$0.8732 \pm 0.0131$** | **$0.8181 \pm 0.0276$** | **$0.0041 \pm 0.0039$** |

### 4. Deduplicated External Holdout Generalization (`SEntFiN 1.0`, $N = 9,460$)

| Model | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | ECE | Neu→Neg Error Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VADER** | **0.6344** | **0.6242** | 0.5757 | **0.6589** | **0.6380** | 0.3335 | 3.64% |
| **Loughran–McDonald** | 0.5921 | 0.5850 | **0.5917** | 0.6205 | 0.5429 | 0.4049 | 6.37% |
| **TF-IDF + Logistic Regression** | 0.4921 | 0.4440 | 0.3304 | 0.5781 | 0.4235 | **0.2117** | 2.08% |
| **TF-IDF + Linear SVM** | 0.4895 | 0.4353 | 0.3235 | 0.5796 | 0.4029 | 0.4093 | **1.23%** |

> **Key Finding on Out-of-Distribution Generalization:** While TF-IDF + Linear SVM achieves strong in-domain performance on Financial PhraseBank (`Macro-F1 = 0.8791`), its performance drops sharply on the external `SEntFiN 1.0` entity-headline holdout (`Macro-F1 = 0.4353`) due to n-gram vocabulary shift between corporate earnings sentences and financial news headlines. Rule-based lexicons generalize more gracefully (`VADER Macro-F1 = 0.6242`), motivating our Step 2 contextual transformer investigation (`bert-base-uncased` vs. `yiyanghkust/finbert-pretrain`).

### 5. Statistical Significance Comparisons (Held-Out Test Set, $B = 10,000$ Paired Bootstrap)

| Comparison (Model A vs. Model B) | McNemar $\chi^2$ | McNemar $p$-value | $\Delta$Macro-F1 (95% CI) | Bootstrap $p$-value | Significant ($p < 0.05$) |
|---|---:|---:|---:|---:|---|
| **VADER vs. Loughran–McDonald** | 14.792 | 0.0001 | $+0.1178\ [+0.0609, +0.1776]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **VADER vs. TF-IDF + Logistic Regression** | 7.890 | 0.0046 | $-0.1034\ [-0.1772, -0.0285]$ | 0.0070 | **Yes ($p < 0.05$)** |
| **VADER vs. TF-IDF + Linear SVM** | 10.453 | 0.0011 | $-0.1257\ [-0.1999, -0.0510]$ | 0.0007 | **Yes ($p < 0.05$)** |
| **Loughran–McDonald vs. TF-IDF + Logistic Regression** | 25.536 | 0.0000 | $-0.2212\ [-0.3098, -0.1345]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **Loughran–McDonald vs. TF-IDF + Linear SVM** | 29.536 | 0.0000 | $-0.2436\ [-0.3310, -0.1573]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **TF-IDF + Logistic Regression vs. TF-IDF + Linear SVM** | 0.900 | 0.3438 | $-0.0223\ [-0.0512, +0.0038]$ | 0.1064 | **No** |

### 6. Visualizations

#### Confusion Matrices Across Baselines
![Confusion Matrices](results/step1/rq1_confusion_matrices.png)

#### RQ1 Neutral-to-Negative Misclassification Comparison
![RQ1 Neutral to Negative Comparison](results/step1/rq1_neutral_to_negative_comparison.png)

---

## Quick Start (Google Colab & Local Smoke Tests)

### 1. Run Full Step 1 Pipeline in Google Colab
Click the badge at the top of this README or run via CLI inside a Colab runtime:
```bash
python3 colab/run_step1_colab.py --mode full --execute --out-dir results/step1 --splits-dir data/splits --bootstrap-samples 10000
```

### 2. Run Local Unit & Smoke Tests (Zero Network / Synthetic Fixtures)
```bash
python3 -m unittest discover -s tests -v
```

---

## Repository Structure

```text
essential_text/
├── AGENTS.md                  # Shared project rules and leakage-free methodology constraints
├── README.md                  # Project overview, Step 1 results, and usage instructions
├── requirements.txt           # Python dependencies for Colab execution
├── data/
│   └── dataset_audit_verdict.json  # Pinned SHA-256 dataset provenance and row-level audit
├── proposal/
│   ├── proposal.tex           # Revised project proposal LaTeX source
│   ├── proposal.pdf           # Compiled revised 2-page project proposal
│   ├── methodology_and_rubric_audit.md
│   ├── rubric_compliance_matrix.md
│   ├── CHANGELOG_AND_RESPONSE.md
│   └── verify_proposal.py     # Automated proposal verification suite
├── results/
│   └── step1/                 # Full Step 1 evaluation outputs from Google Colab
│       ├── STEP1_REPORT.md
│       ├── summary_metrics.json
│       ├── statistical_comparisons.json
│       ├── rq1_confusion_matrices.png
│       └── rq1_neutral_to_negative_comparison.png
├── src/
│   ├── data/                  # SHA-256 hashing, Jaccard >= 0.80 clustering, split locking & dedup
│   ├── baselines/             # Tier 1 (VADER, Loughran-McDonald) & Tier 2 (TF-IDF + LR/SVM)
│   └── evaluation/            # Classification metrics, ECE, McNemar's test, paired bootstrap CIs
├── colab/
│   └── run_step1_colab.py     # End-to-end Step 1 pipeline runner (smoke & Colab full mode)
├── notebooks/
│   ├── build_step1_notebook.py
│   └── step1_data_and_baselines_colab.ipynb
└── tests/                     # Fast unittest suite (44 tests on synthetic financial fixtures)
```
