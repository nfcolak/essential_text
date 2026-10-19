# Evaluating Domain Adaptation in Financial Sentiment Analysis

**FinBERT vs. Traditional NLP Baselines**  
*Authors: Qiyao Zheng, Benjamin Donick, Necati Furkan Colak*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nfcolak/essential_text/blob/main/notebooks/step1_data_and_baselines_colab.ipynb)

## Overview

This repository contains the reproducible evaluation pipeline, leakage-free data splits, phenomenon-tagged diagnostic dataset, and experiment code for our study on **domain adaptation in 3-class financial sentiment classification** (*positive*, *neutral*, *negative*).

### Research Questions
1. **RQ1 (Domain Polarity Mismatch):** To what extent do general-purpose sentiment methods misclassify financial language, particularly by falsely assigning *negative* polarity to *neutral* descriptive accounting statements (e.g., *cost*, *liability*, *debt*, *depreciation*)?
2. **RQ2 (Controlled Architecture vs. Domain Adaptation):** Under a strictly leakage-free evaluation protocol, how much classification improvement stems from domain-adaptive pretraining (FinBERT) versus general-domain BERT-base and traditional machine-learning baselines?
3. **RQ3 (Linguistic Phenomenon Attribution):** Which linguistic phenomena—financial terminology, negation, hedging, numerical comparisons, or context-dependent expressions—drive model divergence?

## Quick Start (Google Colab & Local Smoke Tests)

### 1. Run Full Step 1 Pipeline in Google Colab
Click the badge above or open:
- **Colab Link:** `https://colab.research.google.com/github/nfcolak/essential_text/blob/main/notebooks/step1_data_and_baselines_colab.ipynb`

Or run via CLI inside a Colab runtime:
```bash
python3 colab/run_step1_colab.py --mode full --execute --out-dir results/step1 --splits-dir data/splits
```

### 2. Run Local Unit & Smoke Tests (Zero Network / Synthetic Fixtures)
```bash
python3 -m unittest discover -s tests -v
```

## Repository Structure

```text
essential_text/
├── AGENTS.md                  # Shared project rules and leakage-free methodology constraints
├── README.md                  # Project overview and usage
├── requirements.txt           # Python dependencies for Colab execution
├── proposal/
│   ├── proposal.tex           # Revised project proposal LaTeX source
│   ├── proposal.pdf           # Compiled revised 2-page project proposal
│   ├── methodology_and_rubric_audit.md
│   ├── rubric_compliance_matrix.md
│   ├── CHANGELOG_AND_RESPONSE.md
│   └── verify_proposal.py     # Automated proposal verification suite
├── src/
│   ├── data/                  # SHA-256 hashing, Jaccard >= 0.80 clustering, split locking & dedup
│   ├── baselines/             # Tier 1 (VADER, Loughran-McDonald) & Tier 2 (TF-IDF + LR/SVM)
│   └── evaluation/            # Classification metrics, ECE, McNemar's test, paired bootstrap CIs
├── colab/
│   └── run_step1_colab.py     # End-to-end Step 1 pipeline runner (smoke & Colab full mode)
├── notebooks/
│   ├── build_step1_notebook.py
│   └── step1_data_and_baselines_colab.ipynb
└── tests/                     # Fast unittest suite (43 tests on synthetic financial fixtures)
```
