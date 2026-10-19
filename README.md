# Evaluating Domain Adaptation in Financial Sentiment Analysis

**FinBERT vs. Traditional NLP Baselines**  
*Authors: Qiyao Zheng, Benjamin Donick, Necati Furkan Colak*

## Overview

This repository contains the reproducible evaluation pipeline, leakage-free data splits, phenomenon-tagged diagnostic dataset, and experiment code for our study on **domain adaptation in 3-class financial sentiment classification** (*positive*, *neutral*, *negative*).

### Research Questions
1. **RQ1 (Domain Polarity Mismatch):** To what extent do general-purpose sentiment methods misclassify financial language, particularly by falsely assigning *negative* polarity to *neutral* descriptive accounting statements (e.g., *cost*, *liability*, *debt*, *depreciation*)?
2. **RQ2 (Controlled Architecture vs. Domain Adaptation):** Under a strictly leakage-free evaluation protocol, how much classification improvement stems from domain-adaptive pretraining (FinBERT) versus general-domain BERT-base and traditional machine-learning baselines?
3. **RQ3 (Linguistic Phenomenon Attribution):** Which linguistic phenomena—financial terminology, negation, hedging, numerical comparisons, or context-dependent expressions—drive model divergence?

## Repository Structure

```text
essential_text/
├── AGENTS.md                  # Shared project rules and leakage-free methodology constraints
├── README.md                  # Project overview and usage
├── proposal/
│   ├── proposal.tex           # Revised project proposal LaTeX source
│   ├── proposal.pdf           # Compiled revised 2-page project proposal
│   ├── methodology_and_rubric_audit.md
│   ├── rubric_compliance_matrix.md
│   ├── CHANGELOG_AND_RESPONSE.md
│   └── verify_proposal.py     # Automated proposal verification suite
├── data/                      # Dataset ingestion, SHA-256 split manifests, deduplication
├── src/                       # Lexicon, TF-IDF+LR/SVM, and BERT/FinBERT models & evaluation
└── results/                   # Metrics, confusion matrices, statistical tests, and error analysis
```
