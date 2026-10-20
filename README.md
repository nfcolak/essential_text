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

> **Important: what "VADER" and "Loughran–McDonald" mean in this Step 1 section.** The Step 1 Tier 1 rows were produced by the project's own embedded word lists in [`src/baselines/lexicon.py`](src/baselines/lexicon.py), not by the published lexicons: a 200-entry VADER-style list and 462 positive / 710 negative LM-style lists, in which 23 accounting terms (for example *cost*, *debt*, *liability*, *depreciation*) are forced to neutral and whose decision thresholds are tuned on the validation split. These lists already encode the answer to RQ1, so they are reported as **finance-adjusted, custom** variants. Step 2 adds the **official** lexicons (vaderSentiment 3.3.2 with its 7,506-entry lexicon and the fixed ±0.05 compound rule; the Loughran–McDonald Master Dictionary 1993–2025 with 354 positive / 2,355 negative words and plain counting) as the headline Tier 1 baselines. The official lexicons score about 0.50 Macro-F1 in-domain, well below the custom lists (0.57–0.75).

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
| **VADER-style (finance-adjusted, custom)** | Tier 1 (Lexicon) | 0.8201 | **0.7534** | 0.6667 | 0.8993 | 0.6941 | 0.1410 | 1.92% (4 / 208) |
| **LM-style (finance-adjusted, custom)** | Tier 1 (Lexicon) | 0.7345 | **0.6355** | 0.4865 | 0.8251 | 0.5949 | 0.2635 | 3.37% (7 / 208) |
| **TF-IDF + Logistic Regression** | Tier 2 (Linear ML) | 0.8938 | **0.8567** | 0.8172 | 0.9431 | 0.8098 | 0.0798 | 0.96% (2 / 208) |
| **TF-IDF + Linear SVM** | Tier 2 (Linear ML) | **0.9056** | **0.8791** | **0.8605** | **0.9434** | **0.8333** | **0.0366** | **0.48% (1 / 208)** |

### 3. 5-Fold Cross-Validation Performance (Training Partition, $N = 1,585$)

| Model | Mean Accuracy ($\pm$ Std) | Mean Macro-F1 ($\pm$ Std) | Mean Neu→Neg Rate ($\pm$ Std) |
|---|---:|---:|---:|
| **VADER-style (custom)** | $0.8000 \pm 0.0242$ | **$0.7165 \pm 0.0371$** | $0.0082 \pm 0.0025$ |
| **LM-style (custom)** | $0.7319 \pm 0.0187$ | **$0.6181 \pm 0.0267$** | $0.0215 \pm 0.0104$ |
| **TF-IDF + Logistic Regression** | $0.8593 \pm 0.0240$ | **$0.8003 \pm 0.0419$** | $0.0123 \pm 0.0106$ |
| **TF-IDF + Linear SVM** | **$0.8732 \pm 0.0131$** | **$0.8181 \pm 0.0276$** | **$0.0041 \pm 0.0039$** |

### 4. Deduplicated External Holdout Generalization (`SEntFiN 1.0`, $N = 9,460$)

| Model | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | ECE | Neu→Neg Error Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VADER-style (custom)** | **0.6344** | **0.6242** | 0.5757 | **0.6589** | **0.6380** | 0.3335 | 3.64% |
| **LM-style (custom)** | 0.5921 | 0.5850 | **0.5917** | 0.6205 | 0.5429 | 0.4049 | 6.37% |
| **TF-IDF + Logistic Regression** | 0.4921 | 0.4440 | 0.3304 | 0.5781 | 0.4235 | **0.2117** | 2.08% |
| **TF-IDF + Linear SVM** | 0.4895 | 0.4353 | 0.3235 | 0.5796 | 0.4029 | 0.4093 | **1.23%** |

> **Key finding on out-of-distribution generalization.** TF-IDF + Linear SVM is strong in-domain on Financial PhraseBank (Macro-F1 0.8791) but drops sharply on the external `SEntFiN 1.0` headline holdout (Macro-F1 0.4353), a vocabulary shift from company-report sentences to news headlines. The lexicon baselines drop less. This also holds for the official lexicons added in Step 2 (VADER 0.525, Loughran–McDonald 0.466 on SEntFiN vs 0.450 for TF-IDF + SVM), but less clearly than the custom VADER-style list (0.624) suggests.

### 5. Statistical Significance Comparisons (Held-Out Test Set, $B = 10,000$ Paired Bootstrap)

| Comparison (Model A vs. Model B) | McNemar $\chi^2$ | McNemar $p$-value | $\Delta$Macro-F1 (95% CI) | Bootstrap $p$-value | Significant ($p < 0.05$) |
|---|---:|---:|---:|---:|---|
| **VADER-style vs. LM-style (custom)** | 14.792 | 0.0001 | $+0.1178\ [+0.0609, +0.1776]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **VADER-style (custom) vs. TF-IDF + Logistic Regression** | 7.890 | 0.0046 | $-0.1034\ [-0.1772, -0.0285]$ | 0.0070 | **Yes ($p < 0.05$)** |
| **VADER-style (custom) vs. TF-IDF + Linear SVM** | 10.453 | 0.0011 | $-0.1257\ [-0.1999, -0.0510]$ | 0.0007 | **Yes ($p < 0.05$)** |
| **LM-style (custom) vs. TF-IDF + Logistic Regression** | 25.536 | 0.0000 | $-0.2212\ [-0.3098, -0.1345]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **LM-style (custom) vs. TF-IDF + Linear SVM** | 29.536 | 0.0000 | $-0.2436\ [-0.3310, -0.1573]$ | 0.0000 | **Yes ($p < 0.05$)** |
| **TF-IDF + Logistic Regression vs. TF-IDF + Linear SVM** | 0.900 | 0.3438 | $-0.0223\ [-0.0512, +0.0038]$ | 0.1064 | **No** |

### 6. Visualizations

#### Confusion Matrices Across Baselines
![Confusion Matrices](results/step1/rq1_confusion_matrices.png)

#### RQ1 Neutral-to-Negative Misclassification Comparison
![RQ1 Neutral to Negative Comparison](results/step1/rq1_neutral_to_negative_comparison.png)

---

## Step 2 Results: Controlled Transformer Fine-Tuning (RQ2)

Full artifacts: [`results/step2/`](results/step2/) ([`STEP2_REPORT.md`](results/step2/STEP2_REPORT.md),
`summary_metrics.json`, `statistical_comparisons.json`, `attribution.json`, figures) and one directory per run in
`results/step2/runs/<config>/<model>/<tag>/` with `result.json` and per-sentence probabilities (`predictions.npz`).

**Protocol.** `google-bert/bert-base-uncased` (general domain) and `yiyanghkust/finbert-pretrain` (financial MLM,
no sentiment supervision) are fine-tuned with one shared recipe (AdamW, lr 2e-5, batch 16, max length 128, 10%
warmup, weight decay 0.01, gradient clipping 1.0, at most 5 epochs, patience 2) on the identical locked Step 1
splits of each PhraseBank agreement level, 3 seeds each (42, 100, 2026). The epoch is chosen by validation Macro-F1
only; test and SEntFiN 1.0 are scored once. Off-the-shelf `ProsusAI/finbert` was trained on PhraseBank, so it is
scored **only** on SEntFiN 1.0 (zero-shot). All baselines were refit in the same environment, so every paired test
compares predictions on identical sentences. Tier 1 has three variants: the plain official lexicons with their
fixed published rules; the proposal's specification, i.e. the official lexicons with decision thresholds tuned on the
validation split (and 3-token negation handling for Loughran–McDonald); and the custom finance-adjusted lists from
Step 1. TF-IDF models tune C ∈ {0.01, 0.1, 1, 10} and the class weight on validation (C = 1 or 10 was chosen in every
run). Hugging Face revisions are pinned in `src/transformer/finetune.py`; split files are checked against
`data/splits/SHA256SUMS` before every run. Runs: 93 (24 baseline, 18 fine-tune, 50 CV, 1 zero-shot) on Apple MPS
(Python 3.9.6, torch 2.8.0, transformers 4.57.6, scikit-learn 1.6.1; exact versions in `requirements-lock.txt`).

The three agreement levels are nested (a 100%-agreement sentence is also in 75agree and 50agree), so columns are
compared within a config; across configs the comparison is descriptive.

### In-domain test Macro-F1 (mean ± sd over 3 seeds; baselines are single runs with seed 42)

| Model | allagree (N = 339) | 75agree (N = 518) | 50agree (N = 726) |
|---|---:|---:|---:|
| VADER (official lexicon) | 0.500 | 0.515 | 0.491 |
| VADER (official lexicon, val-tuned thresholds) | 0.507 | 0.542 | 0.506 |
| Loughran–McDonald (official dictionary) | 0.498 | 0.518 | 0.502 |
| Loughran–McDonald (official, negation + val-tuned thresholds) | 0.505 | 0.520 | 0.503 |
| VADER-style (finance-adjusted, custom) | 0.753 | 0.646 | 0.645 |
| LM-style (finance-adjusted, custom) | 0.636 | 0.597 | 0.567 |
| TF-IDF + Logistic Regression | 0.869 | 0.776 | 0.762 |
| TF-IDF + Linear SVM | 0.870 | 0.791 | 0.768 |
| `bert-base-uncased` | **0.974 ± 0.013** | **0.906 ± 0.004** | **0.872 ± 0.007** |
| FinBERT-MLM (`finbert-pretrain`) | 0.944 ± 0.005 | 0.894 ± 0.004 | 0.846 ± 0.014 |

### External SEntFiN 1.0 Macro-F1 (N = 9,460; models trained on each PhraseBank level)

| Model | trained on allagree | trained on 75agree | trained on 50agree |
|---|---:|---:|---:|
| VADER (official lexicon) | 0.525 | 0.525 | 0.525 |
| VADER (official lexicon, val-tuned thresholds) | 0.481 | 0.528 | 0.525 |
| Loughran–McDonald (official dictionary) | 0.466 | 0.466 | 0.466 |
| Loughran–McDonald (official, negation + val-tuned thresholds) | 0.469 | 0.469 | 0.469 |
| VADER-style (finance-adjusted, custom) | 0.624 | 0.625 | 0.624 |
| LM-style (finance-adjusted, custom) | 0.585 | 0.585 | 0.585 |
| TF-IDF + Logistic Regression | 0.451 | 0.475 | 0.475 |
| TF-IDF + Linear SVM | 0.450 | 0.472 | 0.469 |
| `bert-base-uncased` | 0.685 ± 0.021 | 0.676 ± 0.012 | 0.676 ± 0.018 |
| FinBERT-MLM | 0.669 ± 0.028 | 0.653 ± 0.047 | 0.704 ± 0.010 |
| `ProsusAI/finbert` (off-the-shelf, zero-shot) | **0.727** | | |

5-fold CV on the allagree training partition (scored on the fold used for epoch selection, so optimistic):
FinBERT-MLM 0.952 ± 0.013, `bert-base-uncased` 0.950 ± 0.013, TF-IDF + SVM 0.816 ± 0.028, VADER-style (custom)
0.716 ± 0.041, VADER (official, val-tuned) 0.529 ± 0.039, VADER (official) 0.492 ± 0.038, Loughran–McDonald
(official, val-tuned) 0.483 ± 0.026, Loughran–McDonald (official) 0.480 ± 0.026.

Tuning the thresholds of the official lexicons on validation, as the proposal specified, adds at most +0.03
Macro-F1; the large gap to the custom lists (0.57–0.75) comes from their edited word lists, not from tuning.

### RQ1: the neutral/negative boundary (neutral recall R_neu, negative precision P_neg)

As planned in the proposal's RQ1 traceability matrix: R_neu and P_neg per model, Δ with the seed + sentence
bootstrap (B = 10,000) and McNemar on gold-neutral sentences only (correct = predicted neutral), Holm within each
seed family. Full tables for every config: `STEP2_REPORT.md`, section "RQ1: neutral/negative boundary", and
`results/step2/rq1_boundary.json`.

| Pair (A vs B), allagree-trained | Split | R_neu A / B | ΔR_neu [95% CI] | P_neg A / B | ΔP_neg [95% CI] |
|---|---|---|---:|---|---:|
| VADER (official) vs LM (official) | test | 0.591 / 0.899 | −0.308 [−0.390, −0.225] | 0.345 / 0.475 | −0.130 [−0.321, +0.056] |
| VADER vs LM (official, val-tuned) | test | 0.731 / 0.899 | −0.168 [−0.244, −0.093] | 0.421 / 0.576 | −0.155 [−0.386, +0.071] |
| BERT vs FinBERT-MLM | test | 0.992 / 0.978 | +0.014 [−0.008, +0.040] | 1.000 / 0.955 | +0.045 [+0.000, +0.116] |
| VADER (official) vs LM (official) | SEntFiN | 0.590 / 0.871 | −0.281 [−0.298, −0.263] | 0.656 / 0.687 | −0.031 [−0.055, −0.009] |
| VADER vs LM (official, val-tuned) | SEntFiN | 0.729 / 0.871 | −0.142 [−0.158, −0.126] | 0.704 / 0.696 | +0.008 [−0.017, +0.033] |
| BERT vs FinBERT-MLM | SEntFiN | 0.858 / 0.879 | −0.021 [−0.071, +0.031] | 0.855 / 0.848 | +0.006 [−0.012, +0.025] |
| ProsusAI vs BERT | SEntFiN | 0.752 / 0.858 | −0.106 [−0.136, −0.068] | 0.757 / 0.855 | −0.097 [−0.114, −0.081] |

**Reading.** The general-purpose lexicon loses many more neutral sentences than the financial one: VADER's neutral
recall is 0.14–0.34 below Loughran–McDonald's in every config and on SEntFiN (all intervals exclude zero; neutral-only
McNemar significant in 3/3 seed families). This is the RQ1 effect at the lexicon level, and validation tuning only
halves it. Negative precision differs much less between the two lexicons (in-domain intervals include zero). Between
the controlled encoders there is no domain-pretraining advantage at this boundary: ΔR_neu (BERT − FinBERT-MLM) includes
zero in all six config × split cells, and ΔP_neg is significant only on SEntFiN after 50agree training (+0.048, BERT
better; the allagree test interval touches zero, p = 0.17). The off-the-shelf ProsusAI/finbert has lower neutral recall and negative precision on SEntFiN than both
controlled encoders.

### Architecture vs. domain-pretraining attribution (ΔMacro-F1, seed + sentence bootstrap, B = 10,000)

`Δarch` = BERT − best TF-IDF model (chosen by **validation** Macro-F1; SVM in every config). It compares a
pretrained contextual encoder with a bag-of-n-grams model, so it includes general-domain pretraining, not only the
architecture. `Δdomain` = FinBERT-MLM − BERT. The interval resamples sentences **and** the three trained seeds, so it
includes seed-to-seed variation.

| Config | Split | Δarch [95% CI] | Δdomain [95% CI] | Δdomain per seed (42, 100, 2026) |
|---|---|---:|---:|---|
| allagree | test | +0.104 [+0.062, +0.150] | −0.029 [−0.057, −0.006] | −0.030, −0.015, −0.044 |
| 75agree | test | +0.116 [+0.073, +0.160] | −0.012 [−0.039, +0.014] | −0.011, −0.018, −0.008 |
| 50agree | test | +0.104 [+0.067, +0.143] | −0.026 [−0.056, +0.005] | −0.018, −0.025, −0.035 |
| allagree | SEntFiN | +0.235 [+0.215, +0.259] | −0.016 [−0.048, +0.016] | +0.023, −0.066, −0.004 |
| 75agree | SEntFiN | +0.203 [+0.187, +0.219] | −0.023 [−0.068, +0.022] | −0.024, +0.013, −0.057 |
| 50agree | SEntFiN | +0.207 [+0.186, +0.227] | +0.028 [+0.007, +0.050] | +0.033, +0.013, +0.037 |

**Reading.** Most of the gain over the traditional baselines comes from the pretrained contextual encoder (about
+0.10 in-domain, +0.20 to +0.24 on SEntFiN; all six intervals exclude zero). Under one shared recipe, domain-adaptive
MLM pretraining adds nothing measurable: in-domain it is slightly negative (−0.01 to −0.03; only the allagree interval
excludes zero), and on SEntFiN two of three cells have intervals around zero and change sign across seeds. The one
positive cell is FinBERT-MLM trained on 50agree (+0.028, all three seeds positive); 50agree is both the largest and
the noisiest training set, and this design cannot separate the two. The off-the-shelf `ProsusAI/finbert` is best on
SEntFiN, but it had more sentiment supervision (all of PhraseBank), so it is a reference point, not a controlled
comparison. McNemar tests are reported per seed (no single "primary" seed), with Holm–Bonferroni within each seed
family; the BERT vs FinBERT-MLM difference on allagree test is significant in one of three seed families.

![Macro-F1 by config](results/step2/step2_macro_f1_by_config.png)
![Attribution](results/step2/step2_attribution.png)

---

## Step 3 Results: Linguistic Phenomenon Attribution (RQ1, RQ3)

Full artifacts: [`results/step3/`](results/step3/) ([`STEP3_REPORT.md`](results/step3/STEP3_REPORT.md),
`annotation_agreement.json`, `phenomenon_analysis.json`, figures) and the diagnostic set in
[`data/diagnostic/`](data/diagnostic/) (see its [`README.md`](data/diagnostic/README.md) for provenance and licence).

**Diagnostic set.** 300 sentences, label-stratified, drawn only from the `sentences_50agree` **test** split
(seed 42; 37 negative, 178 neutral, 85 positive). Each sentence is tagged for five phenomena: financial
terminology, negation, hedging, numerical comparison, context-dependent polarity
([`annotation/GUIDELINES.md`](annotation/GUIDELINES.md)).

**Annotators are LLMs, not humans.** Annotator A was Claude Sonnet 5.5 and annotator B was Gemini 3.8 Flash.
Each worked blind (text and guideline only: no gold label, no prediction, no rule tags, never the other annotator's
file) and tagged by reading, not by keyword scripts; this was verified by the supervising agent from the session
transcripts, which are not part of the repository. Both models may have seen Financial PhraseBank sentences during
pretraining. A third reference is the rule-based tagger `src/analysis/phenomena.py`. Consensus tags require both
annotators. The first round used guideline v1; its context-dependent definition covered only 3 of 300 sentences
and missed cases such as "operating profit decreased", so the definition was widened (v2: a direction word applied
to a positive- or negative-valence financial target, plus job cuts) and that one tag was re-annotated blind by the
same two models. The v1 files are kept as `annotations_{a,b}_v1.csv`.

| Phenomenon | κ A–B | κ A–rules | κ B–rules | consensus n |
|---|---:|---:|---:|---:|
| financial terminology | 0.961 | 0.925 | 0.961 | 13 |
| negation | 1.000 | 1.000 | 1.000 | 10 |
| hedging | 0.893 | 0.778 | 0.827 | 53 |
| numerical comparison | 0.965 | 0.895 | 0.882 | 50 |
| context-dependent (v2) | 0.975 | 0.645 | 0.654 | 47 |

Agreement is high partly because the guideline defines each phenomenon with explicit cue words, so both LLMs
and the rule tagger look at similar cues; for financial terminology and negation the LLM consensus differs from the
rule tags on at most 1 sentence. Subsets with n < 20 get confidence intervals only, no p-values.

**Error rate on the diagnostic subset (seed mean; models trained on 50agree).** Hedging (n = 53): VADER (official)
0.340, VADER-style 0.264, TF-IDF + SVM 0.208, BERT 0.088, FinBERT-MLM 0.138. Numerical comparison (n = 50): VADER
(official) 0.600, Loughran–McDonald (official) 0.820, TF-IDF + SVM 0.120, BERT 0.033, FinBERT-MLM 0.067. Error
rates depend on the gold-label mix of each subset (numerical-comparison sentences are almost never neutral), so the
report also gives label-balanced rates and the label mix. On this subset the architecture effect on hedging is
+0.119 accuracy [+0.019, +0.233] (p = 0.030), which does **not** survive Holm correction over the ten tests of the
table (Holm p = 0.18); no domain effect is distinguishable from zero.

**Full test splits and SEntFiN with rule tags** (report section 5; larger n, cruder tags). On allagree test, Δarch
is significant after Holm for financial terminology (+0.185), hedging (+0.175) and numerical comparison (+0.071); on
SEntFiN it is significant for all five phenomena in every config. Δdomain is significant after Holm in only two of
the 30 phenomenon × config cells, with opposite signs (75agree, SEntFiN, numerical comparison −0.101; 50agree,
SEntFiN, context-dependent +0.091), so there is no stable phenomenon-level domain-pretraining effect.

**RQ1 (accounting terms → false negative).** On SEntFiN 1.0, gold-neutral headlines with a financial term (rule
tags, n = 142) are labelled negative far more often than other neutral headlines (n = 3,267): VADER (official) 0.570
vs 0.078, Loughran–McDonald (official) 0.479 vs 0.073, BERT 0.322–0.378 vs 0.029–0.043 and FinBERT-MLM 0.146–0.385
vs 0.031–0.081 (range over the three training levels), ProsusAI 0.415 vs 0.109. **Most of this effect is one cue
word:** 47 of the 142 headlines contain *loss/losses*, and most models tag them negative (lexicons 98–100%, BERT
79–94%, ProsusAI 94%, FinBERT-MLM 79–82% except 32% when trained on 75agree; TF-IDF 11–40%). SEntFiN often labels a reported loss as neutral for the named entity, whereas PhraseBank
labels most reported losses negative, so this part reflects a label-convention difference between the datasets as
much as model error. Without the loss-only headlines (n = 96) the gap shrinks to VADER (official) 0.375 vs 0.078,
Loughran–McDonald (official) 0.229 vs 0.073, BERT 0.083–0.108 vs 0.029–0.043, FinBERT-MLM 0.069–0.191 vs
0.031–0.081, ProsusAI 0.167 vs 0.109 (interval includes zero). So the RQ1 mechanism is clear for the
general-purpose lexicons, small for the fine-tuned encoders. Inside the PhraseBank test splits the term subsets hold
only 6–15 gold-neutral sentences, too few for a conclusion.

![Phenomenon error heatmap](results/step3/step3_phenomenon_error_heatmap.png)
![RQ1 accounting terms](results/step3/step3_rq1_accounting_terms.png)

### Caveats

- One shared fine-tuning recipe, no per-model hyperparameter search; FinBERT-MLM may need a different learning
  rate or more epochs, so `Δdomain` measures "domain pretraining under the BERT recipe".
- The Step 1 tables come from Google Colab. Rerun locally, the splits and lexicon baselines are identical, while
  TF-IDF Macro-F1 moves by up to 0.012 (different scikit-learn/scipy builds). Steps 2 and 3 use the local refit
  throughout, which is why their TF-IDF numbers differ slightly from Step 1.
- Six fine-tunes first hit Apple MPS "Insufficient Memory" command-buffer errors (four with NaN loss). They were
  quarantined and rerun in fresh processes with zero errors; the runner now rejects non-finite losses and
  probabilities and a transformer that predicts a single class.
- The context-dependent definition was widened after the first-round κ (0.593) was known; the v2 κ (0.975) is
  therefore not an independent test of the original guideline.

### Deviations from the proposal

- **Annotation.** The proposal planned double annotation by two team members; it was done by two blind LLM
  annotators (above). Human spot-checking of a sample is still open.
- **Tier 1 lexicons.** Step 1 used custom finance-adjusted word lists under the names VADER and Loughran–McDonald;
  the official lexicons (plain, and with the proposal's validation-tuned thresholds and LM negation handling) were
  added in Step 2 and are now the headline Tier 1 rows.
- **Split manifests.** The full manifests (with sentence text, about 11 MB) stay git-ignored and are regenerated by
  one command (`--fpb-config all`), pinned by `data/splits/SHA256SUMS`, which `run_step2.py` checks before running.
  The committed deliverable is [`data/splits/assignments/`](data/splits/assignments/): text-free per-config split
  and 5-fold CV assignments plus the SEntFiN holdout ids, keyed by sentence id and SHA-256
  (`scripts/export_split_assignments.py --check` verifies them against the manifests).
- **Pre-tagger.** The proposal planned lexical/dependency pre-tagging; negation scope is detected with token windows
  and regular expressions (`src/analysis/phenomena.py`), not a dependency parser.
- **FiQA.** The proposal mentioned FiQA headlines next to SEntFiN 1.0 as external data; only SEntFiN 1.0 is used.
- **Cross-validation.** 5-fold stratified CV is run on the allagree training partition only (all baselines and both
  encoders); the 75agree and 50agree levels use the single locked 70/15/15 split with three seeds.
- **SEntFiN size.** The proposal planned a 2,000-sentence holdout; all 9,460 unambiguous, deduplicated headlines
  are used.
- **κ criterion.** The proposal's acceptance criterion κ ≥ 0.75 failed for context-dependent in round 1 (0.593); it
  holds after the v2 re-annotation, with the caveat above.
- **Phenomenon deltas.** Defined on F1 in the proposal; Step 3 reports them on accuracy because several subsets
  have very few sentences of one class, where per-class F1 is undefined or unstable.
- **Effort record.** The proposal's 55-hour per-member work plan is not documented against actual hours in this
  repository.

---

## Quick Start

### 1. Step 1 (data, splits, baselines)
```bash
python3 colab/run_step1_colab.py --mode full --execute --fpb-config all --out-dir results/step1 --splits-dir data/splits --bootstrap-samples 10000
```
`--fpb-config all` writes the locked manifests of all three agreement levels to `data/splits/` (git-ignored,
regenerated deterministically; Step 2 runs record their SHA-256).

### 2. Step 2 (fine-tuning matrix, resumable; prints the plan without `--execute`)
```bash
# official Loughran-McDonald dictionary (git-ignored; SHA-256 checked by src/baselines/lexicon_official.py)
curl -L -o data/raw/LM_MasterDictionary_1993-2025.csv "https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
python3 scripts/run_step2.py --stage all --execute
python3 scripts/export_split_assignments.py --check            # verify committed split assignments
python3 scripts/summarize_step2.py --runs-dir results/step2/runs --out-dir results/step2 --bootstrap-samples 10000
```

### 3. Step 3 (diagnostic set and phenomenon analysis)
```bash
python3 scripts/build_diagnostic_set.py --execute          # 300-sentence subset, key and rule tags
python3 scripts/build_diagnostic_set.py --execute --rule-tags-only   # recompute rule tags only
python3 scripts/run_step3_analysis.py                      # needs data/diagnostic/annotations_{a,b}.csv
```

### 4. Unit tests (offline, synthetic fixtures, tiny local models)
```bash
HF_HUB_OFFLINE=1 python3 -m unittest discover -s tests -v
```

---

## Repository Structure

```text
essential_text/
├── AGENTS.md                  # Shared project rules and leakage-free methodology constraints
├── README.md                  # Project overview, results, and usage instructions
├── requirements.txt           # Python dependencies (imports only)
├── requirements-lock.txt      # Exact versions of the run environment
├── annotation/
│   └── GUIDELINES.md          # Five-phenomenon annotation guideline
├── data/
│   ├── dataset_audit_verdict.json  # Pinned SHA-256 dataset provenance and row-level audit
│   ├── splits/SHA256SUMS      # SHA-256 of the locked split files (the files themselves are regenerated)
│   ├── splits/assignments/    # Text-free split / CV-fold / SEntFiN-holdout assignments (committed deliverable)
│   └── diagnostic/            # 300-sentence diagnostic set: items, key, rule tags, LLM annotations A/B, README
├── proposal/                  # Revised proposal (LaTeX/PDF), rubric audit, verification script
├── results/
│   ├── step1/                 # Step 1 outputs from Google Colab
│   ├── step2/                 # Step 2 report, statistics, attribution, figures, runs/ with per-run predictions
│   └── step3/                 # Step 3 report, agreement, phenomenon analysis, figures
├── scripts/
│   ├── run_step2.py           # Resumable Step 2 run matrix (baselines, fine-tune, CV, zero-shot)
│   ├── summarize_step2.py     # Multi-seed statistics, McNemar + Holm, bootstrap, attribution, figures
│   ├── build_diagnostic_set.py
│   └── run_step3_analysis.py  # Cohen's kappa, per-phenomenon errors and attribution, RQ1 analysis
├── src/
│   ├── data/                  # Hashing, Jaccard >= 0.80 clustering, split locking, dedup, IO helpers
│   ├── baselines/             # Tier 1 official + custom lexicons, Tier 2 TF-IDF + LR/SVM, registry
│   ├── transformer/           # Tier 3 fine-tuning and zero-shot scoring (PyTorch + Hugging Face)
│   ├── evaluation/            # Metrics, ECE, McNemar, paired bootstrap, multi-seed statistics
│   └── analysis/              # Phenomenon tagger and per-phenomenon attribution
├── colab/
│   └── run_step1_colab.py     # End-to-end Step 1 pipeline runner (smoke & full mode)
├── notebooks/
│   ├── build_step1_notebook.py
│   └── step1_data_and_baselines_colab.ipynb
└── tests/                     # Offline unittest suite (189 tests)
```
