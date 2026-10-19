# Methodology & Rubric Audit: Resolving Instructor Feedback on Financial Sentiment Analysis Proposal

**Project Title:** Evaluating Domain Adaptation in Financial Sentiment Analysis: FinBERT vs. Traditional NLP Baselines  
**Authors:** Qiyao Zheng, Benjamin Donick, Necati Furkan Colak  
**Target Deliverable:** Revised 2-page Course Project Proposal (`proposal/proposal.tex` $\rightarrow$ `proposal/proposal.pdf`)  

---

## 1. Executive Summary

The original project proposal (`proposal/proposal_v1_original.pdf`) received two substantive methodological critiques from the teaching team alongside the 7-criteria course proposal grading rubric:

1. **Invalid Text-Generation Evaluation Metrics:** Why are `BLEU`, `ROUGE`, and `METEOR` listed as evaluation metrics for a 3-class sentiment classification task?
2. **FinBERT Training Data Leakage on Financial PhraseBank:** Given that off-the-shelf `ProsusAI/finbert` (Araci, 2019) was supervisedly fine-tuned on the Financial PhraseBank dataset, how does the experimental design prevent data leakage when evaluating on Financial PhraseBank?
3. **Course Rubric Completeness:** How does the revised proposal explicitly satisfy all seven grading dimensions (`Scope`, `Value`, `Baseline`, `Analysis`, `Data accessible`, `NLP+code`, `Method fits`) to earn a `go ahead!` verdict?

This audit documents the technical root causes of both methodological issues in `v1` and specifies the exact mathematical, experimental, and architectural fixes implemented in `v2`.

---

## 2. Critique 1: Why `BLEU`, `ROUGE`, and `METEOR` Are Invalid for Sentiment Classification—and What Replaces Them

### 2.1 Diagnosis of the `v1` Flaw

In Section *Data and Experimental Setup* of `proposal_v1_original.pdf`, the evaluation paragraph stated:

> *"Performance will be evaluated using BLEU, ROUGE, METEOR, accuracy, macro-F1, precision, recall, and confusion matrices, with particular attention to class-specific errors and the neutral class."*

Including `BLEU`, `ROUGE`, and `METEOR` in a categorical sentiment classification study is a category error between **sequence-to-sequence text generation** and **discrete multi-class classification**:

1. **Mathematical Definition Mismatch:**
   - **BLEU** (Papineni et al., 2002) computes geometric means of modified $n$-gram precisions ($n \in \{1, 2, 3, 4\}$) multiplied by a sequence-length brevity penalty $\text{BP} = \min(1, \exp(1 - r/c))$ between a candidate token sequence $c$ and reference text $r$.
   - **ROUGE-N / ROUGE-L** (Lin, 2004) measures $n$-gram recall and longest common subsequence (LCS) overlap between generated summaries and reference summaries.
   - **METEOR** (Banerjee & Lavie, 2005) computes harmonic means of unigram precision and recall using WordNet synonymy, stemming, and a chunk-fragmentation penalty across multi-word sequences.
2. **Incompatible Output Space:**
   - In 3-class financial sentiment analysis, each sentence $x_i$ is mapped to a discrete element of an unordered categorical label set $\mathcal{Y} = \{\text{positive}, \text{neutral}, \text{negative}\}$ (or integer class IDs $\{0, 1, 2\}$) via a predicted probability simplex vector $\hat{\mathbf{p}}_i \in \Delta^2$.
   - None of the models under study (VADER, Loughran–McDonald, TF-IDF + Logistic Regression/SVM, `bert-base-uncased`, or FinBERT) generate free-form text sequences.
   - If applied naively to single-word class names, higher-order $n$-grams ($n \ge 2$) are undefined ($0/0$), brevity and fragmentation penalties are vacuous, and unigram overlap collapses to a noisy proxy for binary accuracy that ignores class imbalance and threshold calibration.

### 2.2 Replacement Classification & Statistical Evaluation Protocol in `v2`

In `proposal/proposal.tex` (`v2`), **all references to `BLEU`, `ROUGE`, and `METEOR` are completely removed** (enforced as a hard zero-match invariant across both `proposal.tex` and `proposal.pdf`). They are replaced by a rigorous multi-class classification and statistical inference suite tailored to the skewed class distribution of financial text:

| Metric / Test | Role in `v2` Evaluation | Technical Justification & Mathematical Formulation |
| :--- | :--- | :--- |
| **Macro-F1** *(Primary Metric)* | Headline model comparison & early-stopping criterion | Financial PhraseBank is strongly imbalanced (~59.4% `neutral`, ~28.1% `positive`, ~12.5% `negative` in the full 4,845-sentence corpus; 61.4% `neutral`, 25.2% `positive`, 13.4% `negative` in `sentences_allagree`). Unweighted $\text{Macro-F1} = \frac{1}{3}\sum_{c=1}^{3} F_{1,c}$ weights the minority `negative` class equally with `neutral`, preventing majority-class inflation. |
| **Accuracy** | Secondary aggregate metric | $\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$. Reported alongside Macro-F1 for direct comparability with Malo et al. (2014) [1] and Araci (2019) [3]. |
| **Per-Class Precision ($P_c$), Recall ($R_c$), and $F_{1,c}$** | Diagnostic boundary evaluation (`neutral` vs. `negative`) | Quantifies the core domain-adaptation hypothesis (RQ1): general-domain models treat routine accounting terms (*cost*, *liability*, *depreciation*, *debt*) as negative, causing high false-positive rates on `negative` (low $P_{\text{neg}}$) and high false-negative rates on `neutral` (low $R_{\text{neu}}$). |
| **$3 \times 3$ Confusion Matrices** | Directional error flow analysis | Full cross-tabulation $C_{jk} = \sum_{i} \mathbb{I}(y_i = j \land \hat{y}_i = k)$ across $\{\text{positive}, \text{neutral}, \text{negative}\}$ for both full test sets and phenomenon-specific subsets. |
| **McNemar's Paired Test** | Pairwise statistical significance of accuracy/error shifts | Tests whether two classifiers $A$ and $B$ evaluated on the *exact same test instances* have significantly different error rates using discordant pairs $n_{01}$ ($A$ wrong, $B$ right) and $n_{10}$ ($A$ right, $B$ wrong): $\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$ (or exact binomial test when $n_{01} + n_{10} < 25$), with Holm–Bonferroni correction across model pairs. |
| **Paired Bootstrap Resampling ($B = 10{,}000$)** | Confidence intervals & $p$-values for $\Delta\text{Macro-F1}$ | Because Macro-F1 is a non-linear ratio of counts across classes, McNemar's test alone cannot test $\Delta\text{Macro-F1}$. We draw $B = 10{,}000$ paired bootstrap samples of size $N_{\text{test}}$ with replacement to compute 95% percentile confidence intervals and two-sided empirical $p$-values for $\text{Macro-F1}(A) - \text{Macro-F1}(B)$. |

---

## 3. Critique 2: Why Off-the-Shelf `ProsusAI/finbert` Leaks Financial PhraseBank—and the Two-Stage Leakage-Free Protocol

### 3.1 Diagnosis of the `v1` Flaw

In `proposal_v1_original.pdf`, the setup proposed comparing general-domain BERT and FinBERT [3] on Financial PhraseBank [1] and noted *"avoiding data leakage"* as a general challenge without specifying checkpoint lineage or split isolation.

#### How `ProsusAI/finbert` Was Trained (Araci, 2019)
Araci (2019) [3] constructed FinBERT in two distinct stages:
1. **Stage A — Unsupervised Domain-Adaptive Pretraining (MLM + NSP):** Initialized from `bert-base-uncased` (Devlin et al., 2019 [4]) and further pretrained on a large unlabeled financial corpus (TRC2-financial news stories from Reuters + corporate financial filings). Crucially, **Stage A uses zero sentiment labels and does not include Financial PhraseBank annotations**. Similarly, `yiyanghkust/finbert-pretrain` (Yang et al., 2020) is pretrained purely via Masked Language Modeling on 4.9B tokens of corporate 10-K/10-Q filings, earnings call transcripts, and analyst reports prior to any sentiment supervision.
2. **Stage B — Supervised Sentiment Fine-Tuning on Financial PhraseBank:** Araci (2019) then attached a 3-class linear classification head to the Stage A encoder and fine-tuned the full network **directly on Financial PhraseBank** [1].

The widely downloaded Hugging Face checkpoint `ProsusAI/finbert` is the **post-Stage B model**. Therefore:
- **Direct Test-Set Contamination:** Any train/validation/test split created today from Financial PhraseBank will have ~80–85% of its test sentences overlapped with the exact labeled sentences `ProsusAI/finbert` memorized during Stage B fine-tuning.
- **Confounded Experimental Attribution:** Even if one attempted to reconstruct Araci's original split seed, comparing an externally fine-tuned checkpoint against newly trained `bert-base-uncased` or TF-IDF baselines confounds **domain-adaptive pretraining** with differences in hyperparameter schedules, regularization, early-stopping criteria, and split composition.

### 3.2 Two-Stage Leakage-Free Evaluation Protocol in `v2`

To eliminate all forms of data leakage while cleanly isolating the effect of domain adaptation (RQ2), `proposal/proposal.tex` (`v2`) establishes a strict **two-stage experimental design**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Controlled In-Domain Fine-Tuning on Locked PhraseBank Splits       │
│                                                                              │
│  Financial PhraseBank (allagree=2,264 | 75agree=3,453 | full=4,845)          │
│       │                                                                      │
│       ▼  Near-duplicate grouping (Jaccard >= 0.80) + SHA-256 Split Lock      │
│  Stratified 70% Train / 15% Val / 15% Test  (+ 5-Fold Stratified CV)         │
│       │                                                                      │
│       ├──► Tier 1: VADER [6] & Loughran-McDonald [2] (Zero-shot rules)       │
│       ├──► Tier 2: TF-IDF (fit on Train ONLY) + LogReg / Linear SVM          │
│       ├──► Tier 3a: bert-base-uncased [4] (MLM-only) ──► Fine-tune on Train  │
│       └──► Tier 3b: Pre-Sentiment FinBERT MLM Encoder [3] ──► Fine-tune      │
│                     (domain MLM prior to sentiment labels)     on Train      │
│                                                                              │
│  ► Result: Zero test leakage; identical splits & hyperparameters isolate     │
│            the pure causal effect of financial MLM pretraining vs BERT-base. │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: External Out-of-Distribution (OOD) Holdout on SEntFiN 1.0 [5]       │
│                                                                              │
│  SEntFiN 1.0 (10,753 entity-annotated financial headlines)                   │
│       │                                                                      │
│       ▼  Filter unambiguous 3-class sentences + Cross-Dataset Deduplication  │
│          against ALL 4,845 PhraseBank sentences (token Jaccard < 0.80)       │
│  Strictly Disjoint External Test Set (N = 2,000 stratified headlines)        │
│       │                                                                      │
│       └──► Evaluate ALL Stage 1 models + off-the-shelf ProsusAI/finbert      │
│            zero-shot without any training-set overlap.                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Stage 1 Details: Controlled Fine-Tuning from Pre-Sentiment Domain Checkpoints
1. **Checkpoint Selection (Pre-Sentiment MLM Only):**
   - For the in-domain Financial PhraseBank experiments, **we never evaluate off-the-shelf `ProsusAI/finbert`**.
   - Instead, we initialize the domain-adapted Transformer from the **domain-pretrained FinBERT language model checkpoint prior to sentiment supervision** (pretrained solely via Masked Language Modeling on financial corpora—Araci's pre-sentiment MLM weights / `yiyanghkust/finbert-pretrain`) and compare it against general-domain **`bert-base-uncased`** [4].
   - Both encoders begin with freshly initialized 3-class linear classification heads (`nn.Linear(768, 3)`) and zero prior exposure to Financial PhraseBank labels.
2. **Hash-Locked Stratified Splits & Intra-Corpus Deduplication:**
   - Before any preprocessing or modeling, we partition Financial PhraseBank (`sentences_allagree` [2,264 sentences], `sentences_75agree` [3,453 sentences], and full `sentences_50agree` [4,845 sentences]) into a **70% train / 15% validation / 15% test stratified split**, supplemented by **5-fold stratified cross-validation**.
   - Near-duplicate sentences within Financial PhraseBank (normalized token-level Jaccard similarity $\ge 0.80$) are clustered into connected components and assigned to the same split partition so near-identical sentences never cross the train/test boundary.
   - Split assignments are committed as immutable SHA-256 sentence-hash manifests (`splits/fpb_splits_sha256.json`).
3. **Preprocessing & Hyperparameter Parity:**
   - For Tier 2 baselines, `TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True)` and regularization grid searches ($C \in \{0.01, 0.1, 1.0, 10.0\}$ for Logistic Regression and Linear SVM) are fit **strictly on the training fold** (`fit_transform(X_train)` only; validation selects $C$; test is evaluated exactly once).
   - For Tier 3 Transformers, both `bert-base-uncased` and the pre-sentiment FinBERT MLM encoder are fine-tuned on the exact same training split with matched optimization settings: AdamW ($\text{lr} \in \{2\times 10^{-5}, 3\times 10^{-5}\}$, weight decay $0.01$, 10% linear warmup), batch size 32, max sequence length 128, 4 epochs with early stopping on validation Macro-F1, and 3 random seeds ($\{42, 100, 2026\}$).

#### Stage 2 Details: External Out-of-Distribution Holdout on Deduplicated SEntFiN 1.0
1. **Independent Dataset Selection:**
   - We incorporate **SEntFiN 1.0** (Sinha et al., 2022 [5]), a publicly available dataset of 10,753 financial news headlines from *The Economic Times* annotated for entity-level positive, neutral, and negative sentiment.
2. **Cross-Dataset Deduplication Against Financial PhraseBank:**
   - We filter SEntFiN 1.0 to single-entity headlines or multi-entity headlines with unanimous sentiment polarity across entities, yielding >7,200 unambiguous 3-class sentences, from which a stratified holdout benchmark ($N = 2{,}000$) is drawn.
   - Every candidate headline is compared against **all 4,845 sentences of Financial PhraseBank** after lowercasing, punctuation stripping, and whitespace normalization. Any headline with exact match, substring containment, or normalized token-Jaccard similarity $\ge 0.80$ is purged.
3. **Zero-Shot Transfer & Off-the-Shelf Checkpoint Validation:**
   - All models trained in Stage 1—plus the off-the-shelf `ProsusAI/finbert` sentiment checkpoint—are evaluated zero-shot on this deduplicated SEntFiN 1.0 holdout. This allows us to (a) test out-of-distribution generalization beyond Financial PhraseBank and (b) benchmark `ProsusAI/finbert` on strictly unseen financial text.

---

## 4. Architecture Effect vs. Domain-Adaptation Effect Attribution (Addressing Rubric `Analysis` & `Method fits`)

A central requirement of the course rubric (`Analysis` and `Method fits`) is moving beyond single aggregate leaderboard numbers to explain *why* models succeed or fail. By pairing our 3-tier baseline hierarchy with the 5-category linguistic phenomenon taxonomy, our experimental design cleanly decomposes the total performance gain of FinBERT over traditional ML into two orthogonal components for every phenomenon category $k \in \{1, \dots, 5\}$:

$$\underbrace{\text{Macro-F1}(\text{FinBERT}_{\text{MLM}}, k) - \text{Macro-F1}(\text{TF-IDF+SVM}, k)}_{\text{Total Modern Domain-NLP Gain } (\Delta_{\text{total}, k})} = \underbrace{\big[\text{Macro-F1}(\text{BERT}_{\text{base}}, k) - \text{Macro-F1}(\text{TF-IDF+SVM}, k)\big]}_{\text{ Contextual Transformer Architecture Effect } (\Delta_{\text{arch}, k})} + \underbrace{\big[\text{Macro-F1}(\text{FinBERT}_{\text{MLM}}, k) - \text{Macro-F1}(\text{BERT}_{\text{base}}, k)\big]}_{\text{Financial Domain Pretraining Effect } (\Delta_{\text{domain}, k})}$$

And analogously at the lexicon tier:

$$\Delta_{\text{lex-domain}, k} = \text{Macro-F1}(\text{Loughran–McDonald}, k) - \text{Macro-F1}(\text{VADER}, k)$$

### Testable Phenomenon-Specific Hypotheses
1. **Finance-Specific Terminology** (e.g., *liability, cost, depreciation, debt, restructuring*): Expected to show the largest **Domain Pretraining Effect** ($\Delta_{\text{domain}}$) and **Lexicon Domain Effect** ($\Delta_{\text{lex-domain}}$), as general-domain embeddings and VADER assign negative prior polarity to descriptive accounting terms.
2. **Negation & Scope Inversion** (e.g., *not expected to decline*, *no material adverse impact*): Expected to show a strong **Contextual Architecture Effect** ($\Delta_{\text{arch}}$) over bag-of-words/n-gram TF-IDF baselines due to bidirectional self-attention over long-range negation scope.
3. **Hedging & Epistemic Modality** (e.g., *may, might, cautious outlook, contingent upon*): Expected to benefit from both contextual attention ($\Delta_{\text{arch}}$) and exposure to boilerplate forward-looking cautionary statements in financial filings ($\Delta_{\text{domain}}$).
4. **Numerical & Directional Comparisons** (e.g., *operating margin narrowed from 14.2% to 11.8%*, *revenue rose 4% below consensus*): Tests whether subword contextual representations capture directional prepositions (*up/down, rose/fell, beat/missed*) where static lexicons fail completely.
5. **Context-Dependent & Implicit Market Expressions** (e.g., *headwinds eased*, *guidance maintained despite macro pressure*): Evaluates compositional pragmatic reasoning across clause boundaries.

---

## 5. Summary of Verification Invariants for `proposal/proposal.tex`

To guarantee that the revised LaTeX proposal (`proposal/proposal.tex`) and compiled PDF (`proposal/proposal.pdf`) faithfully reflect this audit, the following automated checks are enforced:
1. **Zero Generation Metrics:** Case-insensitive search for `bleu`, `rouge`, or `meteor` in `proposal/proposal.tex` and extracted text of `proposal/proposal.pdf` returns **0 matches**.
2. **Explicit Classification & Statistical Suite:** `proposal/proposal.tex` explicitly specifies Accuracy, Macro-F1, per-class Precision/Recall/F1, $3 \times 3$ confusion matrices, McNemar's paired test, and paired bootstrap resampling ($B = 10{,}000$).
3. **Explicit Leakage-Free Protocol:** `proposal/proposal.tex` explicitly explains why off-the-shelf `ProsusAI/finbert` leaks Financial PhraseBank and specifies both (i) controlled fine-tuning of the pre-sentiment domain-adapted MLM encoder vs. `bert-base-uncased` on hash-locked 70/15/15 stratified splits and (ii) external holdout evaluation on deduplicated SEntFiN 1.0.
4. **Complete Rubric Coverage:** All 7 rubric criteria (`Scope` = 55 hours across 5 work packages, `Value`, `Baseline`, `Analysis`, `Data accessible`, `NLP+code`, `Method fits`) are explicitly addressed within the 2-page A4 limit. See `proposal/rubric_compliance_matrix.md` for the complete 8-row rubric compliance matrix and work-package allocation.
