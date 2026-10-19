# Changelog and Instructor Response: Revised Course Project Proposal

**Project Title:** Evaluating Domain Adaptation in Financial Sentiment Analysis: FinBERT vs. Traditional NLP Baselines  
**Authors:** Qiyao Zheng, Benjamin Donick, Necati Furkan Colak  
**Document Revision:** `v1` (`proposal/proposal_v1_original.pdf`) $\longrightarrow$ `v2` (`proposal/proposal.pdf` compiled from `proposal/proposal.tex`)  

---

## 1. Ready-to-Send Reply Message to Course Instructor and TAs

*You can copy and send the text below directly to the course instructors / TAs via email or the course portal.*

***

**Subject:** Revised Project Proposal: Evaluating Domain Adaptation in Financial Sentiment Analysis (Zheng, Donick, Colak)

Dear Teaching Team,

Thank you for the detailed feedback on our project proposal, *Evaluating Domain Adaptation in Financial Sentiment Analysis: FinBERT vs. Traditional NLP Baselines*. We have revised the proposal to directly resolve both methodological critiques and ensure strict compliance with all seven dimensions of the course rubric. 

Below, we directly answer both questions and outline how the revised 2-page proposal (`proposal.pdf`) resolves them:

### 1. Removal and Replacement of Text-Generation Metrics (`BLEU`, `ROUGE`, `METEOR`)
* **Why they appeared in v1:** Their inclusion in `v1` was an oversight during initial drafting that conflated generation overlap metrics with categorical classification metrics.
* **Why they are invalid:** Our task maps input sentences to discrete unordered sentiment labels $\mathcal{Y} \in \{\text{positive}, \text{neutral}, \text{negative}\}$ via a softmax probability vector $\hat{\mathbf{p}} \in \Delta^2$. None of our models generate text tokens; applying $n$-gram precision/recall or brevity penalties to single-word class labels is mathematically degenerate and ignores class imbalance.
* **What replaces them in v2:** `BLEU`, `ROUGE`, and `METEOR` have been completely purged from the proposal (0 occurrences across `.tex` and `.pdf`). In their place, we evaluate using:
  1. **Macro-F1** (primary headline metric) to account for severe class imbalance in Financial PhraseBank (~59% neutral, ~28% positive, ~13% negative).
  2. **Accuracy** for direct comparability with foundational benchmarks.
  3. **Per-class Precision, Recall, and F1-score**, with diagnostic focus on the **neutral vs. negative boundary** where general-domain models falsely classify descriptive accounting terminology as negative (RQ1).
  4. **$3 \times 3$ Confusion Matrices** and **Expected Calibration Error (ECE)**.
  5. **Paired McNemar's Test** (with continuity correction) and **paired bootstrap resampling** ($B = 10{,}000$ resamples) to establish 95% confidence intervals and empirical $p$-values on $\Delta\text{Macro-F1}$.

### 2. Resolution of FinBERT Data Leakage on Financial PhraseBank
* **The Root Cause:** Off-the-shelf `ProsusAI/finbert` (Araci, 2019) was fine-tuned directly on Financial PhraseBank. Evaluating that checkpoint on arbitrary PhraseBank splits leaks labeled training instances into the test set and confounds domain adaptation with differences in training hyperparameters.
* **Our Two-Stage Leakage-Free Protocol:**
  1. **Stage 1 (Controlled In-Domain Fine-Tuning):** On Financial PhraseBank, rather than evaluating off-the-shelf `ProsusAI/finbert`, we initialize from the **pre-sentiment domain-pretrained FinBERT masked language model** (`yiyanghkust/finbert-pretrain` / Araci's MLM weights prior to sentiment supervision) and compare it against general-domain `bert-base-uncased`. Both Transformer encoders and our TF-IDF baselines are trained, validated, and tested on the **exact same stratified 70%/15%/15% split** (and 5-fold stratified cross-validation). Holdout test splits are locked via SHA-256 sentence hashes, and near-duplicate sentences (token Jaccard $\ge 0.80$) are clustered into the same partition before any modeling. Because neither model has seen PhraseBank sentiment labels prior to our training split, test leakage is eliminated and domain pretraining is cleanly isolated from task supervision.
  2. **Stage 2 (External Out-of-Distribution Holdout):** To evaluate off-the-shelf `ProsusAI/finbert` without training-data contamination, we benchmark all models zero-shot on an independent public dataset—**SEntFiN 1.0** (10,753 financial news headlines)—filtered to unambiguous 3-class sentences and deduplicated against all 4,845 PhraseBank sentences (token Jaccard $< 0.80$).

### 3. Summary of Course Rubric Enhancements
* **Scope (~55 hours total):** Explicit breakdown of **55.0 hours** across 5 Work Packages (~18.3 hours per member across Qiyao Zheng, Benjamin Donick, and Necati Furkan Colak).
* **Value:** Three reusable community artifacts released on GitHub: (i) SHA-256 locked split manifests and PhraseBank–SEntFiN deduplication scripts, (ii) a 300-sentence phenomenon-annotated diagnostic dataset, and (iii) a modular Python evaluation codebase.
* **Baselines (3 tiers):** Lexicon (VADER vs. Loughran–McDonald), Traditional ML (TF-IDF + Logistic Regression / Linear SVM), and Pretrained Transformers (`bert-base-uncased` vs. pre-sentiment FinBERT MLM + SEntFiN 1.0 holdout).
* **Analysis:** Dual-annotated 300-sentence diagnostic breakdown across 5 linguistic phenomena (reporting inter-annotator Cohen's $\kappa$), plus mathematical attribution decomposing performance gains into **Architecture Effect** ($\text{BERT}_{\text{base}} - \text{TF-IDF+SVM}$) versus **Domain Pretraining Effect** ($\text{FinBERT}_{\text{MLM}} - \text{BERT}_{\text{base}}$).
* **Data Accessible:** Explicit sizes and access paths for Financial PhraseBank (`allagree`, `75agree`, full corpus), Loughran–McDonald Master Dictionary, and SEntFiN 1.0.
* **NLP + Code:** PyTorch, Hugging Face `transformers`, and `scikit-learn` pipeline controls (`sublinear_tf=True`, fit on train folds only, locked random seeds $\{42, 100, 2026\}$).
* **Method Fits:** Direct 1-to-1 mapping between RQ1–RQ3 and the baseline tiers, phenomenon slices, and statistical significance tests.

The revised 2-page proposal has been compiled with Tectonic (`proposal.pdf`) and verified by an automated invariant suite (`proposal/verify_proposal.py`). We look forward to your approval to proceed!

Best regards,  
Qiyao Zheng, Benjamin Donick, Necati Furkan Colak

***

---

## 2. Section-by-Section Before/After Diff Summary

| Section | Original Proposal (`v1`) | Revised Proposal (`v2`) | Rubric / Feedback Addressed |
| :--- | :--- | :--- | :--- |
| **Title & Authors** | Identical title & authors, standard draft formatting. | Preserves identical title and 3 authors (*Qiyao Zheng, Benjamin Donick, Necati Furkan Colak*); formatted in clean A4 Times (`mathptmx`, 2cm margins). | Document layout parity; fits strictly within 2-page limit. |
| **Section 1: Motivation & Research Questions** | • Broad description of domain mismatch.<br>• RQ1: General misclassification.<br>• RQ2: Domain adaptation vs. baselines.<br>• RQ3: Linguistic phenomena causing differences. | • Sharpens motivation with concrete accounting examples (*cost, liability, debt, depreciation*) causing false-negative neutral classifications.<br>• **RQ1 (Domain Polarity Mismatch):** Quantifies false assignment of negative polarity to neutral descriptive statements.<br>• **RQ2 (Controlled Architecture vs. Domain Adaptation):** Specifies evaluation under a strictly leakage-free protocol.<br>• **RQ3 (Linguistic Phenomenon Attribution):** Names the 5 phenomena and establishes architecture vs. domain-pretraining attribution. | `Method fits`, `NLP+code`, `Analysis` |
| **Section 2: Data and Experimental Setup** | • Named Financial PhraseBank (~4,845 sentences) generally.<br>• Mentioned 3 tiers conceptually without feature hygiene or library details.<br>• **Flaw 1:** Listed `BLEU`, `ROUGE`, and `METEOR` alongside classification metrics.<br>• **Flaw 2:** Noted "avoiding data leakage" as a challenge without specifying any protocol to handle `ProsusAI/finbert` PhraseBank contamination. | • **Accessible Datasets:** Details Financial PhraseBank [1] consensus subsets (`sentences_allagree` $N=2{,}264$, `sentences_75agree` $N=3{,}453$, full $N=4{,}845$), class imbalance (~59% neutral, ~28% pos, ~13% neg), Loughran–McDonald [2], and external holdout SEntFiN 1.0 [5] ($N=10{,}753$).<br>• **Three-Tier Baselines:** Tier 1 VADER [6] vs. Loughran–McDonald [2]; Tier 2 TF-IDF 1–3-grams (`sublinear_tf=True`, fit on train folds only) + Logistic Regression & Linear SVM; Tier 3 `bert-base-uncased` [4] vs. pre-sentiment FinBERT MLM [3].<br>• **Leakage-Free Protocol:** Explains why `ProsusAI/finbert` leaks PhraseBank; specifies (1) controlled fine-tuning from pre-sentiment domain MLM checkpoint on hash-locked 70/15/15 splits + 5-fold CV, and (2) external OOD holdout on deduplicated SEntFiN 1.0 (token Jaccard $< 0.8$).<br>• **Classification Metrics:** Removes `BLEU`/`ROUGE`/`METEOR`; replaces with Macro-F1 (primary), Accuracy, per-class P/R/F1, $3 \times 3$ confusion matrices, ECE, paired McNemar's test, and paired bootstrap resampling ($B = 10{,}000$). | `Baseline`, `Data accessible`, `NLP+code`, `Method fits`, `Analysis` |
| **Section 3: Error Analysis and Expected Contribution** | • Vaguely stated that misclassified sentences would be grouped into 5 categories.<br>• Generic mention of GitHub repo and documented error categories. | • **Phenomenon-Stratified Analysis (RQ3):** Defines a 300-sentence stratified diagnostic subset with rule-assisted pre-tagging, dual manual annotation, and inter-annotator Cohen's $\kappa$.<br>• **Mathematical Decomposition:** Formulates explicit attribution into **Architecture Effect** ($\Delta_{\text{arch}}(c) = \text{F1}_{\text{BERT-base}}(c) - \text{F1}_{\text{TF-IDF+SVM}}(c)$) vs. **Domain Adaptation Effect** ($\Delta_{\text{domain}}(c) = \text{F1}_{\text{FinBERT-MLM}}(c) - \text{F1}_{\text{BERT-base}}(c)$).<br>• **Reusable Value Produced:** Guarantees 3 public deliverables on GitHub: (i) locked split manifests & deduplication scripts, (ii) 300-sentence diagnostic dataset with guidelines, and (iii) modular Python evaluation pipeline. | `Value`, `Analysis`, `NLP+code` |
| **Section 4: Work Plan and Feasibility** | • Vague 2-sentence timeline without hour estimates, work packages, or team roles. | • **Explicit 55-hour Team Budget** (~18.3h per member across QZ, BD, NFC) structured into **5 Work Packages** with designated member leads:<br>&nbsp;&nbsp;– WP1: Data & Leakage-Free Splits (8h; Lead: Colak)<br>&nbsp;&nbsp;– WP2: Lexicon & TF-IDF Baselines (10h; Lead: Donick)<br>&nbsp;&nbsp;– WP3: BERT vs. FinBERT Fine-Tuning (14h; Lead: Zheng)<br>&nbsp;&nbsp;– WP4: Phenomenon Annotation & Stats (13h; All)<br>&nbsp;&nbsp;– WP5: Reproducible Repo & Report (10h; All). | `Scope` |
| **Section 5: References** | 3 references: Malo et al. [1], Loughran & McDonald [2], Araci [3]. | **Expanded to 6 complete citations:**<br>[1] Malo et al. (2014) — Financial PhraseBank<br>[2] Loughran & McDonald (2011) — Financial Lexicon<br>[3] Araci (2019) — FinBERT<br>[4] Devlin et al. (2019) — BERT (`NAACL-HLT`)<br>[5] Sinha et al. (2022) — SEntFiN 1.0 (`JAIST`)<br>[6] Hutto & Gilbert (2014) — VADER (`ICWSM`). | Citation completeness & academic grounding |

---

## 3. Course Rubric 7-Dimension Compliance Verification

| Dimension | Rubric Requirement | How `v2` Fully Satisfies It | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **1. Scope** | Workload is realistically planned for ~55 hours total team effort. | 55.0h total budget broken into WP1 (8h), WP2 (10h), WP3 (14h), WP4 (13h), and WP5 (10h) across QZ (18.5h), BD (18.5h), NFC (18.0h). | Section 4 of `proposal.tex` & `proposal.pdf` |
| **2. Value** | Produces reusable artifacts for other researchers / students. | Releases SHA-256 split manifests, deduplication scripts, the 300-sentence dual-annotated diagnostic dataset, and modular GitHub pipeline. | Section 3 of `proposal.tex` & `proposal.pdf` |
| **3. Baseline** | Multi-level baseline hierarchy (rules, traditional ML, general neural). | Tier 1 (VADER vs. Loughran–McDonald), Tier 2 (TF-IDF + LogReg / Linear SVM), Tier 3 (`bert-base-uncased` vs. pre-sentiment FinBERT MLM + SEntFiN 1.0 OOD). | Section 2 of `proposal.tex` & `proposal.pdf` |
| **4. Analysis** | Deep diagnostic evaluation beyond single aggregate scores. | 5-phenomenon error taxonomy on 300 sentences with Cohen's $\kappa$; $3 \times 3$ confusion matrices; $\Delta_{\text{arch}}$ vs. $\Delta_{\text{domain}}$ attribution; McNemar's test & paired bootstrap CIs. | Section 2 & 3 of `proposal.tex` & `proposal.pdf` |
| **5. Data Accessible** | Datasets publicly available, explicitly named, with immediate access paths. | Financial PhraseBank (Hugging Face `financial_phrasebank`), Loughran–McDonald dictionary, and SEntFiN 1.0 external holdout explicitly specified with sizes. | Section 2 of `proposal.tex` & `proposal.pdf` |
| **6. NLP + Code** | Concrete NLP methods and modern reproducible implementation stack. | Python, PyTorch, Hugging Face `transformers`, `scikit-learn`, `scipy`/`statsmodels`, locked random seeds, and SHA-256 split integrity. | Sections 2, 3, 4 of `proposal.tex` & `proposal.pdf` |
| **7. Method Fits** | Direct logical connection between research questions and methodology. | RQ1 $\to$ Lexicon & neutral/negative confusion; RQ2 $\to$ Controlled BERT vs. FinBERT + SEntFiN holdout; RQ3 $\to$ Linguistic phenomena & attribution formula. | Sections 1, 2, 3 of `proposal.tex` & `proposal.pdf` |

---

## 4. Automated Programmatic Verification Suite (`proposal/verify_proposal.py`)

The automated verification suite was executed via:
```bash
env -u PYTHONPATH /usr/bin/python3 proposal/verify_proposal.py
```
**Results:** `50 PASSED, 0 FAILED`
* **Zero generation metrics:** 0 occurrences of `bleu`, `rouge`, or `meteor` across both `proposal.tex` and `proposal.pdf`.
* **Exact page count:** `proposal.pdf` is exactly **2 pages**.
* **Protocol & Metric Terms:** Verified presence of `leakage-free`, `ProsusAI/finbert`, `yiyanghkust/finbert-pretrain`, `SEntFiN 1.0`, `70%/15%/15%`, `SHA-256`, `Jaccard`, `Macro-F1`, `Accuracy`, `Precision`, `Recall`, `Confusion Matrices`, `McNemar`, and `bootstrap` ($B = 10{,}000$).
* **Baselines & Work Plan:** Verified presence of `VADER`, `Loughran`, `TF-IDF`, `Logistic Regression`, `Linear`, `BERT`, `FinBERT`, `architecture effect`, `domain-adaptation effect`, `Cohen`, `55-hour`, all 5 Work Packages (`WP1`--`WP5`), and all 3 authors (`Zheng`, `Donick`, `Colak`).
* **References:** Verified complete bibliographic citations for all 6 reference entries (`Malo`, `Loughran`, `Araci`, `Devlin`, `Sinha`, `Hutto`).
