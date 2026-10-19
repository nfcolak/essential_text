# AGENTS.md — essential_text

## Project Overview
Course project repository for **Evaluating Domain Adaptation in Financial Sentiment Analysis: FinBERT vs. Traditional NLP Baselines** (Authors: Qiyao Zheng, Benjamin Donick, Necati Furkan Colak).

## Non-Negotiable Rules
1. **No Data Leakage**: Off-the-shelf `ProsusAI/finbert` was fine-tuned on Financial PhraseBank. Any evaluation on Financial PhraseBank MUST either fine-tune the domain-pretrained MLM checkpoint (`yiyanghkust/finbert-pretrain` / further-pretrained FinBERT prior to sentiment supervision) and `bert-base-uncased` on identical stratified train/val/test splits, or evaluate off-the-shelf checkpoints only on strictly held-out / external datasets (e.g., SEntFiN 1.0 / FiQA) after deduplication.
2. **Classification Metrics Only**: Never use text-generation metrics (`BLEU`, `ROUGE`, `METEOR`) for sentiment classification. Use accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, McNemar's test, and paired bootstrap resampling.
3. **Rubric Alignment**: All proposal documents and experiment plans must explicitly satisfy the 7 course rubric criteria:
   - **Scope**: ~55 hours total team effort with explicit work-package breakdown.
   - **Value**: Clear reusable outputs for others (leakage-free split/protocol, phenomenon-tagged diagnostic dataset, reproducible pipeline).
   - **Baseline**: 3 levels (Lexicon: VADER & Loughran–McDonald; Traditional ML: TF-IDF + Logistic Regression & SVM; General Transformer: `bert-base-uncased`).
   - **Analysis**: Beyond single scores (class-specific neutral/negative confusion, 5 linguistic phenomena error breakdown, architecture vs. domain-pretraining attribution, statistical tests).
   - **Data accessible**: Publicly available datasets named with exact sizes and immediate access paths.
   - **NLP+code**: Explicit Python/PyTorch/HuggingFace/scikit-learn implementation components.
   - **Method fits**: Direct mapping between methods and RQ1, RQ2, RQ3.
4. **Shared Agent Instruction File**: Keep instructions in `AGENTS.md` only (never create `CLAUDE.md` or `GEMINI.md`).
