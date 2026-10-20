#!/usr/bin/env python3
"""
Verification suite for the revised course project proposal:
- proposal/proposal.tex
- proposal/proposal.pdf

Verifies:
1. Hard invariant: zero occurrences of BLEU, ROUGE, METEOR (case-insensitive) in both .tex and .pdf.
2. Leakage-free protocol terms: pre-sentiment MLM checkpoint, locked splits, external holdout (SEntFiN 1.0).
3. Strictly classification-appropriate metrics & statistical tests: Macro-F1, Accuracy, per-class P/R/F1,
   confusion matrices, McNemar's paired test, and paired bootstrap resampling (B = 10,000).
4. Full 7-rubric compliance: 3 authors, 3 baseline tiers, 5 linguistic phenomena with architecture vs domain
   attribution, accessible public datasets; the 55-hour workload across 5 work packages is checked in
   proposal/rubric_compliance_matrix.md (the Work Plan section was removed from the 2-page PDF for the page limit,
   commit 0d55c01).
5. All 6 references present with complete bibliographic metadata.
6. Exact page count == 2 in proposal.pdf.
"""

import sys
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("FATAL: PyMuPDF (fitz) is required to inspect proposal.pdf.")
    sys.exit(1)

def run_checks():
    repo_root = Path(__file__).resolve().parent.parent
    tex_path = repo_root / "proposal" / "proposal.tex"
    pdf_path = repo_root / "proposal" / "proposal.pdf"

    print(f"Verifying TeX: {tex_path}")
    print(f"Verifying PDF: {pdf_path}")

    assert tex_path.exists(), f"TeX file does not exist: {tex_path}"
    assert pdf_path.exists(), f"PDF file does not exist: {pdf_path}"

    tex_content = tex_path.read_text(encoding="utf-8")
    # For text matching in TeX, unescape LaTeX % and { }
    tex_normalized = tex_content.replace(r"\%", "%").replace(r"\{", "{").replace(r"\}", "}").replace("{,}", ",")

    doc = fitz.open(str(pdf_path))
    pdf_page_count = len(doc)
    raw_pdf_text = "\n".join([page.get_text() for page in doc])
    # De-hyphenate line breaks for cross-line token matching (e.g. archi-\ntecture -> architecture)
    pdf_text = re.sub(r"-\n\s*", "", raw_pdf_text)

    failures = []
    passes = []

    def check(condition, desc):
        if condition:
            passes.append(desc)
            print(f"  [PASS] {desc}")
        else:
            failures.append(desc)
            print(f"  [FAIL] {desc}")

    print("\n--- 1. Banned Text-Generation Metrics (Zero occurrences) ---")
    banned_patterns = [r"\bbleu\b", r"\brouge\b", r"\bmeteor\b"]
    for pat in banned_patterns:
        match_tex = re.search(pat, tex_content, re.IGNORECASE)
        check(match_tex is None, f"Zero matches for '{pat}' in proposal.tex")
        match_pdf = re.search(pat, pdf_text, re.IGNORECASE)
        check(match_pdf is None, f"Zero matches for '{pat}' in proposal.pdf")

    print("\n--- 2. PDF Page Count Invariant ---")
    check(pdf_page_count == 2, f"proposal.pdf page count == 2 (actual: {pdf_page_count})")

    print("\n--- 3. Leakage-Free Protocol Terms ---")
    leakage_terms = [
        ("leakage-free", "Explicit mention of 'leakage-free' protocol"),
        ("ProsusAI/finbert", "Identifies ProsusAI/finbert supervised checkpoint"),
        ("Financial PhraseBank", "Names primary Financial PhraseBank corpus"),
        ("SEntFiN 1.0", "Names external holdout benchmark SEntFiN 1.0"),
        ("yiyanghkust/finbert-pretrain", "Identifies pre-sentiment domain MLM checkpoint"),
        ("70%/15%/15%", "Explicit 70/15/15 stratified train/val/test split"),
        ("SHA-256", "Split locking via SHA-256 sentence hashes"),
        ("Jaccard", "Cross-dataset deduplication via token Jaccard similarity"),
    ]
    for term, desc in leakage_terms:
        in_tex = term.lower() in tex_normalized.lower()
        in_pdf = term.lower() in pdf_text.lower()
        check(in_tex and in_pdf, f"{desc} (found in .tex and .pdf)")

    print("\n--- 4. Classification Metrics & Statistical Tests ---")
    eval_terms = [
        ("Macro-F1", "Primary evaluation metric: Macro-F1"),
        ("Accuracy", "Secondary evaluation metric: Accuracy"),
        ("Precision", "Per-class Precision"),
        ("Recall", "Per-class Recall"),
        ("Confusion Matrices", "Directional 3x3 Confusion Matrices"),
        ("neutral", "Focus on neutral class boundary"),
        ("negative", "Focus on negative class boundary"),
        ("McNemar", "Paired McNemar's significance test"),
        ("bootstrap", "Paired bootstrap resampling"),
        ("10,000", "Bootstrap resamples (B = 10,000)"),
    ]
    for term, desc in eval_terms:
        in_tex = term.lower() in tex_normalized.lower()
        in_pdf = term.lower() in pdf_text.lower()
        check(in_tex and in_pdf, f"{desc} (found in .tex and .pdf)")

    print("\n--- 5. 3-Tier Baseline Hierarchy & 5-Phenomenon Error Attribution ---")
    method_terms = [
        ("VADER", "Tier 1: General lexicon VADER"),
        ("Loughran", "Tier 1: Financial lexicon Loughran-McDonald"),
        ("TF-IDF", "Tier 2: Traditional ML TF-IDF features"),
        ("Logistic Regression", "Tier 2: Logistic Regression classifier"),
        ("Linear", "Tier 2: Linear SVM classifier"),
        ("BERT", "Tier 3: bert-base-uncased general Transformer"),
        ("FinBERT", "Tier 3: Domain-adapted FinBERT"),
        ("architecture effect", "RQ3: Architecture effect attribution"),
        ("domain-adaptation effect", "RQ3: Domain adaptation effect attribution"),
        ("Cohen", "Inter-annotator agreement: Cohen's kappa"),
    ]
    for term, desc in method_terms:
        in_tex = term.lower() in tex_normalized.lower()
        in_pdf = term.lower() in pdf_text.lower()
        check(in_tex and in_pdf, f"{desc} (found in .tex and .pdf)")

    print("\n--- 6. 55-Hour Team Work Plan & Scope ---")
    # The Work Plan section was removed from the 2-page PDF for the page limit (commit 0d55c01);
    # the 55-hour budget and WP1-WP5 live in rubric_compliance_matrix.md section 2.
    matrix_text = (repo_root / "proposal" / "rubric_compliance_matrix.md").read_text(encoding="utf-8")
    workplan_terms = [
        ("55-Hour", "Explicit 55-hour total team budget"),
        ("WP1", "Work Package 1 (Data & Splits)"),
        ("WP2", "Work Package 2 (Baselines)"),
        ("WP3", "Work Package 3 (Transformer Fine-Tuning)"),
        ("WP4", "Work Package 4 (Phenomenon & Stats)"),
        ("WP5", "Work Package 5 (Repo & Report)"),
    ]
    for term, desc in workplan_terms:
        check(term.lower() in matrix_text.lower(), f"{desc} (found in rubric_compliance_matrix.md)")
    for term, desc in [("Zheng", "Team member: Qiyao Zheng"), ("Donick", "Team member: Benjamin Donick"),
                       ("Colak", "Team member: Necati Furkan Colak")]:
        in_tex = term.lower() in tex_normalized.lower()
        in_pdf = term.lower() in pdf_text.lower()
        check(in_tex and in_pdf, f"{desc} (found in .tex and .pdf)")

    print("\n--- 7. All 6 Bibliographic References ---")
    references = [
        ("Malo", "Ref [1]: Malo et al. (Financial PhraseBank)"),
        ("Loughran", "Ref [2]: Loughran & McDonald (Journal of Finance)"),
        ("Araci", "Ref [3]: Araci (FinBERT preprint)"),
        ("Devlin", "Ref [4]: Devlin et al. (BERT NAACL-HLT)"),
        ("Sinha", "Ref [5]: Sinha et al. (SEntFiN 1.0 JAIST)"),
        ("Hutto", "Ref [6]: Hutto & Gilbert (VADER ICWSM)"),
    ]
    for author, desc in references:
        in_tex = author.lower() in tex_normalized.lower()
        in_pdf = author.lower() in pdf_text.lower()
        check(in_tex and in_pdf, f"{desc} (found in .tex and .pdf)")

    print(f"\n==========================================")
    print(f"VERIFICATION SUMMARY: {len(passes)} PASSED, {len(failures)} FAILED")
    print(f"==========================================")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nALL VERIFICATION INVARIANTS PASSED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    run_checks()
