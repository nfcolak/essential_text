"""Baseline registry: model keys, display names and the factory used by the Step 2 runner."""

from __future__ import annotations

from typing import Any

BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "vader_official": {
        "display": "VADER (official lexicon)", "family": "lexicon", "variant": "official",
    },
    "vader_official_tuned": {
        "display": "VADER (official lexicon, val-tuned thresholds)", "family": "lexicon",
        "variant": "official_tuned",
    },
    "lm_official": {
        "display": "Loughran-McDonald (official dictionary)", "family": "lexicon", "variant": "official",
    },
    "lm_official_tuned": {
        "display": "Loughran-McDonald (official dictionary, negation + val-tuned thresholds)",
        "family": "lexicon", "variant": "official_tuned",
    },
    "vader": {
        "display": "VADER-style (finance-adjusted, custom)", "family": "lexicon", "variant": "finance_adjusted",
    },
    "loughran_mcdonald": {
        "display": "LM-style (finance-adjusted, custom)", "family": "lexicon", "variant": "finance_adjusted",
    },
    "tfidf_logreg": {"display": "TF-IDF + Logistic Regression", "family": "tfidf", "variant": None},
    "tfidf_svm": {"display": "TF-IDF + Linear SVM", "family": "tfidf", "variant": None},
}

DISPLAY_NAMES: dict[str, str] = dict(
    {key: spec["display"] for key, spec in BASELINE_SPECS.items()},
    bert_base="bert-base-uncased",
    finbert_mlm="FinBERT-MLM (finbert-pretrain)",
    prosus_finbert="ProsusAI/finbert (zero-shot)",
)

# Keys that predate the official lexicons: built through the Step 1 factory by its model name.
_STEP1_NAMES = {
    "vader": "VADER",
    "loughran_mcdonald": "Loughran-McDonald",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
}


def make_baseline(key: str, seed: int = 42) -> Any:
    """Return a fresh, unfitted baseline for a registry key."""
    if key == "vader_official":
        from src.baselines.lexicon_official import VaderOfficialBaseline

        return VaderOfficialBaseline()
    if key == "lm_official":
        from src.baselines.lexicon_official import LMOfficialBaseline

        return LMOfficialBaseline()
    if key == "vader_official_tuned":
        from src.baselines.lexicon_official import VaderOfficialTunedBaseline

        return VaderOfficialTunedBaseline()
    if key == "lm_official_tuned":
        from src.baselines.lexicon_official import LMOfficialTunedBaseline

        return LMOfficialTunedBaseline()
    if key in _STEP1_NAMES:
        from colab.run_step1_colab import instantiate_fresh_model

        return instantiate_fresh_model(_STEP1_NAMES[key], seed=seed)
    raise ValueError(f"Unknown baseline key: {key!r}; valid: {list(BASELINE_SPECS)}")
