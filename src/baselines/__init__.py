"""Baselines module for financial sentiment analysis.

Exports:
- VaderBaseline: Tier 1 general/financial lexicon with accounting term neutralization
- LoughranMcDonaldBaseline: Tier 1 financial domain lexicon with 3-token negation
- TfidfMLBaseline: Tier 2 TF-IDF + Logistic Regression / Linear SVM with calibration
"""

from src.baselines.lexicon import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NEUTRAL_ACCOUNTING_TERMS,
    LoughranMcDonaldBaseline,
    VaderBaseline,
)
from src.baselines.tfidf_ml import TfidfMLBaseline

__all__ = [
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "NEUTRAL_ACCOUNTING_TERMS",
    "VaderBaseline",
    "LoughranMcDonaldBaseline",
    "TfidfMLBaseline",
]
