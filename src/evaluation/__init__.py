"""Evaluation metrics and statistical testing module."""

from src.evaluation.metrics import (
    CLASS_NAMES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    compute_classification_metrics,
    compute_ece,
)
from src.evaluation.stats import mcnemar_test, paired_bootstrap_ci

__all__ = [
    "compute_classification_metrics",
    "compute_ece",
    "mcnemar_test",
    "paired_bootstrap_ci",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "CLASS_NAMES",
]
