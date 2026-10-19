"""Classification metrics, neutral/negative confusion, and expected calibration error.

Implements evaluation protocols strictly aligned with course requirements:
- Accuracy
- Macro-F1
- Per-class Precision / Recall / F1 / Support
- 3x3 Confusion Matrix
- neutral_to_negative_rate (RQ1 metric: false-negative misclassification of neutral financial sentences)
- 10-bin Expected Calibration Error (ECE)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABEL_TO_ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
CLASS_NAMES: list[str] = ["negative", "neutral", "positive"]


def compute_ece(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute 10-bin Expected Calibration Error (ECE) for multi-class classification.

    Confidence is defined as the maximum predicted class probability:
        conf_i = max_k P(Y = k | X_i)
    and the prediction is:
        pred_i = argmax_k P(Y = k | X_i).

    Equal-width confidence bins are formed over [0, 1]. For each non-empty bin B_m:
        acc(B_m) = (1 / |B_m|) sum_{i in B_m} I(pred_i == y_i)
        conf(B_m) = (1 / |B_m|) sum_{i in B_m} conf_i
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth integer labels in {0, 1, 2}.
    y_prob : np.ndarray
        Predicted class probability simplex of shape (N, C) where C >= 2.
    n_bins : int, default=10
        Number of equal-width confidence bins in [0, 1].

    Returns
    -------
    float
        Expected Calibration Error in [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0:
        return 0.0

    if probs.ndim != 2 or probs.shape[0] != len(y_t):
        raise ValueError(
            f"Shape mismatch: y_prob shape {probs.shape} does not match len(y_true)={len(y_t)}"
        )

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_t).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_t)

    for m in range(n_bins):
        low, high = bin_edges[m], bin_edges[m + 1]
        if m == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification metrics, RQ1 neutral-to-negative confusion, and ECE.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth integer labels (0=negative, 1=neutral, 2=positive).
    y_pred : Sequence[int]
        Predicted integer labels (0=negative, 1=neutral, 2=positive).
    y_prob : np.ndarray | None, default=None
        Predicted probability simplex of shape (N, 3). If provided, 10-bin ECE is computed.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "accuracy": float
        - "macro_f1": float
        - "per_class": dict mapping class names to {"precision", "recall", "f1", "support"}
        - "confusion_matrix": 3x3 nested list [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]]
        - "neutral_to_negative_rate": float (C_{1,0} / max(1, sum_j C_{1,j}))
        - "ece": float | None (10-bin Expected Calibration Error when y_prob is provided)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}"
        )
    if len(y_t) == 0:
        raise ValueError("Empty sequences provided to compute_classification_metrics.")

    labels = [0, 1, 2]

    # Overall Accuracy
    acc = float(accuracy_score(y_t, y_p))

    # Macro-F1 across 3 classes
    macro_f1 = float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0.0))

    # Per-class Precision, Recall, F1, Support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_t, y_p, labels=labels, zero_division=0.0
    )
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    # 3x3 Confusion Matrix
    cm_arr = confusion_matrix(y_t, y_p, labels=labels)
    cm_list = [[int(val) for val in row] for row in cm_arr]

    # Neutral-to-Negative misclassification rate for RQ1
    # Row 1 is true neutral; col 0 is predicted negative.
    c10 = cm_list[1][0]
    total_neutrals = sum(cm_list[1])
    neutral_to_negative_rate = float(c10 / max(1, total_neutrals))

    # Expected Calibration Error (if probabilities supplied)
    ece_val = None
    if y_prob is not None:
        ece_val = compute_ece(y_t, y_prob, n_bins=10)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm_list,
        "neutral_to_negative_rate": neutral_to_negative_rate,
        "ece": ece_val,
    }
