"""Statistical hypothesis tests for comparing sentiment classifiers.

Implements:
- McNemar's test with continuity-corrected chi-squared and exact binomial p-value
- Paired bootstrap resampling for Delta Macro-F1 (95% CI and empirical p-value)
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
) -> dict[str, Any]:
    """Perform McNemar's paired test comparing predictions of Model A and Model B.

    Computes correctness indicators c_a = I(y_{pred,a} == y_{true}) and
    c_b = I(y_{pred,b} == y_{true}), discordant counts n_{10} (A correct, B wrong)
    and n_{01} (A wrong, B correct), continuity-corrected chi-squared statistic:
        chi2 = (|n_{10} - n_{01}| - 1)^2 / (n_{10} + n_{01})  (0.0 if n_{10} + n_{01} == 0),
    and exact two-sided binomial p-value under H0: p = 0.5.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth labels.
    y_pred_a : Sequence[int]
        Predictions from Model A.
    y_pred_b : Sequence[int]
        Predictions from Model B.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - "n10": int (Model A correct, Model B incorrect)
        - "n01": int (Model A incorrect, Model B correct)
        - "n11": int (Both correct)
        - "n00": int (Both incorrect)
        - "chi2": float (continuity-corrected chi-squared)
        - "statistic": float (alias for chi2)
        - "p_value": float (exact two-sided binomial p-value)
    """
    y_t = np.asarray(y_true)
    y_a = np.asarray(y_pred_a)
    y_b = np.asarray(y_pred_b)

    if not (len(y_t) == len(y_a) == len(y_b)):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)}, len(y_pred_a)={len(y_a)}, len(y_pred_b)={len(y_b)}"
        )

    if len(y_t) == 0:
        return {
            "n10": 0,
            "n01": 0,
            "n11": 0,
            "n00": 0,
            "chi2": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    corr_a = (y_a == y_t)
    corr_b = (y_b == y_t)

    n10 = int(np.sum(corr_a & (~corr_b)))
    n01 = int(np.sum((~corr_a) & corr_b))
    n11 = int(np.sum(corr_a & corr_b))
    n00 = int(np.sum((~corr_a) & (~corr_b)))

    total_discordant = n10 + n01

    if total_discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Edwards continuity-corrected chi-squared statistic
        diff = max(0.0, abs(n10 - n01) - 1.0)
        chi2 = float((diff ** 2) / total_discordant)

        # Exact two-sided binomial p-value under H0: p = 0.5
        x = min(n10, n01)
        if x * 2 == total_discordant:
            p_val = 1.0
        else:
            try:
                from scipy.stats import binomtest
                p_val = float(binomtest(x, n=total_discordant, p=0.5).pvalue)
            except (ImportError, AttributeError):
                cum_prob = sum(math.comb(total_discordant, i) for i in range(x + 1)) * (0.5 ** total_discordant)
                p_val = float(min(1.0, 2.0 * cum_prob))

    return {
        "n10": n10,
        "n01": n01,
        "n11": n11,
        "n00": n00,
        "chi2": chi2,
        "statistic": chi2,
        "p_value": p_val,
    }


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast Macro-F1 computation for 3-class predictions {0, 1, 2}."""
    encoded = y_true * 3 + y_pred
    counts = np.bincount(encoded, minlength=9).reshape(3, 3)
    tp = np.diag(counts)
    pp = counts.sum(axis=0)  # predicted positives
    ap = counts.sum(axis=1)  # actual positives
    denom = pp + ap
    f1s = np.zeros(3, dtype=float)
    mask = denom > 0
    f1s[mask] = 2.0 * tp[mask] / denom[mask]
    return float(np.mean(f1s))


def paired_bootstrap_ci(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    B: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap 95% confidence interval for Delta Macro-F1 and empirical p-value.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth integer labels {0, 1, 2}.
    y_pred_a : Sequence[int]
        Predictions from Model A.
    y_pred_b : Sequence[int]
        Predictions from Model B.
    B : int, default=10000
        Number of paired bootstrap resamples.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "delta_macro_f1": float (Macro-F1(A) - Macro-F1(B))
        - "ci_low": float (2.5th percentile of delta)
        - "ci_high": float (97.5th percentile of delta)
        - "ci_95": tuple[float, float]
        - "p_value": float (two-sided empirical p-value)
        - "b_samples": int (B)
        - "f1_a": float (observed Macro-F1 for A)
        - "f1_b": float (observed Macro-F1 for B)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_a = np.asarray(y_pred_a, dtype=int)
    y_b = np.asarray(y_pred_b, dtype=int)

    if not (len(y_t) == len(y_a) == len(y_b)):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)}, len(y_pred_a)={len(y_a)}, len(y_pred_b)={len(y_b)}"
        )

    n_samples = len(y_t)
    if n_samples == 0:
        return {
            "delta_macro_f1": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_95": (0.0, 0.0),
            "p_value": 1.0,
            "b_samples": B,
            "f1_a": 0.0,
            "f1_b": 0.0,
        }

    f1_a_obs = _fast_macro_f1(y_t, y_a)
    f1_b_obs = _fast_macro_f1(y_t, y_b)
    delta_obs = f1_a_obs - f1_b_obs

    # If predictions are identical, difference is identically 0
    if np.array_equal(y_a, y_b):
        return {
            "delta_macro_f1": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_95": (0.0, 0.0),
            "p_value": 1.0,
            "b_samples": B,
            "f1_a": float(f1_a_obs),
            "f1_b": float(f1_b_obs),
        }

    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n_samples, size=n_samples)
        f1_a_b = _fast_macro_f1(y_t[idx], y_a[idx])
        f1_b_b = _fast_macro_f1(y_t[idx], y_b[idx])
        deltas[b] = f1_a_b - f1_b_b

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    # Centered bootstrap hypothesis test (Efron & Tibshirani 1993)
    centered = deltas - delta_obs
    p_value = float(np.mean(np.abs(centered) >= np.abs(delta_obs)))
    if abs(delta_obs) < 1e-12:
        p_value = 1.0

    return {
        "delta_macro_f1": float(delta_obs),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_95": (float(ci_low), float(ci_high)),
        "p_value": float(p_value),
        "b_samples": int(B),
        "f1_a": float(f1_a_obs),
        "f1_b": float(f1_b_obs),
    }
