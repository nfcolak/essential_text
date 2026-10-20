"""Multi-seed aggregation helpers for the Step 2 run matrix.

Provides:
- ``load_runs``: the only reader of ``<runs_dir>/<config>/<model_key>/<tag>/`` run directories
- ``mean_sd``: seed mean and sample standard deviation
- ``holm_bonferroni``: step-down Holm adjustment of a family of p-values
- ``seed_averaged_bootstrap``: paired sentence bootstrap of a seed-averaged metric difference
- ``seed_sentence_bootstrap``: two-level (sentences and seeds) bootstrap of the same difference
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np

_SPLITS = ("val", "test", "external")
_METRICS = ("macro_f1", "accuracy", "neutral_recall", "negative_precision")


def load_runs(runs_dir: str) -> list[dict]:
    """Load every completed run below ``runs_dir``.

    The layout is ``<runs_dir>/<config>/<model_key>/<tag>/{result.json, predictions.npz}``.
    Directories whose ``result.json`` is absent or not ``"status": "completed"`` (including
    ``<tag>.tmp`` leftovers) are ignored. Each returned dict is the ``result.json`` content plus
    ``"run_dir"``, ``"config"``, ``"model_key"``, ``"tag"`` (taken from the path when the file
    lacks them) and ``"predictions"``: ``{split: {"ids", "y", "prob"}}``.
    """
    runs: list[dict] = []
    if not os.path.isdir(runs_dir):
        return runs
    for config in sorted(os.listdir(runs_dir)):
        config_dir = os.path.join(runs_dir, config)
        if not os.path.isdir(config_dir):
            continue
        for model_key in sorted(os.listdir(config_dir)):
            model_dir = os.path.join(config_dir, model_key)
            if not os.path.isdir(model_dir):
                continue
            for tag in sorted(os.listdir(model_dir)):
                run_dir = os.path.join(model_dir, tag)
                result_path = os.path.join(run_dir, "result.json")
                if tag.endswith(".tmp") or not os.path.isfile(result_path):
                    continue
                with open(result_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
                if result.get("status") != "completed":
                    continue
                result["run_dir"] = run_dir
                result.setdefault("config", config)
                result.setdefault("model_key", model_key)
                result["tag"] = tag
                result["predictions"] = _load_predictions(os.path.join(run_dir, "predictions.npz"))
                runs.append(result)
    return runs


def _load_predictions(path: str) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {}
    if not os.path.isfile(path):
        return predictions
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        for split in _SPLITS:
            if f"{split}_prob" in names:
                predictions[split] = {
                    "ids": np.asarray(data[f"{split}_ids"]).astype(str),
                    "y": np.asarray(data[f"{split}_y"], dtype=np.int64),
                    "prob": np.asarray(data[f"{split}_prob"], dtype=np.float32),
                }
    return predictions


def mean_sd(values: Sequence[float]) -> dict[str, Any]:
    """Mean and sample standard deviation (ddof=1, 0.0 for a single value) of ``values``."""
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0}
    arr = np.asarray(clean, dtype=float)
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return {"mean": float(np.mean(arr)), "sd": sd, "n": n}


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values: monotone non-decreasing in rank, capped at 1.0."""
    m = len(pvalues)
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, (m - rank) * float(p))
        adjusted[key] = float(min(1.0, running))
    return {key: adjusted[key] for key in pvalues}


def _cell_onehot(y_true: np.ndarray, preds: Sequence[np.ndarray]) -> np.ndarray:
    """(n, 9 * len(preds)) one-hot of the confusion cell ``y * 3 + pred`` of every prediction vector."""
    n = len(y_true)
    onehot = np.zeros((n, 9 * len(preds)), dtype=np.float64)
    rows = np.arange(n)
    for j, pred in enumerate(preds):
        onehot[rows, 9 * j + y_true * 3 + pred] = 1.0
    return onehot


def _metric_from_counts(counts: np.ndarray, metric: str) -> np.ndarray:
    """Metric per row of ``counts`` (rows x 9, row-major 3x3 confusion matrix: true x predicted).

    ``neutral_recall`` = cm[1,1] / true-neutral row sum, ``negative_precision`` = cm[0,0] / predicted-negative
    column sum; both 0 when the denominator is 0.
    """
    cm = counts.reshape(-1, 3, 3)
    tp = np.einsum("bii->bi", cm)
    if metric == "accuracy":
        return tp.sum(axis=1) / np.maximum(cm.sum(axis=(1, 2)), 1.0)
    if metric == "neutral_recall":
        row = cm[:, 1, :].sum(axis=1)
        return np.where(row > 0, cm[:, 1, 1] / np.maximum(row, 1.0), 0.0)
    if metric == "negative_precision":
        col = cm[:, :, 0].sum(axis=1)
        return np.where(col > 0, cm[:, 0, 0] / np.maximum(col, 1.0), 0.0)
    pp = cm.sum(axis=1)
    ap = cm.sum(axis=2)
    denom = pp + ap
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return f1.mean(axis=1)


def seed_averaged_bootstrap(
    y_true: Sequence[int],
    preds_a: Sequence[Sequence[int]],
    preds_b: Sequence[Sequence[int]],
    metric: str = "macro_f1",
    B: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired sentence bootstrap of ``mean_s metric(A_s) - mean_s metric(B_s)``.

    ``preds_a`` / ``preds_b`` are lists with one prediction vector per seed (a single-seed or
    deterministic model passes a list of one). Sentences are resampled with replacement with
    ``np.random.default_rng(seed)``; every resample is scored for every seed of both models.

    Returns ``{"delta", "ci_low", "ci_high", "p_value", "B", "n"}`` (95% percentile CI; two-sided
    p = 2 * the smaller tail share of resampled deltas at or beyond zero, capped at 1).
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    y = np.asarray(y_true, dtype=np.int64)
    n = len(y)
    a_list = [np.asarray(p, dtype=np.int64) for p in preds_a]
    b_list = [np.asarray(p, dtype=np.int64) for p in preds_b]
    if not a_list or not b_list:
        raise ValueError("preds_a and preds_b must each hold at least one prediction vector")
    if n == 0:
        raise ValueError("y_true is empty")
    for pred in a_list + b_list:
        if len(pred) != n:
            raise ValueError(f"Length mismatch: len(y_true)={n}, len(pred)={len(pred)}")

    all_preds = a_list + b_list
    onehot = _cell_onehot(y, all_preds)
    n_a = len(a_list)

    def delta_from_counts(counts: np.ndarray) -> np.ndarray:
        per_model = [
            _metric_from_counts(counts[:, 9 * j : 9 * (j + 1)], metric) for j in range(len(all_preds))
        ]
        return np.mean(per_model[:n_a], axis=0) - np.mean(per_model[n_a:], axis=0)

    delta_obs = float(delta_from_counts(onehot.sum(axis=0, keepdims=True))[0])

    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=np.float64)
    chunk = max(1, min(B, 250))
    done = 0
    while done < B:
        size = min(chunk, B - done)
        idx = rng.integers(0, n, size=(size, n))
        flat = (idx + (np.arange(size) * n)[:, None]).ravel()
        weights = np.bincount(flat, minlength=size * n).reshape(size, n).astype(np.float64)
        deltas[done : done + size] = delta_from_counts(weights @ onehot)
        done += size

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    tail = min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return {
        "delta": delta_obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, 2.0 * tail)),
        "B": int(B),
        "n": int(n),
        "metric": metric,
    }


def seed_sentence_bootstrap(
    y_true: Sequence[int],
    preds_a: Sequence[Sequence[int]],
    preds_b: Sequence[Sequence[int]],
    metric: str = "macro_f1",
    B: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Two-level bootstrap of ``mean_s metric(A_s) - mean_s metric(B_s)`` over sentences and seeds.

    Per resample, ``n`` sentence indices are drawn with replacement and, independently for A and
    B, ``k`` seed indices are drawn with replacement from that model's ``k`` prediction vectors
    (``k = 1`` always selects the one run). The statistic is the mean over the drawn seeds of the
    metric of A minus the same for B, both scored on the drawn sentences. Sentences and seeds use
    separate random streams, so with one run per model the result equals ``seed_averaged_bootstrap``.

    Returns the keys of ``seed_averaged_bootstrap`` plus ``"per_seed_delta"`` (``metric(A_i) -
    metric(B_i)`` for equal seed positions when both models have ``k`` seeds, else ``A_i - B_0``)
    and ``"method": "seed+sentence"``. The two-sided p-value is ``min(1, 2 * min(P(d <= 0),
    P(d >= 0)))``, so the smallest non-zero value is ``2 / B``.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    y = np.asarray(y_true, dtype=np.int64)
    n = len(y)
    a_list = [np.asarray(p, dtype=np.int64) for p in preds_a]
    b_list = [np.asarray(p, dtype=np.int64) for p in preds_b]
    if not a_list or not b_list:
        raise ValueError("preds_a and preds_b must each hold at least one prediction vector")
    if n == 0:
        raise ValueError("y_true is empty")
    for pred in a_list + b_list:
        if len(pred) != n:
            raise ValueError(f"Length mismatch: len(y_true)={n}, len(pred)={len(pred)}")

    n_models = len(a_list) + len(b_list)
    k_a, k_b = len(a_list), len(b_list)
    onehot = _cell_onehot(y, a_list + b_list)

    def per_model_metrics(counts: np.ndarray) -> np.ndarray:
        """(rows, n_models) metric of every prediction vector for every row of ``counts``."""
        return np.stack(
            [_metric_from_counts(counts[:, 9 * j : 9 * (j + 1)], metric) for j in range(n_models)], axis=1
        )

    observed = per_model_metrics(onehot.sum(axis=0, keepdims=True))[0]
    delta_obs = float(np.mean(observed[:k_a]) - np.mean(observed[k_a:]))
    if k_a == k_b:
        per_seed_delta = [float(observed[i] - observed[k_a + i]) for i in range(k_a)]
    else:
        per_seed_delta = [float(observed[i] - observed[k_a]) for i in range(k_a)]

    sentence_rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng([seed, 1])
    deltas = np.empty(B, dtype=np.float64)
    chunk = max(1, min(B, 250))
    done = 0
    while done < B:
        size = min(chunk, B - done)
        idx = sentence_rng.integers(0, n, size=(size, n))
        flat = (idx + (np.arange(size) * n)[:, None]).ravel()
        weights = np.bincount(flat, minlength=size * n).reshape(size, n).astype(np.float64)
        scores = per_model_metrics(weights @ onehot)
        pick_a = seed_rng.integers(0, k_a, size=(size, k_a))
        pick_b = seed_rng.integers(0, k_b, size=(size, k_b))
        mean_a = np.take_along_axis(scores[:, :k_a], pick_a, axis=1).mean(axis=1)
        mean_b = np.take_along_axis(scores[:, k_a:], pick_b, axis=1).mean(axis=1)
        deltas[done : done + size] = mean_a - mean_b
        done += size

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    tail = min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return {
        "delta": delta_obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, 2.0 * tail)),
        "B": int(B),
        "n": int(n),
        "metric": metric,
        "per_seed_delta": per_seed_delta,
        "method": "seed+sentence",
    }
