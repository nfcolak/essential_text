"""Annotator agreement, consensus tags and per-phenomenon error / attribution statistics (Step 3).

Self-contained: numpy and the standard library only. Every bootstrap resamples sentences, so a
sentence-level 0/1 vector is averaged over the seeds of a model first (the mean over seeds is
linear, so this equals averaging the per-seed statistics on every resample).

Prediction containers are `preds_by_model[model] = [ {record_id: predicted_label}, ... ]`, one dict
per seed (a baseline has a single dict). `tags` maps an id to its 0/1 phenomenon dict and `y_true`
maps the same id to the gold label.
"""

from __future__ import annotations

import csv
import math
from typing import Any

import numpy as np

PHENOMENA = (
    "financial_terminology",
    "negation",
    "hedging",
    "numerical_comparison",
    "context_dependent",
)
SMALL_N = 20
_CHUNK_CELLS = 4_000_000  # resample matrix cells per chunk (bounds memory)


def read_annotations(path: str) -> dict[str, dict[str, int]]:
    """Read an annotation CSV (`diag_id` + the 5 phenomenon columns, optional `note`).

    Raises ValueError on a missing column, a duplicate id or a value outside {0, 1}.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in ("diag_id",) + PHENOMENA if c not in header]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; header is {header}.")
        reader.fieldnames = header
        out: dict[str, dict[str, int]] = {}
        for line_no, row in enumerate(reader, start=2):
            diag_id = (row.get("diag_id") or "").strip()
            if not diag_id:
                raise ValueError(f"{path} line {line_no}: empty diag_id.")
            if diag_id in out:
                raise ValueError(f"{path} line {line_no}: duplicate diag_id '{diag_id}'.")
            tags: dict[str, int] = {}
            for name in PHENOMENA:
                value = (row.get(name) or "").strip()
                if value not in ("0", "1"):
                    raise ValueError(
                        f"{path} line {line_no}: column '{name}' must be 0 or 1, got '{value}'."
                    )
                tags[name] = int(value)
            out[diag_id] = tags
    return out


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two label sequences.

    Returns 1.0 when both raters are constant and identical (chance agreement is 1) and nan when
    the sequences are empty or differ in length.
    """
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = 0.0
    for label in set(a) | set(b):
        expected += (sum(1 for x in a if x == label) / n) * (sum(1 for y in b if y == label) / n)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else float("nan")
    return (observed - expected) / (1.0 - expected)


def _column(tags: dict, ids: list[str], name: str) -> list[int]:
    return [int(tags[i][name]) for i in ids]


def _raw_agreement(a: list[int], b: list[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else float("nan")


def _prevalence(a: list[int]) -> float:
    return sum(a) / len(a) if a else float("nan")


def agreement_table(ann_a: dict, ann_b: dict, rules: dict) -> dict[str, Any]:
    """Per-phenomenon kappa (A-B, A-rule, B-rule), raw agreement and prevalence of A, B and the rules.

    Only ids present in all three sources are compared.
    """
    ids = sorted(set(ann_a) & set(ann_b) & set(rules))
    table: dict[str, Any] = {}
    for name in PHENOMENA:
        a, b, r = _column(ann_a, ids, name), _column(ann_b, ids, name), _column(rules, ids, name)
        table[name] = {
            "n": len(ids),
            "kappa_ab": cohen_kappa(a, b),
            "kappa_a_rule": cohen_kappa(a, r),
            "kappa_b_rule": cohen_kappa(b, r),
            "raw_agreement_ab": _raw_agreement(a, b),
            "raw_agreement_a_rule": _raw_agreement(a, r),
            "raw_agreement_b_rule": _raw_agreement(b, r),
            "prevalence_a": _prevalence(a),
            "prevalence_b": _prevalence(b),
            "prevalence_rule": _prevalence(r),
        }
    return table


def consensus_tags(ann_a: dict, ann_b: dict, mode: str = "both") -> dict[str, dict[str, int]]:
    """Merge two annotators: `both` = A and B, `either` = A or B (ids present in both)."""
    if mode not in ("both", "either"):
        raise ValueError(f"mode must be 'both' or 'either', got '{mode}'.")
    merged: dict[str, dict[str, int]] = {}
    for diag_id in sorted(set(ann_a) & set(ann_b)):
        merged[diag_id] = {}
        for name in PHENOMENA:
            x, y = int(ann_a[diag_id][name]), int(ann_b[diag_id][name])
            merged[diag_id][name] = (x & y) if mode == "both" else (x | y)
    return merged


def _correct_matrix(ids: list[str], y_true: dict, seed_preds: list[dict]) -> np.ndarray:
    """(n_seeds, n_ids) 0/1 array: prediction == gold."""
    gold = np.array([int(y_true[i]) for i in ids])
    return np.stack(
        [np.array([int(p[i]) for i in ids]) == gold for p in seed_preds]
    ).astype(float)


def _boot_means(vectors: dict[str, np.ndarray], B: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap means of several per-sentence vectors under shared resample indices (paired)."""
    n = len(next(iter(vectors.values())))
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in vectors}
    step = max(1, _CHUNK_CELLS // max(n, 1))
    done = 0
    while done < B:
        c = min(step, B - done)
        idx = rng.integers(0, n, size=(c, n))
        for k, v in vectors.items():
            out[k][done:done + c] = v[idx].mean(axis=1)
        done += c
    return out


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _p_two_sided(samples: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min(np.mean(samples <= 0), np.mean(samples >= 0))))


def _rate_entry(vec: np.ndarray, boot: np.ndarray, per_seed: list[float]) -> dict[str, Any]:
    low, high = _ci(boot)
    return {
        "error_rate": float(vec.mean()),
        "ci_low": low,
        "ci_high": high,
        "per_seed": per_seed,
        "n": int(len(vec)),
        "small_n": bool(len(vec) < SMALL_N),
    }


def _empty_rate() -> dict[str, Any]:
    return {"error_rate": None, "ci_low": None, "ci_high": None, "per_seed": [], "n": 0,
            "small_n": True}


def _subset_rates(ids: list[str], y_true: dict, preds_by_model: dict, B: int, seed: int) -> dict:
    if not ids:
        return {m: _empty_rate() for m in preds_by_model}
    errors = {m: 1.0 - _correct_matrix(ids, y_true, seeds) for m, seeds in preds_by_model.items()}
    boots = _boot_means({m: e.mean(axis=0) for m, e in errors.items()}, B, seed)
    return {
        m: _rate_entry(e.mean(axis=0), boots[m], [float(x) for x in e.mean(axis=1)])
        for m, e in errors.items()
    }


def phenomenon_error_rates(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                           B: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Seed-mean error rate (1 - accuracy) per model on tag = 1 and the tag = 0 complement.

    Result: `{phenomenon: {"n_tag1", "n_tag0", "small_n", "models": {model: {"tag1": entry,
    "tag0": entry}}}}` with entry = `{error_rate, ci_low, ci_high (sentence bootstrap), per_seed, n,
    small_n}`.
    """
    ids_all = sorted(i for i in tags if i in y_true)
    result: dict[str, Any] = {}
    for name in PHENOMENA:
        pos = [i for i in ids_all if int(tags[i][name]) == 1]
        neg = [i for i in ids_all if int(tags[i][name]) == 0]
        r1 = _subset_rates(pos, y_true, preds_by_model, B, seed)
        r0 = _subset_rates(neg, y_true, preds_by_model, B, seed)
        result[name] = {
            "n_tag1": len(pos),
            "n_tag0": len(neg),
            "small_n": bool(len(pos) < SMALL_N),
            "models": {m: {"tag1": r1[m], "tag0": r0[m]} for m in preds_by_model},
        }
    return result


def phenomenon_attribution(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                           tfidf_best: str, B: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Architecture and domain-pretraining effect on accuracy within each tag = 1 subset.

    `delta_arch = acc(bert_base) - acc(tfidf_best)`, `delta_domain = acc(finbert_mlm) -
    acc(bert_base)`, both seed-averaged, with a sentence-bootstrap 95% CI and a two-sided p-value.
    `small_n` is set when the subset has fewer than 20 sentences.
    """
    for key in ("bert_base", "finbert_mlm", tfidf_best):
        if key not in preds_by_model:
            raise KeyError(f"Model '{key}' is missing from preds_by_model.")
    ids_all = sorted(i for i in tags if i in y_true)
    result: dict[str, Any] = {}
    for name in PHENOMENA:
        pos = [i for i in ids_all if int(tags[i][name]) == 1]
        entry: dict[str, Any] = {"n": len(pos), "small_n": bool(len(pos) < SMALL_N),
                                 "tfidf_best": tfidf_best}
        if not pos:
            empty = {"delta": None, "ci_low": None, "ci_high": None, "p_value": None}
            entry["delta_arch"], entry["delta_domain"] = dict(empty), dict(empty)
            result[name] = entry
            continue
        acc = {
            k: _correct_matrix(pos, y_true, preds_by_model[k]).mean(axis=0)
            for k in ("bert_base", "finbert_mlm", tfidf_best)
        }
        boots = _boot_means(acc, B, seed)
        for label, hi, lo in (("delta_arch", "bert_base", tfidf_best),
                              ("delta_domain", "finbert_mlm", "bert_base")):
            draws = boots[hi] - boots[lo]
            low, high = _ci(draws)
            entry[label] = {
                "delta": float(acc[hi].mean() - acc[lo].mean()),
                "ci_low": low,
                "ci_high": high,
                "p_value": _p_two_sided(draws),
            }
        result[name] = entry
    return result


def rq1_terminology_comparison(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                               B: int = 10000, seed: int = 42) -> dict[str, Any]:
    """RQ1: neutral-to-negative rate on gold-neutral sentences, financial_terminology = 1 vs = 0.

    Result: `{model: {"term1": entry, "term0": entry, "difference": {delta, ci_low, ci_high}}}`
    with entry = `{rate, ci_low, ci_high, n, small_n}`.
    """
    neutral = [i for i in sorted(tags) if i in y_true and int(y_true[i]) == 1]
    groups = {
        "term1": [i for i in neutral if int(tags[i]["financial_terminology"]) == 1],
        "term0": [i for i in neutral if int(tags[i]["financial_terminology"]) == 0],
    }
    vecs: dict[str, dict[str, np.ndarray]] = {}
    for g, ids in groups.items():
        vecs[g] = {
            m: np.stack([np.array([int(p[i]) == 0 for i in ids]) for p in seeds]).astype(float)
            .mean(axis=0)
            for m, seeds in preds_by_model.items()
        } if ids else {}
    boots = {g: _boot_means(v, B, seed + k) if v else {}
             for k, (g, v) in enumerate(vecs.items())}
    result: dict[str, Any] = {}
    for m in preds_by_model:
        entry: dict[str, Any] = {}
        for g, ids in groups.items():
            if ids:
                low, high = _ci(boots[g][m])
                entry[g] = {"rate": float(vecs[g][m].mean()), "ci_low": low, "ci_high": high,
                            "n": len(ids), "small_n": bool(len(ids) < SMALL_N)}
            else:
                entry[g] = {"rate": None, "ci_low": None, "ci_high": None, "n": 0, "small_n": True}
        if groups["term1"] and groups["term0"]:
            # independent groups: resample each with its own stream, then difference
            diff = boots["term1"][m] - boots["term0"][m]
            low, high = _ci(diff)
            entry["difference"] = {"delta": entry["term1"]["rate"] - entry["term0"]["rate"],
                                   "ci_low": low, "ci_high": high}
        else:
            entry["difference"] = {"delta": None, "ci_low": None, "ci_high": None}
        result[m] = entry
    return result
