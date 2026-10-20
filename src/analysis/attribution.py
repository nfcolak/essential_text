"""Annotator agreement, consensus tags and per-phenomenon error / attribution statistics (Step 3).

Self-contained: numpy and the standard library only. Every bootstrap is two-level: each resample
draws sentences with replacement and, independently per model, draws its trained seeds with
replacement (a baseline with one run always reuses it), so the intervals include seed variation.

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

    Only ids present in all three sources are compared. `n_consensus_vs_rule_diff` counts the
    sentences where the both-annotators consensus differs from the rule tag.
    """
    ids = sorted(set(ann_a) & set(ann_b) & set(rules))
    table: dict[str, Any] = {}
    for name in PHENOMENA:
        a, b, r = _column(ann_a, ids, name), _column(ann_b, ids, name), _column(rules, ids, name)
        table[name] = {
            "n": len(ids),
            "n_consensus_vs_rule_diff": sum(1 for x, y, z in zip(a, b, r) if (x & y) != z),
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



LABELS = (0, 1, 2)  # negative, neutral, positive
_LEMMAS = {
    "losses": "loss", "taxes": "tax", "liabilities": "liability", "leveraged": "leverage",
    "bears": "bear",
}


def normalise_cue(term: str) -> str:
    """Lemma family of a matched financial-terminology cue (loss/losses -> loss, ...)."""
    t = " ".join(term.lower().replace("-", " ").split())
    t = t.replace("write down", "write-down").replace("write off", "write-off")
    t = t.replace("writedown", "write-down").replace("writeoff", "write-off")
    if t in _LEMMAS:
        return _LEMMAS[t]
    if t.endswith("ss") or not t.endswith("s"):
        return t
    return t[:-1]


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _correct_matrix(ids: list[str], y_true: dict, seed_preds: list[dict]) -> np.ndarray:
    """(n_seeds, n_ids) 0/1 array: prediction == gold."""
    gold = np.array([int(y_true[i]) for i in ids])
    return np.stack(
        [np.array([int(p[i]) for i in ids]) == gold for p in seed_preds]
    ).astype(float)


def _boot_stats(mats: dict[str, np.ndarray], B: int, seed: int,
                gold: np.ndarray | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Two-level (seed + sentence) bootstrap of several (n_seeds, n) 0/1 matrices.

    Per resample one set of sentence indices is shared by all matrices (paired); every matrix
    draws its own k seed indices with replacement (k = 1 -> the single run). The statistic is the
    mean over the drawn seeds of the sentence mean. With `gold` the label-balanced mean (mean over
    the gold classes present in the resample of the class mean) is returned as well.
    """
    n = next(iter(mats.values())).shape[1]
    rng = np.random.default_rng(seed)
    means = {k: np.empty(B) for k in mats}
    bals = {k: np.empty(B) for k in mats} if gold is not None else None
    step = max(1, _CHUNK_CELLS // max(n, 1))
    done = 0
    while done < B:
        c = min(step, B - done)
        idx = rng.integers(0, n, size=(c, n))
        if gold is not None:
            gi = gold[idx]
            masks = [(gi == lab) for lab in LABELS]
            counts = [m.sum(axis=1) for m in masks]
            present = sum((cnt > 0).astype(float) for cnt in counts)
        for key, mat in mats.items():
            k = mat.shape[0]
            draws = rng.integers(0, k, size=(c, k))
            acc = np.zeros(c)
            acc_bal = np.zeros(c)
            for j in range(k):
                vals = mat[draws[:, j][:, None], idx]
                acc += vals.mean(axis=1)
                if gold is not None:
                    bal = np.zeros(c)
                    for m, cnt in zip(masks, counts):
                        bal += np.where(cnt > 0, (vals * m).sum(axis=1) / np.maximum(cnt, 1), 0.0)
                    acc_bal += bal / present
            means[key][done:done + c] = acc / k
            if bals is not None:
                bals[key][done:done + c] = acc_bal / k
        done += c
    return means, bals


def _balanced_point(mat: np.ndarray, gold: np.ndarray) -> float:
    """Seed-mean of the mean per-class value over the gold classes present."""
    per_class = [mat[:, gold == lab].mean(axis=1) for lab in LABELS if (gold == lab).any()]
    return float(np.mean(per_class, axis=0).mean())


def _gold_mix(gold: list[int]) -> dict[str, int]:
    return {"n_neg": int(sum(1 for g in gold if g == 0)),
            "n_neu": int(sum(1 for g in gold if g == 1)),
            "n_pos": int(sum(1 for g in gold if g == 2))}


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _p_two_sided(samples: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min(np.mean(samples <= 0), np.mean(samples >= 0))))


def _empty_rate() -> dict[str, Any]:
    return {"error_rate": None, "ci_low": None, "ci_high": None, "balanced_error": None,
            "balanced_ci_low": None, "balanced_ci_high": None, "per_seed": [], "n": 0,
            "small_n": True}


def _subset_rates(ids: list[str], y_true: dict, preds_by_model: dict, B: int, seed: int) -> dict:
    if not ids:
        return {m: _empty_rate() for m in preds_by_model}
    gold = np.array([int(y_true[i]) for i in ids])
    errors = {m: 1.0 - _correct_matrix(ids, y_true, seeds) for m, seeds in preds_by_model.items()}
    means, bals = _boot_stats(errors, B, seed, gold)
    out = {}
    for m, e in errors.items():
        low, high = _ci(means[m])
        blow, bhigh = _ci(bals[m])
        out[m] = {
            "error_rate": float(e.mean()), "ci_low": low, "ci_high": high,
            "balanced_error": _balanced_point(e, gold),
            "balanced_ci_low": blow, "balanced_ci_high": bhigh,
            "per_seed": [float(x) for x in e.mean(axis=1)],
            "n": int(len(ids)), "small_n": bool(len(ids) < SMALL_N),
        }
    return out


def phenomenon_error_rates(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                           B: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Seed-mean error rate (1 - accuracy) per model on tag = 1 and the tag = 0 complement.

    Result: `{phenomenon: {"n_tag1", "n_tag0", "small_n", "gold_mix_tag1", "gold_mix_tag0",
    "models": {model: {"tag1": entry, "tag0": entry}}}}` with entry = `{error_rate, ci_low,
    ci_high, balanced_error, balanced_ci_low, balanced_ci_high, per_seed, n, small_n}`. CIs come
    from the two-level (seed + sentence) bootstrap. `balanced_error` is the mean over the gold
    classes present (>= 1 sentence) of the class error rate; `gold_mix_*` counts the gold labels.
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
            "gold_mix_tag1": _gold_mix([int(y_true[i]) for i in pos]),
            "gold_mix_tag0": _gold_mix([int(y_true[i]) for i in neg]),
            "models": {m: {"tag1": r1[m], "tag0": r0[m]} for m in preds_by_model},
        }
    return result


def _per_seed_delta(hi: np.ndarray, lo: np.ndarray) -> list[float]:
    """Per-seed accuracy difference: equal seed positions when both have k seeds, else hi_i - lo_0."""
    a, b = hi.mean(axis=1), lo.mean(axis=1)
    if len(a) == len(b):
        return [float(x) for x in a - b]
    return [float(x - b[0]) for x in a]


def phenomenon_attribution(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                           tfidf_best: str, B: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Architecture and domain-pretraining effect on accuracy within each tag = 1 subset.

    `delta_arch = acc(bert_base) - acc(tfidf_best)`, `delta_domain = acc(finbert_mlm) -
    acc(bert_base)`, both seed-averaged, with a two-level (seed + sentence) bootstrap 95% CI.
    Subsets with fewer than 20 sentences are `small_n`: CI only, `p_value` and `p_holm` are None.
    Other rows carry the raw two-sided p-value and `p_holm`, Holm-adjusted within this table
    (the tested rows of the 5 phenomena x 2 deltas). A zero p is reported as p < 2/B and enters
    the Holm step as 2/B.
    """
    for key in ("bert_base", "finbert_mlm", tfidf_best):
        if key not in preds_by_model:
            raise KeyError(f"Model '{key}' is missing from preds_by_model.")
    ids_all = sorted(i for i in tags if i in y_true)
    result: dict[str, Any] = {}
    tested: list[tuple[str, str]] = []
    for name in PHENOMENA:
        pos = [i for i in ids_all if int(tags[i][name]) == 1]
        entry: dict[str, Any] = {"n": len(pos), "small_n": bool(len(pos) < SMALL_N),
                                 "tfidf_best": tfidf_best}
        if not pos:
            empty = {"delta": None, "ci_low": None, "ci_high": None, "p_value": None,
                     "p_holm": None, "per_seed_delta": []}
            entry["delta_arch"], entry["delta_domain"] = dict(empty), dict(empty)
            result[name] = entry
            continue
        mats = {k: _correct_matrix(pos, y_true, preds_by_model[k])
                for k in ("bert_base", "finbert_mlm", tfidf_best)}
        boots, _ = _boot_stats(mats, B, seed)
        for label, hi, lo in (("delta_arch", "bert_base", tfidf_best),
                              ("delta_domain", "finbert_mlm", "bert_base")):
            draws = boots[hi] - boots[lo]
            low, high = _ci(draws)
            entry[label] = {
                "delta": float(mats[hi].mean() - mats[lo].mean()),
                "ci_low": low,
                "ci_high": high,
                "p_value": None if entry["small_n"] else _p_two_sided(draws),
                "p_holm": None,
                "per_seed_delta": _per_seed_delta(mats[hi], mats[lo]),
                "B": int(len(draws)),
            }
            if not entry["small_n"]:
                tested.append((name, label))
        result[name] = entry
    if tested:
        floor = 2.0 / B
        raw = [max(result[n][lab]["p_value"], floor) if result[n][lab]["p_value"] == 0.0
               else result[n][lab]["p_value"] for n, lab in tested]
        for (n, lab), adj in zip(tested, holm_adjust(raw)):
            result[n][lab]["p_holm"] = float(adj)
    return result


def rq1_terminology_comparison(tags: dict, y_true: dict, preds_by_model: dict[str, list[dict]],
                               B: int = 10000, seed: int = 42,
                               drop_ids: set | None = None) -> dict[str, Any]:
    """RQ1: neutral-to-negative rate on gold-neutral sentences, financial_terminology = 1 vs = 0.

    `drop_ids` removes sentences from the term = 1 group (used for the loss-excluded table).
    Result: `{model: {"term1": entry, "term0": entry, "difference": {delta, ci_low, ci_high}}}`
    with entry = `{rate, ci_low, ci_high, n, small_n, na}`; `na` is set when n < 5. CIs resample
    sentences and seeds (two-level).
    """
    drop = drop_ids or set()
    neutral = [i for i in sorted(tags) if i in y_true and int(y_true[i]) == 1]
    groups = {
        "term1": [i for i in neutral
                  if int(tags[i]["financial_terminology"]) == 1 and i not in drop],
        "term0": [i for i in neutral if int(tags[i]["financial_terminology"]) == 0],
    }
    mats: dict[str, dict[str, np.ndarray]] = {}
    for g, ids in groups.items():
        mats[g] = {
            m: np.stack([np.array([int(p[i]) == 0 for i in ids]) for p in seeds]).astype(float)
            for m, seeds in preds_by_model.items()
        } if ids else {}
    boots = {g: _boot_stats(v, B, seed + k)[0] if v else {}
             for k, (g, v) in enumerate(mats.items())}
    result: dict[str, Any] = {}
    for m in preds_by_model:
        entry: dict[str, Any] = {}
        for g, ids in groups.items():
            if ids:
                low, high = _ci(boots[g][m])
                entry[g] = {"rate": float(mats[g][m].mean()), "ci_low": low, "ci_high": high,
                            "n": len(ids), "small_n": bool(len(ids) < SMALL_N),
                            "na": bool(len(ids) < 5)}
            else:
                entry[g] = {"rate": None, "ci_low": None, "ci_high": None, "n": 0,
                            "small_n": True, "na": True}
        if groups["term1"] and groups["term0"]:
            diff = boots["term1"][m] - boots["term0"][m]  # independent groups
            low, high = _ci(diff)
            entry["difference"] = {"delta": entry["term1"]["rate"] - entry["term0"]["rate"],
                                   "ci_low": low, "ci_high": high}
        else:
            entry["difference"] = {"delta": None, "ci_low": None, "ci_high": None}
        result[m] = entry
    return result


def loss_only_ids(cues: dict[str, list[str]]) -> set:
    """Ids whose only financial-terminology cue is loss/losses."""
    return {i for i, terms in cues.items() if terms and {normalise_cue(t) for t in terms} == {"loss"}}


def rq1_cue_breakdown(cues: dict[str, list[str]], y_true: dict,
                      preds_by_model: dict[str, list[dict]], B: int = 10000,
                      seed: int = 42) -> dict[str, Any]:
    """RQ1 neutral-to-negative rate per cue-word family on gold-neutral sentences.

    `cues[id]` lists the matched terms of a sentence (any case); a sentence counts once in every
    family it contains. Result: `{cue: {"n", "small_n", "na", "models": {model: {rate, ci_low,
    ci_high}}}}`, cues ordered by n (descending), then name.
    """
    families: dict[str, list[str]] = {}
    for i in sorted(cues):
        if i not in y_true or int(y_true[i]) != 1:
            continue
        for fam in sorted({normalise_cue(t) for t in cues[i]}):
            families.setdefault(fam, []).append(i)
    result: dict[str, Any] = {}
    ordered = sorted(families, key=lambda f: (-len(families[f]), f))
    for k, fam in enumerate(ordered):
        ids = families[fam]
        mats = {m: np.stack([np.array([int(p[i]) == 0 for i in ids]) for p in seeds]).astype(float)
                for m, seeds in preds_by_model.items()}
        boots, _ = _boot_stats(mats, B, seed + 100 + k)
        models = {}
        for m, mat in mats.items():
            low, high = _ci(boots[m])
            models[m] = {"rate": float(mat.mean()), "ci_low": low, "ci_high": high}
        result[fam] = {"n": len(ids), "small_n": bool(len(ids) < SMALL_N),
                       "na": bool(len(ids) < 5), "models": models}
    return result
