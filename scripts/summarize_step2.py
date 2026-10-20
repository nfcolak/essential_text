"""Summarise the Step 2 run matrix: seed statistics, paired tests, attribution, report, figures.

Reads every run only through ``src.evaluation.multiseed.load_runs`` and writes into ``--out-dir``:
``summary_metrics.json``, ``statistical_comparisons.json``, ``attribution.json``,
``STEP2_REPORT.md`` and four PNG figures. Exits 2 when planned runs are missing unless
``--allow-partial`` is given.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})

from src.evaluation.metrics import CLASS_NAMES  # noqa: E402
from src.evaluation.multiseed import (  # noqa: E402
    holm_bonferroni,
    load_runs,
    mean_sd,
    seed_averaged_bootstrap,
    seed_sentence_bootstrap,
)
from src.evaluation.stats import mcnemar_test  # noqa: E402

CONFIGS = ("sentences_allagree", "sentences_75agree", "sentences_50agree")
BASELINE_MODELS = ("vader_official", "vader_official_tuned", "lm_official", "lm_official_tuned",
                   "vader", "loughran_mcdonald", "tfidf_logreg", "tfidf_svm")
TRANSFORMER_MODELS = ("bert_base", "finbert_mlm")
MODELS = BASELINE_MODELS + TRANSFORMER_MODELS
ZEROSHOT_MODEL = "prosus_finbert"
ZEROSHOT_CONFIG = "external_only"
CV_CONFIG = "sentences_allagree"
PRIMARY_SEED = 42
DEFAULT_SEEDS = (42, 100, 2026)
CV_FOLDS = (0, 1, 2, 3, 4)
SPLITS = ("test", "external")
# RQ1 boundary pairs (a, b), compared on every config x {test, external}
BOUNDARY_PAIRS = (
    ("vader_official", "lm_official"),
    ("vader_official_tuned", "lm_official_tuned"),
    ("vader", "loughran_mcdonald"),
    ("bert_base", "finbert_mlm"),
)
# extra pairs on the external split only (zero-shot ProsusAI/finbert against the fine-tuned encoders)
BOUNDARY_PAIRS_EXTERNAL = (("prosus_finbert", "bert_base"), ("prosus_finbert", "finbert_mlm"))
PER_CLASS_STATS = ("precision", "recall", "f1")

FIGURES = (
    "step2_macro_f1_by_config.png",
    "step2_external_generalization.png",
    "step2_attribution.png",
    "step2_confusion_matrices_allagree.png",
)


def display_name(key: str) -> str:
    """Display name of a model key (lazy registry import; the key itself when unavailable)."""
    try:
        from src.baselines.registry import DISPLAY_NAMES
    except ImportError:
        return key
    return DISPLAY_NAMES.get(key, key)


def _seed_of(tag: str) -> int | None:
    return int(tag[4:]) if tag.startswith("seed") and tag[4:].isdigit() else None


class RunTable:
    """Index of completed runs: (config, model, tag) -> run dict."""

    def __init__(self, runs: list[dict]) -> None:
        self.by_key = {(r["config"], r["model_key"], r["tag"]): r for r in runs}
        self.runs = runs

    def get(self, config: str, model: str, tag: str) -> dict | None:
        return self.by_key.get((config, model, tag))

    def seeds(self, config: str, model: str) -> dict[int, dict]:
        """Seed -> run for the ``seed<k>`` tagged runs of one config x model."""
        out: dict[int, dict] = {}
        for (c, m, tag), run in self.by_key.items():
            seed = _seed_of(tag)
            if c == config and m == model and seed is not None:
                out[seed] = run
        return dict(sorted(out.items()))

    def cv_folds(self, model: str) -> dict[int, dict]:
        out = {}
        for (c, m, tag), run in self.by_key.items():
            if c == CV_CONFIG and m == model and tag.startswith("cv_fold") and tag[7:].isdigit():
                out[int(tag[7:])] = run
        return dict(sorted(out.items()))


def planned_runs(seeds: tuple[int, ...]) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    for config in CONFIGS:
        for model in BASELINE_MODELS:
            plan.append((config, model, "seed42"))
    for config in CONFIGS:
        for model in TRANSFORMER_MODELS:
            for seed in seeds:
                plan.append((config, model, f"seed{seed}"))
    for model in MODELS:
        for fold in CV_FOLDS:
            plan.append((CV_CONFIG, model, f"cv_fold{fold}"))
    plan.append((ZEROSHOT_CONFIG, ZEROSHOT_MODEL, "zeroshot"))
    return plan


# ---------------------------------------------------------------------------
# prediction access
# ---------------------------------------------------------------------------


def _split(run: dict, split: str) -> dict | None:
    return run["predictions"].get(split)


def _aligned_pred(run: dict, split: str, ref_ids: np.ndarray) -> np.ndarray:
    """Argmax predictions of ``run`` on ``split`` in the row order of ``ref_ids``."""
    block = _split(run, split)
    if block is None:
        raise KeyError(f"{run['run_dir']} has no '{split}' predictions")
    pred = np.argmax(block["prob"], axis=1).astype(np.int64)
    ids = block["ids"]
    if np.array_equal(ids, ref_ids):
        return pred
    order = {i: k for k, i in enumerate(ids.tolist())}
    if len(order) != len(ids) or set(order) != set(ref_ids.tolist()):
        raise ValueError(f"{run['run_dir']}: '{split}' ids differ from the reference id set")
    return pred[[order[i] for i in ref_ids.tolist()]]


def _model_preds(table: RunTable, config: str, model: str, split: str, ref_ids: np.ndarray) -> dict[int, np.ndarray]:
    """seed -> aligned argmax predictions of a (config, model) on a split."""
    if model == ZEROSHOT_MODEL:
        run = table.get(ZEROSHOT_CONFIG, ZEROSHOT_MODEL, "zeroshot")
        if run is None or _split(run, split) is None:
            return {}
        return {PRIMARY_SEED: _aligned_pred(run, split, ref_ids)}
    out = {}
    for seed, run in table.seeds(config, model).items():
        if _split(run, split) is not None:
            out[seed] = _aligned_pred(run, split, ref_ids)
    return out


def _reference(table: RunTable, config: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(ids, y) of the first run with a ``split`` block in ``config`` (None when no run has one)."""
    for model in MODELS:
        for run in table.seeds(config, model).values():
            block = _split(run, split)
            if block is not None:
                return block["ids"], block["y"]
    return None


# ---------------------------------------------------------------------------
# summary_metrics.json
# ---------------------------------------------------------------------------


def _metric_values(runs: list[dict], split: str) -> dict[str, Any]:
    metrics = [r["metrics"][split] for r in runs if r.get("metrics", {}).get(split)]
    out: dict[str, Any] = {
        "accuracy": mean_sd([m["accuracy"] for m in metrics]),
        "macro_f1": mean_sd([m["macro_f1"] for m in metrics]),
        "neutral_to_negative_rate": mean_sd([m["neutral_to_negative_rate"] for m in metrics]),
        "ece": mean_sd([m["ece"] for m in metrics]),
        "per_class_f1": {
            name: mean_sd([m["per_class"][name]["f1"] for m in metrics]) for name in CLASS_NAMES
        },
        # seed mean +- sd of per-class precision / recall / F1, read from result.json metrics ``per_class``
        "per_class": {
            name: {stat: mean_sd([m["per_class"][name][stat] for m in metrics]) for stat in PER_CLASS_STATS}
            for name in CLASS_NAMES
        },
    }
    return out


def build_summary(table: RunTable) -> dict[str, Any]:
    summary: dict[str, Any] = {"configs": {}, "cv": {}, "zeroshot": {}, "run_counts": {}}
    for config in CONFIGS:
        summary["configs"][config] = {}
        for model in MODELS:
            runs = list(table.seeds(config, model).values())
            if not runs:
                continue
            entry = {
                "seeds": sorted(table.seeds(config, model)),
                "n_runs": len(runs),
                "test": _metric_values(runs, "test"),
                "external": _metric_values(runs, "external"),
                "best_epoch": mean_sd([r.get("best_epoch") for r in runs]),
            }
            summary["configs"][config][model] = entry
    for model in MODELS:
        folds = list(table.cv_folds(model).values())
        if folds:
            summary["cv"][model] = {
                "n_folds": len(folds),
                "val": _metric_values(folds, "val"),
                "best_epoch": mean_sd([r.get("best_epoch") for r in folds]),
            }
    zs = table.get(ZEROSHOT_CONFIG, ZEROSHOT_MODEL, "zeroshot")
    if zs is not None:
        summary["zeroshot"][ZEROSHOT_MODEL] = {
            "external": zs["metrics"].get("external"),
            "n": zs.get("n"),
        }
    summary["run_counts"] = {
        "total": len(table.runs),
        "by_config": {c: sum(1 for r in table.runs if r["config"] == c) for c in (*CONFIGS, ZEROSHOT_CONFIG)},
        "cv_runs": sum(1 for r in table.runs if r["tag"].startswith("cv_fold")),
    }
    return summary


# ---------------------------------------------------------------------------
# statistical_comparisons.json
# ---------------------------------------------------------------------------


def _pair_seeds(preds_a: dict[int, np.ndarray], preds_b: dict[int, np.ndarray]) -> list[int]:
    """Seeds a McNemar pairing exists for: equal seeds, or the fixed seed of a one-run model."""
    if len(preds_a) > 1 and len(preds_b) > 1:
        return sorted(set(preds_a) & set(preds_b))
    if len(preds_a) > 1:
        return sorted(preds_a)
    if len(preds_b) > 1:
        return sorted(preds_b)
    return sorted(set(preds_a) | set(preds_b))[:1] if preds_a and preds_b else []


def _pick(preds: dict[int, np.ndarray], seed: int) -> np.ndarray:
    return preds[seed] if seed in preds else next(iter(preds.values()))


def compare_pair(
    y: np.ndarray, preds_a: dict[int, np.ndarray], preds_b: dict[int, np.ndarray], B: int, seed: int
) -> dict[str, Any] | None:
    seeds = _pair_seeds(preds_a, preds_b)
    if not seeds:
        return None
    per_seed = {}
    for s in seeds:
        res = mcnemar_test(y, _pick(preds_a, s), _pick(preds_b, s))
        per_seed[str(s)] = {k: res[k] for k in ("n10", "n01", "chi2", "p_value")}
    boot = seed_sentence_bootstrap(
        y, list(preds_a.values()), list(preds_b.values()), metric="macro_f1", B=B, seed=seed
    )
    return {
        "mcnemar_by_seed": per_seed,
        "bootstrap_macro_f1": boot,
        "n_seeds_a": len(preds_a),
        "n_seeds_b": len(preds_b),
    }


def _family_p(res: dict[str, Any], seed: int) -> float:
    """Raw McNemar p of a pair in the family of ``seed``; a single-run pair counts in every family."""
    per_seed = res["mcnemar_by_seed"]
    if str(seed) in per_seed:
        return float(per_seed[str(seed)]["p_value"])
    return float(next(iter(per_seed.values()))["p_value"])


def apply_seed_families(results: dict[str, Any]) -> list[int]:
    """Holm per seed family; adds ``p_min``, ``p_max``, ``p_holm_by_seed``, ``n_seeds_significant``.

    One family per fine-tuning seed present in any pair; a pair of single-run models contributes its
    one p-value to every family. Returns the family seeds.
    """
    family_seeds = sorted({int(s) for res in results.values() for s in res["mcnemar_by_seed"]})
    holm_by_seed = {
        s: holm_bonferroni({k: _family_p(res, s) for k, res in results.items()}) for s in family_seeds
    }
    for key, res in results.items():
        raw = [_family_p(res, s) for s in family_seeds]
        adjusted = {str(s): holm_by_seed[s][key] for s in family_seeds}
        res["p_min"] = float(min(raw))
        res["p_max"] = float(max(raw))
        res["p_holm_by_seed"] = adjusted
        res["n_seeds_significant"] = int(sum(1 for v in adjusted.values() if v < 0.05))
        res["n_families"] = len(family_seeds)
    return family_seeds


def build_comparisons(table: RunTable, B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": (
            "McNemar per seed pairing; there is no primary seed. Holm-Bonferroni is applied separately "
            "within each seed family (all pairs of one config and split under that seed; pairs of "
            "single-run models appear with their one run in every family). p_min / p_max are the "
            "smallest and largest raw McNemar p over the families; n_seeds_significant counts the "
            "families in which the Holm-adjusted p is below 0.05. Pairs: 45 on test; 55 on external "
            "(the 45 plus the 10 prosus_finbert pairs). The bootstrap is the seed+sentence two-level "
            "bootstrap of the Macro-F1 difference."
        ),
        "configs": {},
    }
    for config in CONFIGS:
        out["configs"][config] = {}
        for split in SPLITS:
            ref = _reference(table, config, split)
            if ref is None:
                continue
            ids, y = ref
            preds = {m: _model_preds(table, config, m, split, ids) for m in MODELS}
            pairs = [(a, b) for a, b in itertools.combinations(MODELS, 2)]
            if split == "external":
                preds[ZEROSHOT_MODEL] = _model_preds(table, config, ZEROSHOT_MODEL, split, ids)
                pairs += [(ZEROSHOT_MODEL, m) for m in MODELS]
            results: dict[str, Any] = {}
            for a, b in pairs:
                if not preds[a] or not preds[b]:
                    continue
                res = compare_pair(y, preds[a], preds[b], B, seed)
                if res is not None:
                    res.update({"model_a": a, "model_b": b})
                    results[f"{a}__vs__{b}"] = res
            family_seeds = apply_seed_families(results)
            out["configs"][config][split] = {"n": int(len(y)), "family_seeds": family_seeds, "pairs": results}
    return out


# ---------------------------------------------------------------------------
# rq1_boundary.json
# ---------------------------------------------------------------------------


def _boundary_rates(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(neutral recall, negative precision) of one prediction vector; 0 when the denominator is 0."""
    gold_neu = y == 1
    pred_neg = pred == 0
    r_neu = float(np.mean(pred[gold_neu] == 1)) if gold_neu.any() else 0.0
    p_neg = float(np.mean(y[pred_neg] == 0)) if pred_neg.any() else 0.0
    return r_neu, p_neg


def _boundary_model(y: np.ndarray, preds: dict[int, np.ndarray]) -> dict[str, float]:
    rates = [_boundary_rates(y, p) for p in preds.values()]
    return {"neutral_recall": float(np.mean([r[0] for r in rates])),
            "negative_precision": float(np.mean([r[1] for r in rates])), "n_seeds": len(rates)}


def compare_boundary_pair(
    y: np.ndarray, preds_a: dict[int, np.ndarray], preds_b: dict[int, np.ndarray], B: int, seed: int
) -> dict[str, Any] | None:
    """R_neu / P_neg of both models, their seed+sentence bootstrap deltas and the gold-neutral McNemar."""
    seeds = _pair_seeds(preds_a, preds_b)
    neutral = y == 1
    if not seeds or not neutral.any():
        return None
    per_seed = {}
    for s in seeds:
        res = mcnemar_test(y[neutral], _pick(preds_a, s)[neutral], _pick(preds_b, s)[neutral])
        per_seed[str(s)] = {k: res[k] for k in ("n10", "n01", "chi2", "p_value")}
    out: dict[str, Any] = {
        "a": _boundary_model(y, preds_a),
        "b": _boundary_model(y, preds_b),
        "n_neutral": int(neutral.sum()),
        "mcnemar_by_seed": per_seed,
    }
    for metric in ("neutral_recall", "negative_precision"):
        out[f"delta_{metric}"] = seed_sentence_bootstrap(
            y, list(preds_a.values()), list(preds_b.values()), metric=metric, B=B, seed=seed
        )
    return out


def build_rq1_boundary(table: RunTable, B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": (
            "R_neu = recall of gold-neutral sentences (predicted neutral / gold neutral); P_neg = precision of "
            "the negative class (gold negative / predicted negative); both are seed means. delta_* = A - B with "
            "the seed+sentence bootstrap. McNemar is restricted to gold-neutral sentences (correct = predicted "
            "neutral), one test per seed pairing; Holm-Bonferroni within each seed family over the pairs of one "
            "config x split block."
        ),
        "configs": {},
    }
    for config in CONFIGS:
        out["configs"][config] = {}
        for split in SPLITS:
            ref = _reference(table, config, split)
            if ref is None:
                continue
            ids, y = ref
            pairs = list(BOUNDARY_PAIRS) + (list(BOUNDARY_PAIRS_EXTERNAL) if split == "external" else [])
            results: dict[str, Any] = {}
            for a, b in pairs:
                preds_a = _model_preds(table, config, a, split, ids)
                preds_b = _model_preds(table, config, b, split, ids)
                if not preds_a or not preds_b:
                    continue
                res = compare_boundary_pair(y, preds_a, preds_b, B, seed)
                if res is not None:
                    res.update({"model_a": a, "model_b": b})
                    results[f"{a}__vs__{b}"] = res
            family_seeds = apply_seed_families(results)
            out["configs"][config][split] = {"n": int(len(y)), "family_seeds": family_seeds, "pairs": results}
    return out


# ---------------------------------------------------------------------------
# attribution.json
# ---------------------------------------------------------------------------


def select_tfidf_best(table: RunTable, config: str) -> tuple[str, dict[str, float]] | None:
    """The TF-IDF model with the higher validation Macro-F1 (test is never consulted)."""
    scores: dict[str, float] = {}
    for model in ("tfidf_logreg", "tfidf_svm"):
        run = table.get(config, model, f"seed{PRIMARY_SEED}")
        if run is not None and run.get("metrics", {}).get("val"):
            scores[model] = float(run["metrics"]["val"]["macro_f1"])
    if not scores:
        return None
    best = max(scores, key=lambda m: (scores[m], m == "tfidf_logreg"))
    return best, scores


def attribution_cell(
    y: np.ndarray, a: list[np.ndarray], b: list[np.ndarray], metric: str, B: int, seed: int
) -> dict[str, Any]:
    """Seed+sentence bootstrap result plus per-seed delta summary and the conditional (old) result."""
    cell = seed_sentence_bootstrap(y, a, b, metric=metric, B=B, seed=seed)
    deltas = cell["per_seed_delta"]
    stats = mean_sd(deltas)
    mean = stats["mean"]
    cell["per_seed_delta_mean"] = mean
    cell["per_seed_delta_sd"] = stats["sd"] if stats["n"] > 1 else None
    cell["n_seeds_same_sign"] = int(sum(1 for d in deltas if d * mean > 0)) if mean else 0
    cell["n_seed_deltas"] = len(deltas)
    cell["conditional_on_seeds"] = seed_averaged_bootstrap(y, a, b, metric=metric, B=B, seed=seed)
    return cell


def build_attribution(table: RunTable, B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": {
            "delta_arch": "bert_base - tfidf_best (tfidf_best chosen by validation Macro-F1)",
            "delta_domain": "finbert_mlm - bert_base",
            "ci": (
                "seed+sentence two-level bootstrap (sentences and trained seeds both resampled); "
                "conditional_on_seeds is the sentence-only bootstrap with the seeds held fixed"
            ),
        },
        "configs": {},
    }
    for config in CONFIGS:
        picked = select_tfidf_best(table, config)
        if picked is None:
            continue
        best, val_scores = picked
        out["configs"][config] = {"tfidf_best": best, "tfidf_val_macro_f1": val_scores}
        for split in SPLITS:
            ref = _reference(table, config, split)
            if ref is None:
                continue
            ids, y = ref
            tf = _model_preds(table, config, best, split, ids)
            bert = _model_preds(table, config, "bert_base", split, ids)
            fin = _model_preds(table, config, "finbert_mlm", split, ids)
            entry: dict[str, Any] = {"n": int(len(y))}
            for name, (a, b) in {"delta_arch": (bert, tf), "delta_domain": (fin, bert)}.items():
                if a and b:
                    entry[name] = {
                        metric: attribution_cell(y, list(a.values()), list(b.values()), metric, B, seed)
                        for metric in ("macro_f1", "accuracy")
                    }
            out["configs"][config][split] = entry
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _no_data(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _val(entry: dict | None, split: str, metric: str = "macro_f1") -> tuple[float, float]:
    if not entry or not entry.get(split) or entry[split][metric]["mean"] is None:
        return float("nan"), 0.0
    return entry[split][metric]["mean"], entry[split][metric]["sd"] or 0.0


def fig_macro_f1_by_config(summary: dict, path: str) -> None:
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(20, 5.5), sharey=True)
    for ax, config in zip(axes, CONFIGS):
        entries = summary["configs"].get(config, {})
        if not entries:
            _no_data(ax, config)
            continue
        vals = [_val(entries.get(m), "test") for m in MODELS]
        ax.bar(range(len(MODELS)), [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3, color="#4c78a8")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([display_name(m) for m in MODELS], rotation=45, ha="right", fontsize=8)
        ax.set_title(config)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("test Macro-F1 (mean ± sd over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_external(summary: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    width = 0.8 / max(1, len(CONFIGS))
    for k, config in enumerate(CONFIGS):
        entries = summary["configs"].get(config, {})
        vals = [_val(entries.get(m), "external") for m in MODELS]
        xs = np.arange(len(MODELS)) + k * width
        ax.bar(xs, [v[0] for v in vals], width, yerr=[v[1] for v in vals], capsize=2, label=f"trained on {config}")
    zs = summary["zeroshot"].get(ZEROSHOT_MODEL)
    if zs and zs.get("external"):
        ax.axhline(zs["external"]["macro_f1"], color="black", linestyle="--", label=f"{display_name(ZEROSHOT_MODEL)}")
    ax.set_xticks(np.arange(len(MODELS)) + width * (len(CONFIGS) - 1) / 2)
    ax.set_xticklabels([display_name(m) for m in MODELS], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SEntFiN 1.0 Macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_attribution(attribution: dict, path: str) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        labels, arch, domain = [], [], []
        for config in CONFIGS:
            entry = attribution["configs"].get(config, {}).get(split)
            if not entry:
                continue
            labels.append(config.replace("sentences_", ""))
            arch.append(entry.get("delta_arch", {}).get("macro_f1"))
            domain.append(entry.get("delta_domain", {}).get("macro_f1"))
        if not labels:
            _no_data(ax, split)
            continue
        x = np.arange(len(labels))
        for offset, series, name, color in ((-0.2, arch, "delta_arch (BERT - TF-IDF)", "#e45756"),
                                            (0.2, domain, "delta_domain (FinBERT-MLM - BERT)", "#54a24b")):
            d = [s["delta"] if s else np.nan for s in series]
            lo = [s["delta"] - s["ci_low"] if s else 0 for s in series]
            hi = [s["ci_high"] - s["delta"] if s else 0 for s in series]
            ax.bar(x + offset, d, 0.4, yerr=[lo, hi], capsize=3, label=name, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{split} (Macro-F1 difference, 95% seed+sentence CI)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_confusions(table: RunTable, path: str) -> None:
    fig, axes = plt.subplots(1, len(TRANSFORMER_MODELS), figsize=(10, 4.5))
    for ax, model in zip(axes, TRANSFORMER_MODELS):
        runs = list(table.seeds(CV_CONFIG, model).values())
        mats = [np.asarray(r["metrics"]["test"]["confusion_matrix"]) for r in runs if r["metrics"].get("test")]
        if not mats:
            _no_data(ax, model)
            continue
        total = np.sum(mats, axis=0)
        ax.imshow(total, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(total[i, j])), ha="center", va="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title(f"{display_name(model)} (test, summed over {len(mats)} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pm(stat: dict | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    if stat.get("n", 0) < 2 or stat.get("sd") is None:
        return f"{stat['mean']:.{digits}f} (sd n/a)"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _p_txt(pv: float, B: int) -> str:
    """p-value text; 0 means no resample crossed zero, i.e. p below the resolution 2 / B."""
    return f"p < {2.0 / B:.4g}" if pv == 0 else f"p={pv:.4f}"


def _ci(boot: dict | None) -> str:
    if not boot:
        return "n/a"
    p_txt = _p_txt(boot["p_value"], boot["B"])
    return f"{boot['delta']:+.3f} [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}], {p_txt}"


def build_report(
    summary: dict, comparisons: dict, attribution: dict, boundary: dict, missing: list[str], args: Any
) -> str:
    lines = ["# Step 2 Report: Transformer Fine-Tuning vs Baselines", ""]
    lines += [f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `{args.runs_dir}`.", ""]
    if missing:
        lines += [f"## Missing runs ({len(missing)})", "",
                  "This report was produced with `--allow-partial`. The following planned runs were absent:", ""]
        lines += [f"- `{m}`" for m in missing] + [""]

    lines += ["## Protocol", "",
              "- Both encoders (`bert_base`, `finbert_mlm`) use one shared fixed recipe (AdamW, linear warmup, "
              "same batch size, learning rate and epoch budget) on the Step 1 locked train split of each config.",
              "- The epoch is selected by validation Macro-F1 only; the test split and SEntFiN are scored once "
              "with the selected checkpoint.",
              "- Fine-tuned models run with several seeds; tables show mean ± sample sd over seeds. Baselines are "
              "deterministic and have the single run `seed42` (sd shown as n/a).",
              "- Baseline fitting: the official lexicons (VADER, Loughran-McDonald) use fixed published rules and "
              "nothing is tuned. The finance-adjusted lexicon variants tune their decision thresholds on the "
              "validation split of each config, and the TF-IDF models tune their hyperparameters on validation per "
              "config.",
              "- The 5-fold CV on `sentences_allagree` is scored on the fold validation part, the same part used "
              "for epoch selection (as in Step 1 CV), so it is an optimistic estimate and not a test score.",
              "- The three PhraseBank levels (allagree, 75agree, 50agree) are nested: a sentence in a stricter "
              "level is also in the looser ones. Models and test splits are never mixed across levels, so tables "
              "are compared within a config and across configs only descriptively.",
              "- `prosus_finbert` (ProsusAI/finbert) is zero-shot and scored only on SEntFiN 1.0 (it was "
              "fine-tuned on Financial PhraseBank, so scoring it there would leak).",
              "- Confidence intervals include seed variation: the bootstrap resamples sentences and, for the "
              "fine-tuned models, the trained seeds (two-level, seed+sentence). The earlier sentence-only "
              "interval, conditional on the three seeds, is stored as `conditional_on_seeds` in `attribution.json`.",
              f"- Bootstrap p-value (B = {args.bootstrap_samples}): p = min(1, 2 * min(P(d <= 0), P(d >= 0))) over "
              f"the resampled differences d, so the smallest non-zero value is 2/B; 0 is printed as "
              f"p < {2.0 / args.bootstrap_samples:.4g}.",
              "- McNemar is a per-seed diagnostic. There is no primary seed: each seed pairing forms its own "
              "family, Holm-Bonferroni is applied within each family (pairs of single-run models enter every "
              "family with their one run), and the report gives p_min, p_max and the number of seed families "
              "with Holm-adjusted p < 0.05. The seed+sentence bootstrap of the Macro-F1 difference is the "
              "headline test.",
              "- RQ1 boundary: R_neu is the recall of gold-neutral sentences and P_neg the precision of the "
              "negative class (seed means; deltas use the seed+sentence bootstrap). The McNemar test in that "
              "section is restricted to gold-neutral sentences (correct = predicted neutral), with Holm within "
              "each seed family over the pairs of one config and split.",
              "- Per-class precision, recall and F1 are the seed mean ± sd of the values stored in each "
              "`result.json` (`metrics.<split>.per_class`).",
              "- Attribution: `delta_arch` = BERT − `tfidf_best` and `delta_domain` = FinBERT-MLM − BERT; "
              "`tfidf_best` is the TF-IDF model with the higher validation Macro-F1 in that config. Per-seed "
              "deltas are listed beside each interval.", ""]

    for split, title in (("test", "In-domain test"), ("external", "SEntFiN 1.0 (external)")):
        lines += [f"## {title}: Macro-F1 / accuracy (mean ± sd)", ""]
        for config in CONFIGS:
            entries = summary["configs"].get(config, {})
            if not entries:
                continue
            lines += [f"### {config}", "", "| model | seeds | Macro-F1 | accuracy | neutral→negative | ECE |",
                      "|---|---|---|---|---|---|"]
            for model in MODELS:
                e = entries.get(model)
                if not e:
                    continue
                s = e[split]
                lines.append(f"| {display_name(model)} | {e['n_runs']} | {_pm(s['macro_f1'])} | {_pm(s['accuracy'])} | "
                             f"{_pm(s['neutral_to_negative_rate'])} | {_pm(s['ece'])} |")
            lines.append("")
    zs = summary["zeroshot"].get(ZEROSHOT_MODEL)
    if zs and zs.get("external"):
        lines += [f"### {display_name(ZEROSHOT_MODEL)} (SEntFiN only)", "",
                  f"Macro-F1 {zs['external']['macro_f1']:.3f}, accuracy {zs['external']['accuracy']:.3f}.", ""]

    if summary["cv"]:
        lines += ["## 5-fold CV on sentences_allagree (fold validation part)", "",
                  "| model | folds | Macro-F1 | accuracy |", "|---|---|---|---|"]
        for model in MODELS:
            e = summary["cv"].get(model)
            if e:
                lines.append(f"| {display_name(model)} | {e['n_folds']} | {_pm(e['val']['macro_f1'])} | {_pm(e['val']['accuracy'])} |")
        lines.append("")

    lines += ["## Per-class precision / recall / F1 (sentences_allagree)", ""]
    for split, title in (("test", "In-domain test"), ("external", "SEntFiN 1.0 (allagree-trained)")):
        entries = summary["configs"].get(CV_CONFIG, {})
        if not entries:
            continue
        header = ["model"] + [f"{name} {stat}" for name in CLASS_NAMES for stat in ("P", "R", "F1")]
        lines += [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for model in MODELS:
            e = entries.get(model)
            if not e or not e[split].get("per_class"):
                continue
            cells = [_pm(e[split]["per_class"][name][stat]) for name in CLASS_NAMES for stat in PER_CLASS_STATS]
            lines.append(f"| {display_name(model)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## RQ1: neutral/negative boundary", "",
              "R_neu = recall of gold-neutral sentences, P_neg = precision of the negative class (seed means); "
              "Δ = A − B with the 95% seed+sentence bootstrap CI. McNemar covers gold-neutral sentences only.", ""]
    for split, title in (("test", "In-domain test"), ("external", "SEntFiN 1.0")):
        for config, per_split in boundary["configs"].items():
            block = per_split.get(split)
            if not block or not block["pairs"]:
                continue
            lines += [f"### {title}: {config} (N = {block['n']})", "",
                      "| pair (A vs B) | R_neu A / B | P_neg A / B | ΔR_neu [95% CI] | ΔP_neg [95% CI] | "
                      "neutral-only McNemar p range | families Holm p < 0.05 |", "|---|---|---|---|---|---|---|"]
            for res in block["pairs"].values():
                lines.append(
                    f"| {display_name(res['model_a'])} vs {display_name(res['model_b'])} | "
                    f"{res['a']['neutral_recall']:.3f} / {res['b']['neutral_recall']:.3f} | "
                    f"{res['a']['negative_precision']:.3f} / {res['b']['negative_precision']:.3f} | "
                    f"{_ci(res['delta_neutral_recall'])} | {_ci(res['delta_negative_precision'])} | "
                    f"{res['p_min']:.3g}–{res['p_max']:.3g} | {res['n_seeds_significant']}/{res['n_families']} |")
            lines.append("")

    lines += ["## Architecture vs domain-pretraining attribution", "",
              "Differences with the 95% seed+sentence bootstrap CI.", "",
              "| config | split | tfidf_best | delta_arch Macro-F1 | delta_arch accuracy | "
              "delta_domain Macro-F1 | delta_domain accuracy |", "|---|---|---|---|---|---|---|"]
    for config, cfg in attribution["configs"].items():
        for split in SPLITS:
            e = cfg.get(split)
            if not e:
                continue
            arch, dom = e.get("delta_arch", {}), e.get("delta_domain", {})
            lines.append(f"| {config} | {split} | {display_name(cfg['tfidf_best'])} | {_ci(arch.get('macro_f1'))} | "
                         f"{_ci(arch.get('accuracy'))} | {_ci(dom.get('macro_f1'))} | {_ci(dom.get('accuracy'))} |")
    lines.append("")

    lines += ["### Per-seed Macro-F1 deltas", "",
              "| config | split | delta | per-seed deltas | mean ± sd | seeds with the mean's sign | "
              "sentence-only CI (seeds fixed) |", "|---|---|---|---|---|---|---|"]
    for config, cfg in attribution["configs"].items():
        for split in SPLITS:
            e = cfg.get(split)
            if not e:
                continue
            for name in ("delta_arch", "delta_domain"):
                cell = e.get(name, {}).get("macro_f1")
                if not cell:
                    continue
                per_seed = ", ".join(f"{d:+.3f}" for d in cell["per_seed_delta"])
                sd = cell["per_seed_delta_sd"]
                spread = f"{cell['per_seed_delta_mean']:+.3f} ± {sd:.3f}" if sd is not None else \
                    f"{cell['per_seed_delta_mean']:+.3f} (sd n/a)"
                cond = cell["conditional_on_seeds"]
                lines.append(f"| {config} | {split} | {name} | {per_seed} | {spread} | "
                             f"{cell['n_seeds_same_sign']}/{cell['n_seed_deltas']} | "
                             f"[{cond['ci_low']:+.3f}, {cond['ci_high']:+.3f}] |")
    lines.append("")

    for split, title in (("test", "In-domain test"), ("external", "SEntFiN 1.0")):
        lines += [f"## Paired comparisons: {title}", ""]
        for config, per_split in comparisons["configs"].items():
            block = per_split.get(split)
            if not block:
                continue
            lines += [f"### {config} (N = {block['n']})", "",
                      "| pair | McNemar p range (min–max) | seed families Holm p < 0.05 | per-seed p | "
                      "seed+sentence bootstrap ΔMacro-F1 [95% CI] |",
                      "|---|---|---|---|---|"]
            for key, res in block["pairs"].items():
                per_seed = ", ".join(f"{s}: {v['p_value']:.3g}" for s, v in res["mcnemar_by_seed"].items())
                b = res["bootstrap_macro_f1"]
                lines.append(f"| {display_name(res['model_a'])} vs {display_name(res['model_b'])} | "
                             f"{res['p_min']:.3g}–{res['p_max']:.3g} | "
                             f"{res['n_seeds_significant']}/{res['n_families']} | {per_seed} | "
                             f"{b['delta']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}], "
                             f"{_p_txt(b['p_value'], b['B'])} |")
            lines.append("")

    lines += ["## Figures", ""] + [f"- `{name}`" for name in FIGURES] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise Step 2 runs.")
    parser.add_argument("--runs-dir", default="results/step2/runs")
    parser.add_argument("--out-dir", default="results/step2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="fine-tuning seeds the plan expects")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.runs_dir):
        print(f"ERROR: runs directory not found: {args.runs_dir}", file=sys.stderr)
        return 2
    table = RunTable(load_runs(args.runs_dir))
    missing = [f"{c}/{m}/{t}" for c, m, t in planned_runs(tuple(args.seeds)) if table.get(c, m, t) is None]
    if missing and not args.allow_partial:
        print(f"ERROR: {len(missing)} planned run(s) missing (use --allow-partial to summarise anyway):", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    summary = build_summary(table)
    comparisons = build_comparisons(table, args.bootstrap_samples, args.seed)
    attribution = build_attribution(table, args.bootstrap_samples, args.seed)
    boundary = build_rq1_boundary(table, args.bootstrap_samples, args.seed)
    summary["missing_runs"] = missing

    def dump(name: str, obj: Any) -> None:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2)

    dump("summary_metrics.json", summary)
    dump("statistical_comparisons.json", comparisons)
    dump("attribution.json", attribution)
    dump("rq1_boundary.json", boundary)

    fig_macro_f1_by_config(summary, os.path.join(args.out_dir, FIGURES[0]))
    fig_external(summary, os.path.join(args.out_dir, FIGURES[1]))
    fig_attribution(attribution, os.path.join(args.out_dir, FIGURES[2]))
    fig_confusions(table, os.path.join(args.out_dir, FIGURES[3]))

    report = build_report(summary, comparisons, attribution, boundary, missing, args)
    with open(os.path.join(args.out_dir, "STEP2_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote summary to {args.out_dir} ({len(table.runs)} runs, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
