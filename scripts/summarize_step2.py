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

from src.evaluation.metrics import CLASS_NAMES  # noqa: E402
from src.evaluation.multiseed import (  # noqa: E402
    holm_bonferroni,
    load_runs,
    mean_sd,
    seed_averaged_bootstrap,
)
from src.evaluation.stats import mcnemar_test  # noqa: E402

CONFIGS = ("sentences_allagree", "sentences_75agree", "sentences_50agree")
BASELINE_MODELS = ("vader", "loughran_mcdonald", "tfidf_logreg", "tfidf_svm")
TRANSFORMER_MODELS = ("bert_base", "finbert_mlm")
MODELS = BASELINE_MODELS + TRANSFORMER_MODELS
ZEROSHOT_MODEL = "prosus_finbert"
ZEROSHOT_CONFIG = "external_only"
CV_CONFIG = "sentences_allagree"
PRIMARY_SEED = 42
DEFAULT_SEEDS = (42, 100, 2026)
CV_FOLDS = (0, 1, 2, 3, 4)
SPLITS = ("test", "external")

FIGURES = (
    "step2_macro_f1_by_config.png",
    "step2_external_generalization.png",
    "step2_attribution.png",
    "step2_confusion_matrices_allagree.png",
)


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
    primary_seed = PRIMARY_SEED if PRIMARY_SEED in seeds else seeds[0]
    boot = seed_averaged_bootstrap(
        y, list(preds_a.values()), list(preds_b.values()), metric="macro_f1", B=B, seed=seed
    )
    return {
        "mcnemar_by_seed": per_seed,
        "primary_seed": primary_seed,
        "p_primary": per_seed[str(primary_seed)]["p_value"],
        "bootstrap_macro_f1": boot,
        "n_seeds_a": len(preds_a),
        "n_seeds_b": len(preds_b),
    }


def build_comparisons(table: RunTable, B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": (
            "McNemar per seed; p_primary is the seed-42 pairing; p_holm is Holm-Bonferroni over the "
            "pairs listed for that config and split (15 pairs on test; 21 on external, the 15 plus the "
            "6 prosus_finbert pairs)."
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
            holm = holm_bonferroni({k: v["p_primary"] for k, v in results.items()})
            for key, p_holm in holm.items():
                results[key]["p_holm"] = p_holm
            out["configs"][config][split] = {"n": int(len(y)), "pairs": results}
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


def build_attribution(table: RunTable, B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": {
            "delta_arch": "bert_base - tfidf_best (tfidf_best chosen by validation Macro-F1)",
            "delta_domain": "finbert_mlm - bert_base",
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
                        metric: seed_averaged_bootstrap(
                            y, list(a.values()), list(b.values()), metric=metric, B=B, seed=seed
                        )
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
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(15, 4.5), sharey=True)
    for ax, config in zip(axes, CONFIGS):
        entries = summary["configs"].get(config, {})
        if not entries:
            _no_data(ax, config)
            continue
        vals = [_val(entries.get(m), "test") for m in MODELS]
        ax.bar(range(len(MODELS)), [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3, color="#4c78a8")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right")
        ax.set_title(config)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("test Macro-F1 (mean ± sd over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_external(summary: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(1, len(CONFIGS))
    for k, config in enumerate(CONFIGS):
        entries = summary["configs"].get(config, {})
        vals = [_val(entries.get(m), "external") for m in MODELS]
        xs = np.arange(len(MODELS)) + k * width
        ax.bar(xs, [v[0] for v in vals], width, yerr=[v[1] for v in vals], capsize=2, label=f"trained on {config}")
    zs = summary["zeroshot"].get(ZEROSHOT_MODEL)
    if zs and zs.get("external"):
        ax.axhline(zs["external"]["macro_f1"], color="black", linestyle="--", label="prosus_finbert (zero-shot)")
    ax.set_xticks(np.arange(len(MODELS)) + width * (len(CONFIGS) - 1) / 2)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
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
        ax.set_title(f"{split} (Macro-F1 difference, 95% CI)")
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
        ax.set_title(f"{model} (test, summed over {len(mats)} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pm(stat: dict | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _ci(boot: dict | None) -> str:
    if not boot:
        return "n/a"
    return f"{boot['delta']:+.3f} [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}], p={boot['p_value']:.4f}"


def build_report(summary: dict, comparisons: dict, attribution: dict, missing: list[str], args: Any) -> str:
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
              "deterministic and have the single run `seed42`.",
              "- The 5-fold CV on `sentences_allagree` is scored on the fold validation part, the same part used "
              "for epoch selection (as in Step 1 CV), so it is an optimistic estimate and not a test score.",
              "- `prosus_finbert` is scored only on SEntFiN 1.0 (it was fine-tuned on Financial PhraseBank, so "
              "scoring it there would leak).",
              "- Paired tests: McNemar per seed (primary p = seed 42 pairing), Holm-Bonferroni over the pairs of "
              "one config and split, and a paired sentence bootstrap of the seed-averaged Macro-F1 difference "
              f"(B = {args.bootstrap_samples}, 95% percentile CI).",
              "- Attribution: `delta_arch` = `bert_base` − `tfidf_best` and `delta_domain` = `finbert_mlm` − "
              "`bert_base`; `tfidf_best` is the TF-IDF model with the higher validation Macro-F1 in that config.", ""]

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
                lines.append(f"| {model} | {e['n_runs']} | {_pm(s['macro_f1'])} | {_pm(s['accuracy'])} | "
                             f"{_pm(s['neutral_to_negative_rate'])} | {_pm(s['ece'])} |")
            lines.append("")
    zs = summary["zeroshot"].get(ZEROSHOT_MODEL)
    if zs and zs.get("external"):
        lines += ["### prosus_finbert (zero-shot, SEntFiN only)", "",
                  f"Macro-F1 {zs['external']['macro_f1']:.3f}, accuracy {zs['external']['accuracy']:.3f}.", ""]

    if summary["cv"]:
        lines += ["## 5-fold CV on sentences_allagree (fold validation part)", "",
                  "| model | folds | Macro-F1 | accuracy |", "|---|---|---|---|"]
        for model in MODELS:
            e = summary["cv"].get(model)
            if e:
                lines.append(f"| {model} | {e['n_folds']} | {_pm(e['val']['macro_f1'])} | {_pm(e['val']['accuracy'])} |")
        lines.append("")

    lines += ["## Architecture vs domain-pretraining attribution", "",
              "| config | split | tfidf_best | delta_arch Macro-F1 | delta_arch accuracy | "
              "delta_domain Macro-F1 | delta_domain accuracy |", "|---|---|---|---|---|---|---|"]
    for config, cfg in attribution["configs"].items():
        for split in SPLITS:
            e = cfg.get(split)
            if not e:
                continue
            arch, dom = e.get("delta_arch", {}), e.get("delta_domain", {})
            lines.append(f"| {config} | {split} | {cfg['tfidf_best']} | {_ci(arch.get('macro_f1'))} | "
                         f"{_ci(arch.get('accuracy'))} | {_ci(dom.get('macro_f1'))} | {_ci(dom.get('accuracy'))} |")
    lines.append("")

    for split, title in (("test", "In-domain test"), ("external", "SEntFiN 1.0")):
        lines += [f"## Paired comparisons: {title}", ""]
        for config, per_split in comparisons["configs"].items():
            block = per_split.get(split)
            if not block:
                continue
            lines += [f"### {config} (N = {block['n']})", "",
                      "| pair | McNemar p (seed 42) | Holm p | per-seed p | bootstrap ΔMacro-F1 [95% CI] |",
                      "|---|---|---|---|---|"]
            for key, res in block["pairs"].items():
                per_seed = ", ".join(f"{s}: {v['p_value']:.3g}" for s, v in res["mcnemar_by_seed"].items())
                b = res["bootstrap_macro_f1"]
                lines.append(f"| {res['model_a']} vs {res['model_b']} | {res['p_primary']:.3g} | {res['p_holm']:.3g} | "
                             f"{per_seed} | {b['delta']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] |")
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
    summary["missing_runs"] = missing

    def dump(name: str, obj: Any) -> None:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2)

    dump("summary_metrics.json", summary)
    dump("statistical_comparisons.json", comparisons)
    dump("attribution.json", attribution)

    fig_macro_f1_by_config(summary, os.path.join(args.out_dir, FIGURES[0]))
    fig_external(summary, os.path.join(args.out_dir, FIGURES[1]))
    fig_attribution(attribution, os.path.join(args.out_dir, FIGURES[2]))
    fig_confusions(table, os.path.join(args.out_dir, FIGURES[3]))

    report = build_report(summary, comparisons, attribution, missing, args)
    with open(os.path.join(args.out_dir, "STEP2_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote summary to {args.out_dir} ({len(table.runs)} runs, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
