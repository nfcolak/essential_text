#!/usr/bin/env python3
"""Compare local zero-shot predictions with paired, fixed-recipe transformer runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_llm_zeroshot import DATASETS, DEFAULT_OUT, NOTE, PHRASEBANK
from src.evaluation.metrics import CLASS_NAMES, compute_classification_metrics
from src.evaluation.multiseed import holm_bonferroni, seed_sentence_bootstrap
from src.evaluation.stats import mcnemar_test

MODELS = ("bert_base", "finbert_mlm")


def read_predictions(directory, split):
    path = directory / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        ids = archive[split + "_ids"]
        y = archive[split + "_y"]
        prob = archive[split + "_prob"]
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Empty, duplicate or malformed ids in {path} ({split})")
    if y.shape != ids.shape or not np.isin(y, range(len(CLASS_NAMES))).all():
        raise ValueError(f"Invalid gold labels in {path} ({split})")
    if prob.shape != (len(ids), len(CLASS_NAMES)) or not np.isfinite(prob).all():
        raise ValueError(f"Invalid prediction matrix in {path} ({split})")
    return ids, y, prob


def summarize_dataset(name, args):
    directory = args.out / name
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    split = "test" if name in PHRASEBANK else "external"
    ids, y, prob = read_predictions(directory, split)
    if result["n"] != len(ids) or not 0 <= result["n_invalid"] <= len(ids):
        raise ValueError(f"Result counts differ from the predictions for {name}")
    pred = prob.argmax(axis=1)
    metrics = compute_classification_metrics(y, pred, prob)
    entry = {"n": len(ids), "n_invalid": result["n_invalid"], "model": result["model"],
             "digest": result["digest"], "metrics": metrics, "comparisons": {}}
    if name in PHRASEBANK:
        entry["contamination_note"] = NOTE
    run_root = args.step2_dir
    ref_split = split
    if name in ("fiqa", "twitter"):
        run_root = args.extensions_dir
        ref_split = name
    config = PHRASEBANK.get(name, "sentences_allagree")
    for model in MODELS:
        model_dir = run_root / config / model
        runs = sorted((p for p in model_dir.glob("seed*") if p.name[4:].isdigit()),
                      key=lambda p: int(p.name[4:]))
        if not runs:
            raise ValueError(f"No fixed-recipe seed runs in {model_dir}")
        ref_preds = []
        per_seed = {}
        for run in runs:
            ref_ids, ref_y, ref_prob = read_predictions(run, ref_split)
            if not np.array_equal(ids, ref_ids):
                raise ValueError(f"{name}: id/order mismatch with {run} ({ref_split})")
            if not np.array_equal(y, ref_y):
                raise ValueError(f"{name}: gold label mismatch with {run} ({ref_split})")
            ref_pred = ref_prob.argmax(axis=1)
            ref_preds.append(ref_pred)
            per_seed[run.name[4:]] = mcnemar_test(y, pred, ref_pred)
        bootstrap = seed_sentence_bootstrap(y, [pred], ref_preds, B=args.bootstrap, seed=args.seed)
        # The shared helper's per_seed_delta enumerates the first arm (one LLM run).
        # Only its aggregate delta, interval and p-value apply to the full comparison here.
        bootstrap.pop("per_seed_delta")
        entry["comparisons"][model] = {"bootstrap_macro_f1": bootstrap, "mcnemar_by_seed": per_seed}

    comparisons = entry["comparisons"]
    seeds = comparisons[MODELS[0]]["mcnemar_by_seed"].keys()
    if seeds != comparisons[MODELS[1]]["mcnemar_by_seed"].keys():
        raise ValueError(f"{name}: reference models have different fine-tuning seed sets")
    for seed in seeds:
        adjusted = holm_bonferroni({m: comparisons[m]["mcnemar_by_seed"][seed]["p_value"] for m in MODELS})
        for model in MODELS:
            comparisons[model]["mcnemar_by_seed"][seed]["p_holm"] = adjusted[model]
    return entry


def render(summary):
    lines = ["# Zero-shot financial sentiment classification", "",
             "| Dataset | N | Macro-F1 | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Invalid |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in summary["datasets"].items():
        metrics = entry["metrics"]
        scores = [metrics["macro_f1"], metrics["accuracy"]] + [metrics["per_class"][c]["f1"] for c in CLASS_NAMES]
        cells = [name, str(entry["n"])] + [f"{v:.4f}" for v in scores] + [str(entry["n_invalid"])]
        lines.append("| " + " | ".join(cells) + " |")
    if any(name in PHRASEBANK for name in summary["datasets"]):
        lines.extend(["", "For the PhraseBank test sets, " + NOTE + "."])
    lines.extend(["", "## Comparisons with fixed-recipe transformers", "",
                  "Δ is the zero-shot Macro-F1 minus the mean fine-tuned Macro-F1. "
                  f"The 95% intervals use {summary['bootstrap_resamples']} paired sentence and seed resamples. "
                  "Each McNemar test pairs the same zero-shot run with one fine-tuning seed. "
                  "Holm adjustment covers the two reference models within each dataset and seed.", "",
                  "| Dataset | Reference | Δ Macro-F1 | 95% interval | McNemar p / Holm p by seed |",
                  "|---|---|---:|---|---|"])
    for name, entry in summary["datasets"].items():
        for model, comparison in entry["comparisons"].items():
            boot = comparison["bootstrap_macro_f1"]
            tests = ", ".join(f"{s} ({v['p_value']:.4g} / {v['p_holm']:.4g})"
                              for s, v in comparison["mcnemar_by_seed"].items())
            lines.append(f"| {name} | {model} | {boot['delta']:+.4f} | "
                         f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | {tests} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--step2-dir", type=Path, default=Path("results/step2/runs"))
    parser.add_argument("--extensions-dir", type=Path, default=Path("results/extensions/runs"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--bootstrap", "--B", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("bootstrap resamples must be positive")
    try:
        summary = {"bootstrap_resamples": args.bootstrap, "bootstrap_seed": args.seed,
                   "holm_family": "both reference models within dataset and fine-tuning seed",
                   "datasets": {name: summarize_dataset(name, args) for name in args.datasets}}
        outputs = {"summary.json": json.dumps(summary, indent=2) + "\n", "REPORT.md": render(summary)}
        for filename, content in outputs.items():
            path = args.out / filename
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != content:
                    print(f"check failed: {path}", file=sys.stderr)
                    return 1
            else:
                path.write_text(content, encoding="utf-8")
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Checked" if args.check else "Wrote", args.out / "summary.json", "and", args.out / "REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
