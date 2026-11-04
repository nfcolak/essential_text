"""SEntFiN 1.0 by number of named entities: single-entity headlines vs multi-entity headlines.

The proposal filters SEntFiN 1.0 to "unambiguous single-entity" headlines. The pipeline keeps every headline whose
entities share one polarity (9,460 headlines), so 1,607 of them name two or more entities. This script re-scores the
stored Step 2 predictions (no model is refit) on three subsets:

- ``all``: the 9,460 headlines of the headline tables;
- ``single``: headlines with exactly one entity (the proposal's definition);
- ``multi``: headlines with two or more entities that all carry the same label.

For every config x model it reports seed mean +- sd of Macro-F1, accuracy and the neutral-to-negative rate, the
architecture / domain-pretraining attribution on each subset (same seed+sentence bootstrap as
``scripts/summarize_step2.py``) and the RQ1 financial-terminology comparison on single-entity headlines, with and
without the loss-only headlines.

Writes ``<out-dir>/sentfin_entity_subsets.json``; prints a short table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts import summarize_step2 as summ  # noqa: E402
from src.analysis.attribution import loss_only_ids, rq1_terminology_comparison  # noqa: E402
from src.analysis.phenomena import financial_terms, tag_many  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.evaluation.multiseed import load_runs, mean_sd  # noqa: E402

SUBSETS = ("all", "single", "multi")
DEFAULT_RUNS_DIR = os.path.join("results", "step2", "runs")
DEFAULT_SPLITS_DIR = os.path.join("data", "splits")
DEFAULT_OUT_DIR = os.path.join("results", "extensions")


def entity_counts(sentfin_path: str) -> dict[str, int]:
    """Holdout id -> number of named entities in the SEntFiN 1.0 ``Decisions`` field."""
    with open(sentfin_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(r["id"]): len(r.get("entities") or {}) for r in payload["records"]}


def subset_masks(ids: np.ndarray, counts: dict[str, int]) -> dict[str, np.ndarray]:
    missing = [i for i in ids.tolist() if i not in counts]
    if missing:
        raise ValueError(f"{len(missing)} holdout ids have no entity count (e.g. {missing[0]})")
    n_ent = np.array([counts[i] for i in ids.tolist()])
    if (n_ent < 1).any():
        raise ValueError("every SEntFiN holdout headline must name at least one entity")
    return {"all": np.ones(len(ids), dtype=bool), "single": n_ent == 1, "multi": n_ent >= 2}


def _subset_metrics(y: np.ndarray, preds: dict[int, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    per_seed = [compute_classification_metrics(y[mask], p[mask]) for p in preds.values()]
    return {
        "n": int(mask.sum()),
        "n_seeds": len(per_seed),
        "macro_f1": mean_sd([m["macro_f1"] for m in per_seed]),
        "accuracy": mean_sd([m["accuracy"] for m in per_seed]),
        "neutral_to_negative_rate": mean_sd([m["neutral_to_negative_rate"] for m in per_seed]),
    }


def build(table: summ.RunTable, counts: dict[str, int], texts: dict[str, str], B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": {
            "single": "SEntFiN 1.0 headline with exactly one named entity (proposal: unambiguous single-entity)",
            "multi": "two or more named entities, all with the same label (kept by the pipeline, not by the proposal)",
            "delta_arch": "bert_base - tfidf_best (tfidf_best chosen by validation Macro-F1)",
            "delta_domain": "finbert_mlm - bert_base",
            "ci": "seed+sentence two-level bootstrap, as in results/step2/attribution.json",
        },
        "bootstrap": {"B": B, "seed": seed},
        "configs": {},
    }
    for config in summ.CONFIGS:
        ref = summ._reference(table, config, "external")
        if ref is None:
            continue
        ids, y = ref
        masks = subset_masks(ids, counts)
        out.setdefault("n", {s: int(m.sum()) for s, m in masks.items()})
        picked = summ.select_tfidf_best(table, config)
        entry: dict[str, Any] = {"tfidf_best": picked[0] if picked else None, "label_mix": {}, "models": {},
                                 "attribution": {}}
        for subset, mask in masks.items():
            entry["label_mix"][subset] = {name: int((y[mask] == k).sum())
                                          for k, name in enumerate(("negative", "neutral", "positive"))}
        preds_by_model: dict[str, dict[int, np.ndarray]] = {}
        for model in summ.MODELS + (summ.ZEROSHOT_MODEL,):
            preds = summ._model_preds(table, config, model, "external", ids)
            if preds:
                preds_by_model[model] = preds
                entry["models"][model] = {s: _subset_metrics(y, preds, m) for s, m in masks.items()}
        if picked:
            tf = preds_by_model.get(picked[0], {})
            bert = preds_by_model.get("bert_base", {})
            fin = preds_by_model.get("finbert_mlm", {})
            for subset in ("single", "multi"):
                mask = masks[subset]
                cell: dict[str, Any] = {"n": int(mask.sum())}
                for name, (a, b) in {"delta_arch": (bert, tf), "delta_domain": (fin, bert)}.items():
                    if a and b:
                        cell[name] = {
                            metric: summ.attribution_cell(y[mask], [p[mask] for p in a.values()],
                                                          [p[mask] for p in b.values()], metric, B, seed)
                            for metric in ("macro_f1", "accuracy")
                        }
                entry["attribution"][subset] = cell
        # RQ1 on single-entity headlines (rule tags, as in results/step3 section 5)
        single_ids = [i for i, keep in zip(ids.tolist(), masks["single"].tolist()) if keep]
        tags = dict(zip(single_ids, tag_many([texts[i] for i in single_ids])))
        y_true = {i: int(v) for i, v in zip(ids.tolist(), y.tolist())}
        seed_dicts = {m: [dict(zip(ids.tolist(), p.tolist())) for p in preds.values()]
                      for m, preds in preds_by_model.items()}
        cues = {i: financial_terms(texts[i]) for i in single_ids}
        neutral_term = [i for i in single_ids if y_true[i] == 1 and tags[i]["financial_terminology"] == 1]
        loss_only = loss_only_ids({i: cues[i] for i in neutral_term})
        entry["rq1_single_entity"] = {
            "n_neutral_term": len(neutral_term),
            "n_neutral_term_loss_only": len(loss_only),
            "with_loss": rq1_terminology_comparison(tags, y_true, seed_dicts, B=B, seed=seed),
            "without_loss_only": rq1_terminology_comparison(tags, y_true, seed_dicts, B=B, seed=seed,
                                                            drop_ids=loss_only),
        }
        out["configs"][config] = entry
    return out


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _print_table(result: dict[str, Any]) -> None:
    print("n:", result.get("n"))
    for config, entry in result["configs"].items():
        print(f"\n{config}  (Macro-F1 seed mean: all / single / multi)")
        for model, subsets in entry["models"].items():
            vals = " / ".join(f"{subsets[s]['macro_f1']['mean']:.3f}" for s in SUBSETS)
            print(f"  {model:22s} {vals}")
        for subset, cell in entry["attribution"].items():
            for name in ("delta_arch", "delta_domain"):
                if name in cell:
                    c = cell[name]["macro_f1"]
                    print(f"  {name} {subset:6s} {c['delta']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sentfin_path = os.path.join(args.splits_dir, "sentfin_dedup.json")
    counts = entity_counts(sentfin_path)
    with open(sentfin_path, "r", encoding="utf-8") as handle:
        texts = {str(r["id"]): r["text"] for r in json.load(handle)["records"]}
    table = summ.RunTable(load_runs(args.runs_dir))
    result = build(table, counts, texts, args.bootstrap_samples, args.seed)
    if not result["configs"]:
        print(f"no runs with external predictions under {args.runs_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "sentfin_entity_subsets.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(result), handle, indent=2)
        handle.write("\n")
    _print_table(result)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
