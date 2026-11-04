"""Summarise the extra external test sets (FiQA 2018, Twitter Financial News) from ``results/extensions/runs``.

Reads the runs written by ``scripts/run_extra_external.py`` and reports, per training config and model, the seed mean
+- sd of Macro-F1 and accuracy on FiQA (all, headlines, posts), Twitter and, for reference, the rerun PhraseBank test
split and SEntFiN. The attribution cells (``Δarch = BERT - best TF-IDF``, ``Δdomain = FinBERT-MLM - BERT``) use the
same seed+sentence bootstrap as ``results/step2/attribution.json``; the TF-IDF model is chosen by validation Macro-F1.
It also summarises how closely the reruns reproduce the stored Step 2 runs.

Writes ``<out-dir>/extra_external.json`` and prints a table.
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
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.evaluation.multiseed import mean_sd  # noqa: E402

SPLITS = ("test", "external", "fiqa", "twitter")
# reported sets: (name, split, format filter)
SETS = (("phrasebank_test", "test", None), ("sentfin", "external", None), ("fiqa", "fiqa", None),
        ("fiqa_headline", "fiqa", "headline"), ("fiqa_post", "fiqa", "post"), ("twitter", "twitter", None))


def load_runs(runs_dir: str) -> list[dict[str, Any]]:
    """Like ``src.evaluation.multiseed.load_runs`` but keeps every split stored in ``predictions.npz``."""
    runs = []
    for config in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        for model in sorted(os.listdir(os.path.join(runs_dir, config))) if os.path.isdir(
                os.path.join(runs_dir, config)) else []:
            for tag in sorted(os.listdir(os.path.join(runs_dir, config, model))):
                run_dir = os.path.join(runs_dir, config, model, tag)
                path = os.path.join(run_dir, "result.json")
                if tag.endswith(".tmp") or not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
                if result.get("status") != "completed":
                    continue
                preds = {}
                with np.load(os.path.join(run_dir, "predictions.npz"), allow_pickle=False) as data:
                    for split in ("val",) + SPLITS:
                        if f"{split}_prob" in data.files:
                            preds[split] = {"ids": np.asarray(data[f"{split}_ids"]).astype(str),
                                            "y": np.asarray(data[f"{split}_y"], dtype=np.int64),
                                            "prob": np.asarray(data[f"{split}_prob"], dtype=np.float32)}
                result.update(run_dir=run_dir, config=config, model_key=model, tag=tag, predictions=preds)
                runs.append(result)
    return runs


def formats(extra_path: str) -> dict[str, str]:
    with open(extra_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {r["id"]: r.get("format", "") for recs in payload["datasets"].values() for r in recs["records"]}


def _scores(y: np.ndarray, preds: dict[int, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    per = [compute_classification_metrics(y[mask], p[mask]) for p in preds.values()]
    return {"n": int(mask.sum()), "n_seeds": len(per), "macro_f1": mean_sd([m["macro_f1"] for m in per]),
            "accuracy": mean_sd([m["accuracy"] for m in per]),
            "per_class_f1": {c: mean_sd([m["per_class"][c]["f1"] for m in per]) for c in summ.CLASS_NAMES}}


def build(table: summ.RunTable, fmt: dict[str, str], B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"definition": {"delta_arch": "bert_base - tfidf_best (by validation Macro-F1)",
                                          "delta_domain": "finbert_mlm - bert_base",
                                          "ci": "seed+sentence two-level bootstrap"},
                           "bootstrap": {"B": B, "seed": seed}, "configs": {}}
    for config in summ.CONFIGS:
        picked = summ.select_tfidf_best(table, config)
        entry: dict[str, Any] = {"tfidf_best": picked[0] if picked else None, "sets": {}}
        for name, split, want in SETS:
            ref = summ._reference(table, config, split)
            if ref is None:
                continue
            ids, y = ref
            mask = np.ones(len(ids), dtype=bool) if want is None else np.array([fmt.get(i) == want for i in ids])
            cell: dict[str, Any] = {"n": int(mask.sum()),
                                    "label_mix": {c: int((y[mask] == k).sum()) for k, c in enumerate(summ.CLASS_NAMES)},
                                    "models": {}, "attribution": {}}
            preds = {}
            for model in summ.MODELS + ("bert_dapt", summ.ZEROSHOT_MODEL):
                p = summ._model_preds(table, config, model, split, ids)
                if p:
                    preds[model] = p
                    cell["models"][model] = _scores(y, p, mask)
            if picked:
                tf, bert, fin = preds.get(picked[0], {}), preds.get("bert_base", {}), preds.get("finbert_mlm", {})
                dapt = preds.get("bert_dapt", {})
                contrasts = {"delta_arch": (bert, tf), "delta_domain": (fin, bert),
                             # controlled DAPT arm: text effect with BERT's vocabulary, and the remainder
                             "delta_corpus": (dapt, bert), "delta_vocab_budget": (fin, dapt)}
                for dname, (a, b) in contrasts.items():
                    if a and b:
                        c = summ.attribution_cell(y[mask], [v[mask] for v in a.values()],
                                                  [v[mask] for v in b.values()], "macro_f1", B, seed)
                        cell["attribution"][dname] = {k: c[k] for k in ("delta", "ci_low", "ci_high", "p_value", "B",
                                                                         "per_seed_delta", "n_seeds_same_sign")}
            entry["sets"][name] = cell
        out["configs"][config] = entry
    # reproduction of the stored Step 2 runs
    rep: dict[str, Any] = {}
    for run in table.runs:
        r = run.get("reproduction_of_step2") or {}
        for split in ("test", "external"):
            if isinstance(r.get(split), dict) and "label_agreement" in r[split]:
                rep.setdefault(run["model_key"], {}).setdefault(split, []).append(
                    (r[split]["label_agreement"], r[split]["max_abs_prob_diff"]))
    out["reproduction"] = {m: {s: {"runs": len(v), "min_label_agreement": min(a for a, _ in v),
                                   "mean_label_agreement": float(np.mean([a for a, _ in v])),
                                   "max_abs_prob_diff": max(d for _, d in v)} for s, v in d.items()}
                           for m, d in rep.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=os.path.join("results", "extensions", "runs"))
    parser.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    parser.add_argument("--out-dir", default=os.path.join("results", "extensions"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    table = summ.RunTable(load_runs(args.runs_dir))
    if not table.runs:
        print(f"no completed runs under {args.runs_dir}", file=sys.stderr)
        return 2
    result = build(table, formats(os.path.join(args.splits_dir, "extra_external.json")),
                   args.bootstrap_samples, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "extra_external.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    for config, entry in result["configs"].items():
        print(f"\n{config} (Macro-F1 seed mean; tfidf_best = {entry['tfidf_best']})")
        names = list(entry["sets"])
        print("  " + " " * 22 + " ".join(f"{n[:13]:>13s}" for n in names))
        models = sorted({m for s in entry["sets"].values() for m in s["models"]},
                        key=list(summ.MODELS + ("bert_dapt", summ.ZEROSHOT_MODEL)).index)
        for m in models:
            vals = [entry["sets"][n]["models"].get(m) for n in names]
            print(f"  {m:22s}" + " ".join(f"{v['macro_f1']['mean']:13.3f}" if v else " " * 13 for v in vals))
        for d in ("delta_arch", "delta_domain", "delta_corpus", "delta_vocab_budget"):
            vals = [entry["sets"][n]["attribution"].get(d) for n in names]
            print(f"  {d:22s}" + " ".join(f"{v['delta']:+6.3f}[{v['ci_low']:+.2f},{v['ci_high']:+.2f}]"[:13].rjust(13)
                                          if v else " " * 13 for v in vals))
    print("\nreproduction:", json.dumps(result["reproduction"], indent=1))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
