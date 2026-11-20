"""Per-phenomenon error rates with the human tags instead of the LLM consensus.

The report states that repeating the Step 3 error analysis with the human annotator's tags
(`annotations_qiyao.csv`) changes individual error rates by at most 10 points. This script
recomputes that from the stored 50agree runs (no model is run): for each model and phenomenon it
gives the raw error rate (1 - accuracy, seed mean per sentence) on the sentences tagged by the
LLM consensus (Step 3 mode `both`) and on the sentences tagged by the human, and writes
`results/step3/human_tag_error_rates.json`.

    python3 scripts/analyze_human_tag_errors.py            # write the JSON and print a table
    python3 scripts/analyze_human_tag_errors.py --check    # recompute, compare, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attribution import PHENOMENA, consensus_tags, read_annotations  # noqa: E402

DIAG_DIR = os.path.join(REPO_ROOT, "data", "diagnostic")
RUNS_DIR = os.path.join(REPO_ROOT, "results", "step2", "runs", "sentences_50agree")
OUT_PATH = os.path.join(REPO_ROOT, "results", "step3", "human_tag_error_rates.json")
MODELS = ("vader_official", "lm_official", "tfidf_svm", "bert_base", "finbert_mlm")


def sentence_errors(model: str, key_items: list[dict]) -> tuple[dict[str, float], list[str]]:
    """Per diagnostic item: share of the model's seeds that got it wrong."""
    model_dir = os.path.join(RUNS_DIR, model)
    seeds = sorted(d for d in os.listdir(model_dir)
                   if d.startswith("seed") and os.path.exists(os.path.join(model_dir, d, "predictions.npz")))
    if not seeds:
        raise FileNotFoundError(f"No seed runs with predictions.npz in {model_dir}")
    wrong: dict[str, list[int]] = {item["diag_id"]: [] for item in key_items}
    for seed in seeds:
        z = np.load(os.path.join(model_dir, seed, "predictions.npz"))
        index = {str(rid): i for i, rid in enumerate(z["test_ids"])}
        for item in key_items:
            i = index[item["record_id"]]
            if int(z["test_y"][i]) != int(item["label"]):
                raise ValueError(f"Gold label mismatch for {item['diag_id']} in {model}/{seed}")
            wrong[item["diag_id"]].append(int(int(np.argmax(z["test_prob"][i])) != int(item["label"])))
    return {d: float(np.mean(v)) for d, v in wrong.items()}, seeds


def compute() -> dict:
    with open(os.path.join(DIAG_DIR, "diagnostic_key.json"), encoding="utf-8") as handle:
        key_items = json.load(handle)["items"]
    llm = consensus_tags(read_annotations(os.path.join(DIAG_DIR, "annotations_a.csv")),
                         read_annotations(os.path.join(DIAG_DIR, "annotations_b.csv")), mode="both")
    human = read_annotations(os.path.join(DIAG_DIR, "annotations_qiyao.csv"))
    ids = [item["diag_id"] for item in key_items]
    models: dict[str, dict] = {}
    max_diff = 0.0
    for model in MODELS:
        err, seeds = sentence_errors(model, key_items)
        rows = {}
        for ph in PHENOMENA:
            tagged_llm = [d for d in ids if llm[d][ph]]
            tagged_h = [d for d in ids if human[d][ph]]
            e_llm = float(np.mean([err[d] for d in tagged_llm]))
            e_h = float(np.mean([err[d] for d in tagged_h]))
            diff = abs(e_llm - e_h)
            max_diff = max(max_diff, diff)
            rows[ph] = {"n_llm_consensus": len(tagged_llm), "n_human": len(tagged_h),
                        "error_llm_consensus": round(e_llm, 6), "error_human": round(e_h, 6),
                        "abs_diff_points": round(100 * diff, 2)}
        models[model] = {"seeds": seeds, "error_all": round(float(np.mean([err[d] for d in ids])), 6),
                         "phenomena": rows}
    return {"config": "sentences_50agree", "n_items": len(ids),
            "note": ("Raw error rate (1 - accuracy, seed mean per sentence) on the sentences carrying each tag; "
                     "LLM consensus = annotators A and B both mark the tag (Step 3 mode 'both'); human = "
                     "annotations_qiyao.csv."),
            "max_abs_diff_points": round(100 * max_diff, 2), "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="recompute and compare with the stored JSON")
    args = parser.parse_args()
    result = compute()
    for model, m in result["models"].items():
        cells = "; ".join(f"{ph[:8]} {r['error_llm_consensus']*100:.1f}/{r['error_human']*100:.1f}"
                          for ph, r in m["phenomena"].items())
        print(f"{model:15s} ({len(m['seeds'])} seeds) LLM/human error %: {cells}")
    print(f"max abs difference: {result['max_abs_diff_points']} points")
    if args.check:
        with open(OUT_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored != result:
            print("CHECK FAILED: stored JSON differs from the recomputation", file=sys.stderr)
            return 1
        print("CHECK OK")
        return 0
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("wrote", os.path.relpath(OUT_PATH, REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
