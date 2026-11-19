"""Agreement between the human annotation pass and the two LLM annotators (diagnostic set).

Reads `annotations_a.csv`, `annotations_b.csv` (LLM annotators), `annotations_qiyao.csv` (human
annotator c), `rule_tags.csv` and `annotators.json` from the diagnostic directory and writes
`results/step3/human_annotation_agreement.json`: per phenomenon Cohen's kappa and raw agreement of
the human tags against annotator A, annotator B, the A-B consensus (Step 3 mode `both`) and the rule
tagger. The Step 3 analysis itself is not changed by this script; it still uses the A-B consensus.

    python3 scripts/analyze_human_annotation.py            # write the JSON and print a table
    python3 scripts/analyze_human_annotation.py --check    # recompute, compare, write nothing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from typing import Any

from sklearn.metrics import cohen_kappa_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attribution import PHENOMENA, consensus_tags, read_annotations  # noqa: E402

HUMAN_FILE = "annotations_qiyao.csv"
PAIRS = ("h_a", "h_b", "h_consensus", "h_rule")
NOTE = ("Annotators a and b are LLM agents; c is a human team member. The Step 3 analysis still uses "
        "the A-B consensus; this file only measures how well the LLM tags match a human pass.")
TOLERANCE = 1e-12


def _kappa(x: list[int], y: list[int]) -> float | None:
    """Cohen's kappa (scikit-learn); None when undefined (both columns constant)."""
    if len(set(x)) <= 1 and len(set(y)) <= 1:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(cohen_kappa_score(x, y, labels=[0, 1]))
    return value if math.isfinite(value) else None


def _raw(x: list[int], y: list[int]) -> float:
    return sum(1 for p, q in zip(x, y) if p == q) / len(x)


def _prevalence(x: list[int]) -> float:
    return sum(x) / len(x)


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def compute(a: dict, b: dict, h: dict, rule: dict) -> dict[str, Any]:
    """Agreement of the human tags `h` with annotators `a`, `b`, their consensus and the rule tags.

    Every argument maps `diag_id -> {phenomenon: 0/1}`. The four id sets must be identical and
    non-empty, otherwise ValueError. Consensus is 1 only when a and b are both 1 (Step 3 `both`).
    """
    sources = {"a": a, "b": b, "h": h, "rule": rule}
    reference = set(h)
    if not reference:
        raise ValueError("No items: the human annotation is empty.")
    for name, tags in sources.items():
        if set(tags) != reference:
            only_here = sorted(set(tags) - reference)[:5]
            only_human = sorted(reference - set(tags))[:5]
            raise ValueError(
                f"diag_ids of '{name}' do not match the human annotation: "
                f"only in {name} {only_here}, only in h {only_human}.")
    ids = sorted(reference)
    consensus = consensus_tags(a, b, "both")
    columns = {"a": a, "b": b, "consensus": consensus, "rule": rule}
    agreement: dict[str, Any] = {}
    for name in PHENOMENA:
        col_h = [int(h[i][name]) for i in ids]
        cols = {key: [int(tags[i][name]) for i in ids] for key, tags in columns.items()}
        row: dict[str, Any] = {"n": len(ids), "prevalence_h": _prevalence(col_h)}
        for key, col in cols.items():
            row[f"prevalence_{key}"] = _prevalence(col)
            row[f"kappa_h_{key}"] = _kappa(col_h, col)
            row[f"raw_agreement_h_{key}"] = _raw(col_h, col)
        row["kappa_ab"] = _kappa(cols["a"], cols["b"])
        row["disagreements_h_consensus"] = [
            i for i, x, y in zip(ids, col_h, cols["consensus"]) if x != y]
        agreement[name] = row
    mean_kappa = {pair: _mean([agreement[p][f"kappa_{pair}"] for p in PHENOMENA]) for pair in PAIRS}
    mean_kappa["ab"] = _mean([agreement[p]["kappa_ab"] for p in PHENOMENA])
    return {"n_items": len(ids), "agreement": agreement, "mean_kappa": mean_kappa}


def build_payload(diagnostic_dir: str) -> dict[str, Any]:
    """Read the diagnostic directory and return the full JSON payload."""
    def ann(name: str) -> dict:
        return read_annotations(os.path.join(diagnostic_dir, name))

    with open(os.path.join(diagnostic_dir, "annotators.json"), encoding="utf-8") as handle:
        annotators = json.load(handle)
    result = compute(ann("annotations_a.csv"), ann("annotations_b.csv"), ann(HUMAN_FILE),
                     ann("rule_tags.csv"))
    return {"annotators": annotators, "n_items": result["n_items"],
            "agreement": result["agreement"], "mean_kappa": result["mean_kappa"], "note": NOTE}


def _equal(x: Any, y: Any, path: str = "") -> str | None:
    """None when equal (floats within TOLERANCE), else the path of the first difference."""
    if isinstance(x, bool) or isinstance(y, bool):
        return None if x is y else path or "<root>"
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return None if abs(x - y) <= TOLERANCE else path or "<root>"
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return f"{path} (keys differ)"
        for key in sorted(x):
            diff = _equal(x[key], y[key], f"{path}.{key}" if path else str(key))
            if diff:
                return diff
        return None
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return f"{path} (length differs)"
        for index, (p, q) in enumerate(zip(x, y)):
            diff = _equal(p, q, f"{path}[{index}]")
            if diff:
                return diff
        return None
    return None if x == y and type(x) is type(y) else path or "<root>"


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def _table(payload: dict[str, Any]) -> str:
    head = f"{'phenomenon':<22} {'k h-a':>6} {'k h-b':>6} {'k h-AB':>6} {'k h-rule':>8} " \
           f"{'k a-b':>6} {'prev h':>6} {'h!=AB':>5}"
    lines = [head, "-" * len(head)]
    for name in PHENOMENA:
        r = payload["agreement"][name]
        lines.append(
            f"{name:<22} {_fmt(r['kappa_h_a']):>6} {_fmt(r['kappa_h_b']):>6} "
            f"{_fmt(r['kappa_h_consensus']):>6} {_fmt(r['kappa_h_rule']):>8} "
            f"{_fmt(r['kappa_ab']):>6} {r['prevalence_h']:>6.3f} "
            f"{len(r['disagreements_h_consensus']):>5}")
    m = payload["mean_kappa"]
    lines.append(
        f"{'mean kappa':<22} {_fmt(m['h_a']):>6} {_fmt(m['h_b']):>6} {_fmt(m['h_consensus']):>6} "
        f"{_fmt(m['h_rule']):>8} {_fmt(m['ab']):>6}")
    return "\n".join(lines)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnostic-dir", default=os.path.join(REPO_ROOT, "data", "diagnostic"))
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "step3", "human_annotation_agreement.json"))
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare with the existing --out file; write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        payload = build_payload(args.diagnostic_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1 if args.check else 2
    # round-trip through JSON so the comparison and the file see the same types
    payload = json.loads(json.dumps(payload))
    if args.check:
        try:
            with open(args.out, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"CHECK FAILED: cannot read {args.out}: {exc}", file=sys.stderr)
            return 1
        diff = _equal(stored, payload)
        if diff:
            print(f"CHECK FAILED: {args.out} differs from the recomputed values at {diff}",
                  file=sys.stderr)
            return 1
        print(f"CHECK OK: {args.out} matches the recomputed values "
              f"({payload['n_items']} items).")
        return 0
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(_table(payload))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
