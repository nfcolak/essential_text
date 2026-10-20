"""Build the label-stratified diagnostic subset (test split only) for the Step 3 phenomenon annotation.

Without --execute the script only prints what it would write. Never downloads anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.phenomena import PHENOMENA, tag_many  # noqa: E402

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_split_hashes(splits_dir: str, paths: list[str]) -> str | None:
    """Compare the loaded split files with <splits_dir>/SHA256SUMS. Returns the first mismatching path, else None.

    A missing SHA256SUMS file only prints a warning; files not listed in it are not checked.
    """
    sums_path = os.path.join(splits_dir, "SHA256SUMS")
    if not os.path.isfile(sums_path):
        print("WARNING: {} not found, split hashes not verified".format(sums_path), file=sys.stderr)
        return None
    expected: dict[str, str] = {}
    with open(sums_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[os.path.basename(parts[1].lstrip("*"))] = parts[0].lower()
    for path in paths:
        want = expected.get(os.path.basename(path))
        if want is not None and _file_sha256(path) != want:
            return path
    return None


def _rule_tags_only(out_dir: str, execute: bool) -> int:
    items_path = os.path.join(out_dir, "diagnostic_items.csv")
    if not os.path.isfile(items_path):
        print("Missing items file: {}".format(items_path), file=sys.stderr)
        return 2
    with open(items_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["diag_id"] for r in rows]
    tags = tag_many([r["text"] for r in rows])
    print("rule-tags-only: {} items from {}".format(len(rows), items_path))
    print("rule-tag prevalence: {}".format({ph: sum(t[ph] for t in tags) for ph in PHENOMENA}))
    if not execute:
        print("Plan only (no --execute). Would rewrite {} only.".format(os.path.join(out_dir, "rule_tags.csv")))
        return 0
    with open(os.path.join(out_dir, "rule_tags.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diag_id"] + list(PHENOMENA))
        for d, t in zip(ids, tags):
            w.writerow([d] + [t[ph] for ph in PHENOMENA])
    print("Wrote rule_tags.csv for {} items".format(len(rows)))
    return 0


def largest_remainder_counts(class_sizes: dict[int, int], n: int) -> dict[int, int]:
    """Per-class counts ~ n * class share, adjusted by largest remainder so they sum to n."""
    total = sum(class_sizes.values())
    exact = {c: n * s / total for c, s in class_sizes.items()}
    counts = {c: int(np.floor(v)) for c, v in exact.items()}
    missing = n - sum(counts.values())
    order = sorted(class_sizes, key=lambda c: (-(exact[c] - counts[c]), c))
    for c in order[:missing]:
        counts[c] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="sentences_50agree")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--out-dir", default="data/diagnostic")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--rule-tags-only", action="store_true",
                   help="with --execute: recompute rule_tags.csv from the existing diagnostic_items.csv in --out-dir; "
                        "diagnostic_key.json and diagnostic_items.csv stay untouched")
    args = p.parse_args(argv)

    if args.rule_tags_only:
        return _rule_tags_only(args.out_dir, args.execute)

    manifest_path = os.path.join(args.splits_dir, "{}_manifest.json".format(args.config))
    if not os.path.isfile(manifest_path):
        print("Missing manifest: {}".format(manifest_path), file=sys.stderr)
        return 2
    bad = check_split_hashes(args.splits_dir, [manifest_path])
    if bad is not None:
        print("SHA-256 mismatch against SHA256SUMS: {}".format(bad), file=sys.stderr)
        return 2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records_by_id"]
    by_class: dict[int, list[str]] = {}
    for rid in sorted(manifest["splits"]["test"]):
        by_class.setdefault(int(records[rid]["label"]), []).append(rid)
    total = sum(len(v) for v in by_class.values())
    if args.n > total:
        print("Requested n={} but the test split has only {} rows".format(args.n, total), file=sys.stderr)
        return 2
    counts = largest_remainder_counts({c: len(v) for c, v in by_class.items()}, args.n)

    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for c in sorted(by_class):
        idx = rng.choice(len(by_class[c]), size=counts[c], replace=False)
        chosen.extend(by_class[c][i] for i in sorted(idx.tolist()))
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order.tolist()]

    diag_ids = ["d{:03d}".format(i + 1) for i in range(len(chosen))]
    texts = [records[rid]["text"] for rid in chosen]
    tags = tag_many(texts)

    print("config={} seed={} n={} source_split=test (test size {})".format(args.config, args.seed, args.n, total))
    print("per-class counts: {}".format({LABEL_NAMES.get(c, str(c)): counts[c] for c in sorted(counts)}))
    print("rule-tag prevalence: {}".format({ph: sum(t[ph] for t in tags) for ph in PHENOMENA}))
    out_files = ["diagnostic_key.json", "diagnostic_items.csv", "rule_tags.csv"]
    if not args.execute:
        print("Plan only (no --execute). Would write to {}: {}".format(args.out_dir, ", ".join(out_files)))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    key = {
        "config": args.config,
        "seed": args.seed,
        "n": len(chosen),
        "source_split": "test",
        "manifest_sha256": _file_sha256(manifest_path),
        "items": [
            {
                "diag_id": d,
                "record_id": rid,
                "label": int(records[rid]["label"]),
                "label_name": LABEL_NAMES.get(int(records[rid]["label"]), str(records[rid]["label"])),
                "sha256": records[rid]["sha256"],
            }
            for d, rid in zip(diag_ids, chosen)
        ],
    }
    with open(os.path.join(args.out_dir, "diagnostic_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "diagnostic_items.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "text"])
        for d, t in zip(diag_ids, texts):
            w.writerow([d, t])
    with open(os.path.join(args.out_dir, "rule_tags.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diag_id"] + list(PHENOMENA))
        for d, t in zip(diag_ids, tags):
            w.writerow([d] + [t[ph] for ph in PHENOMENA])
    print("Wrote {} items to {}".format(len(chosen), args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
