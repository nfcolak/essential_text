"""Export text-free split assignments (id, SHA-256, label, split, CV fold) from the locked manifests.

Without --execute the script only prints what it would write. --check rebuilds in memory and compares with
the files in --out-dir. Never downloads anything.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.assignments import (  # noqa: E402
    ASSIGNMENT_FIELDS,
    SENTFIN_FIELDS,
    assignments_from_manifest,
    read_csv,
    sentfin_rows,
    write_csv,
)

SENTFIN_FILE = "sentfin_dedup.json"

README = """# Split assignments

Text-free split assignments derived from the locked manifests in `data/splits/` (checksums in `SHA256SUMS`).

## Files

- `<config>.csv` (one per Financial PhraseBank agreement config): columns
  `id,sha256,label,cluster_id,split,cv_fold`. `id` is the sentence id, `sha256` the hash of the normalised
  sentence, `label` is 0/1/2 = negative/neutral/positive, `cluster_id` the near-duplicate cluster (the whole
  cluster sits in one split), `split` is train/val/test, `cv_fold` is the 5-fold CV fold whose validation part
  holds the sentence (train ids only; empty otherwise). Rows are sorted by id.
- `sentfin_holdout.csv`: columns `id,sha256,label` of the deduplicated SEntFiN hold-out sentences.

No sentence text is stored here. Texts come from the pinned Financial PhraseBank zip and the pinned SEntFiN csv
(see `src/data/download.py` for URLs and SHA-256 sums); match them by `sha256`.

## Rebuild and check

Rebuild the manifests: `python3 colab/run_step1_colab.py ... --fpb-config all`. Then verify these files:

    python3 scripts/export_split_assignments.py --check

The check first verifies the inputs against `SHA256SUMS` (exit 2 on mismatch), then compares the rebuilt rows with
the files here (exit 0 identical, 1 otherwise).
"""


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(splits_dir: str, paths: list) -> list:
    """Return the input paths whose SHA-256 differs from SHA256SUMS or that are not listed there."""
    sums_path = os.path.join(splits_dir, "SHA256SUMS")
    if not os.path.isfile(sums_path):
        return [sums_path]
    expected = {}
    with open(sums_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                expected[parts[1].lstrip("*")] = parts[0]
    bad = []
    for p in paths:
        if expected.get(os.path.basename(p)) != _file_sha256(p):
            bad.append(p)
    return bad


def config_name(manifest_path: str) -> str:
    name = os.path.basename(manifest_path)
    if name.startswith("sentences_"):
        name = name[len("sentences_"):]
    if name.endswith("_manifest.json"):
        name = name[: -len("_manifest.json")]
    return name


def build_tables(splits_dir: str) -> dict:
    """{filename: (rows, fields)} for every file the export writes (README excluded)."""
    tables = {}
    for path in sorted(glob.glob(os.path.join(splits_dir, "sentences_*_manifest.json"))):
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        tables[config_name(path) + ".csv"] = (assignments_from_manifest(manifest), ASSIGNMENT_FIELDS)
    sentfin_path = os.path.join(splits_dir, SENTFIN_FILE)
    if os.path.isfile(sentfin_path):
        with open(sentfin_path, encoding="utf-8") as f:
            tables["sentfin_holdout.csv"] = (sentfin_rows(json.load(f)), SENTFIN_FIELDS)
    return tables


def _input_paths(splits_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(splits_dir, "sentences_*_manifest.json")))
    sentfin_path = os.path.join(splits_dir, SENTFIN_FILE)
    if os.path.isfile(sentfin_path):
        paths.append(sentfin_path)
    return paths


def _stringify(rows: list, fields: list) -> list:
    return [{k: str(r[k]) for k in fields} for r in rows]


def check(tables: dict, out_dir: str) -> list:
    """Differences between the rebuilt tables and the files in out_dir (empty list when identical)."""
    diffs = []
    for fname, (rows, fields) in sorted(tables.items()):
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            diffs.append("{}: missing".format(fname))
            continue
        want = _stringify(rows, fields)
        have = read_csv(path)
        if len(want) != len(have):
            diffs.append("{}: {} rows expected, {} found".format(fname, len(want), len(have)))
            continue
        n_diff = sum(1 for a, b in zip(want, have) if a != b)
        if n_diff:
            first = next(i for i, (a, b) in enumerate(zip(want, have)) if a != b)
            diffs.append("{}: {} differing rows (first at row {}, id {})".format(
                fname, n_diff, first + 2, want[first]["id"]))
    return diffs


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="data/splits/assignments")
    ap.add_argument("--execute", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--check", action="store_true", help="compare the rebuilt rows with the files in --out-dir")
    args = ap.parse_args(argv)

    bad = verify_inputs(args.splits_dir, _input_paths(args.splits_dir))
    if bad:
        print("ERROR: input hash mismatch against SHA256SUMS: " + ", ".join(bad), file=sys.stderr)
        return 2
    tables = build_tables(args.splits_dir)
    if not tables:
        print("ERROR: no manifests found in {}".format(args.splits_dir), file=sys.stderr)
        return 2

    if args.check:
        diffs = check(tables, args.out_dir)
        if diffs:
            print("DIFFERENT:")
            for d in diffs:
                print("  " + d)
            return 1
        print("OK: {} files identical to the locked manifests".format(len(tables)))
        return 0

    for fname, (rows, _) in sorted(tables.items()):
        print("{} {} ({} rows)".format("write" if args.execute else "would write",
                                       os.path.join(args.out_dir, fname), len(rows)))
    print("{} {}".format("write" if args.execute else "would write", os.path.join(args.out_dir, "README.md")))
    if not args.execute:
        print("dry run: pass --execute to write")
        return 0
    for fname, (rows, fields) in tables.items():
        write_csv(os.path.join(args.out_dir, fname), rows, fields)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    return 0


if __name__ == "__main__":
    sys.exit(main())
