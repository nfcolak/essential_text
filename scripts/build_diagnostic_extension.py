"""Build blind annotation items for the remainder of the locked 50agree test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phenomena import PHENOMENA, tag_many
from src.data.io import file_sha256, load_manifest, split_arrays

CONFIG = "sentences_50agree"
SEED = 2026
N_ORIGINAL = 300
N_EXTENSION = 426
LABEL_NAMES = ("negative", "neutral", "positive")
SPLITS_DIR = REPO_ROOT.parent / "essential_text" / "data" / "splits"
DIAGNOSTIC_DIR = REPO_ROOT / "data" / "diagnostic"


def load_test(split_dir: Path) -> tuple[dict[str, tuple[str, int]], str]:
    path = split_dir / f"{CONFIG}_manifest.json"
    ids, texts, labels = split_arrays(load_manifest(str(path)), "test")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate record IDs in the test split")
    if any(label not in (0, 1, 2) for label in labels):
        raise ValueError(f"{path}: invalid gold label in the test split")
    return dict(zip(ids, zip(texts, labels))), file_sha256(str(path))


def _csv_bytes(header: tuple[str, ...], rows) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_files(records: dict[str, tuple[str, int]], manifest_sha256: str,
                original_key: dict, *, expected_original: int = N_ORIGINAL,
                expected_extension: int = N_EXTENSION) -> dict[str, bytes]:
    if (original_key["config"], original_key["source_split"], original_key["manifest_sha256"]) != (
        CONFIG, "test", manifest_sha256
    ):
        raise ValueError("The original diagnostic key does not match the locked test manifest")
    original_ids = {item["record_id"] for item in original_key["items"]}
    if len(original_key["items"]) != expected_original or len(original_ids) != expected_original:
        raise ValueError(f"Expected {expected_original} unique records in the original diagnostic key")
    if original_ids - records.keys():
        raise ValueError("The original diagnostic key contains records outside the test split")
    remaining = sorted(records.keys() - original_ids)
    if len(remaining) != expected_extension:
        raise ValueError(f"Expected {expected_extension} remaining test records, found {len(remaining)}")

    random.Random(SEED).shuffle(remaining)
    ext_ids = [f"e{i:03d}" for i in range(1, len(remaining) + 1)]
    texts = [records[record_id][0] for record_id in remaining]
    items = []
    for ext_id, record_id, text in zip(ext_ids, remaining, texts):
        label = records[record_id][1]
        items.append({
            "ext_id": ext_id,
            "record_id": record_id,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    key = {
        "config": CONFIG,
        "seed": SEED,
        "n": len(items),
        "source_split": "test",
        "manifest_sha256": manifest_sha256,
        "items": items,
    }
    return {
        "items.csv": _csv_bytes(("ext_id", "text"), zip(ext_ids, texts)),
        "key.json": (json.dumps(key, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        "rule_tags.csv": _csv_bytes(
            ("diag_id",) + PHENOMENA,
            ([ext_id] + [tags[name] for name in PHENOMENA]
             for ext_id, tags in zip(ext_ids, tag_many(texts))),
        ),
    }


def check_files(out_dir: Path, files: dict[str, bytes]) -> int:
    mismatches = []
    for name, expected in files.items():
        path = out_dir / name
        try:
            matches = path.read_bytes() == expected
        except OSError:
            matches = False
        if not matches:
            mismatches.append(str(path))
    if mismatches:
        print(f"CHECK FAILED: missing, unreadable or different files: {', '.join(mismatches)}",
              file=sys.stderr)
        return 1
    print(f"CHECK OK: {len(files)} files match in {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--diagnostic-dir", type=Path, default=DIAGNOSTIC_DIR)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="write the extension files")
    modes.add_argument("--check", action="store_true", help="compare existing files without writing")
    args = parser.parse_args(argv)
    out_dir = args.diagnostic_dir / "extension"
    try:
        records, manifest_hash = load_test(args.splits_dir)
        original_key = json.loads((args.diagnostic_dir / "diagnostic_key.json").read_bytes())
        files = build_files(records, manifest_hash, original_key)
        if args.check:
            return check_files(out_dir, files)
        if args.execute:
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (out_dir / name).write_bytes(content)
            print(f"Wrote {N_EXTENSION} extension items to {out_dir}")
        else:
            print(f"Plan only: would write {N_EXTENSION} items to {out_dir}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
