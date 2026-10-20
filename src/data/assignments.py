"""Text-free split assignments derived from the locked split manifests.

Pure functions only: no file is read or written except through ``write_csv`` / ``read_csv``.
"""

from __future__ import annotations

import csv
import os

ASSIGNMENT_FIELDS = ["id", "sha256", "label", "cluster_id", "split", "cv_fold"]
SENTFIN_FIELDS = ["id", "sha256", "label"]


def assignments_from_manifest(manifest: dict) -> list[dict]:
    """One row per sentence id of the manifest, sorted by id.

    ``cv_fold`` is the fold whose ``val`` holds the id (train ids only), empty otherwise.
    """
    records = manifest["records_by_id"]
    split_of = {}
    for split, ids in manifest["splits"].items():
        for sid in ids:
            split_of[sid] = split
    fold_of = {}
    for fold in manifest.get("cv_folds", []):
        for sid in fold["val"]:
            fold_of[sid] = fold["fold"]
    rows = []
    for sid in sorted(split_of):
        rec = records[sid]
        split = split_of[sid]
        cluster = rec.get("cluster_id")
        fold = fold_of.get(sid) if split == "train" else None
        rows.append(
            {
                "id": sid,
                "sha256": rec["sha256"],
                "label": int(rec["label"]),
                "cluster_id": "" if cluster is None else int(cluster),
                "split": split,
                "cv_fold": "" if fold is None else int(fold),
            }
        )
    return rows


def sentfin_rows(dedup: dict) -> list[dict]:
    """Held-out SEntFiN rows (id, sha256, label), sorted by id; no sentence text."""
    return [
        {"id": r["id"], "sha256": r["sha256"], "label": int(r["label"])}
        for r in sorted(dedup["records"], key=lambda r: r["id"])
    ]


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def read_csv(path: str) -> list[dict]:
    """Rows as dicts of strings, exactly as written."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
