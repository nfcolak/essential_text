"""Loaders for the locked Step 1 manifests and the deduplicated external holdout.

Read-only helpers used by the Step 2 runner. Nothing here downloads or writes data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CONFIGS = ("sentences_allagree", "sentences_75agree", "sentences_50agree")


def file_sha256(path: str) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str) -> dict[str, Any]:
    """Load a Step 1 split manifest (`splits`, `cv_folds`, `records_by_id`)."""
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key in ("splits", "records_by_id"):
        if key not in manifest:
            raise ValueError(f"Manifest {path} is missing required key '{key}'.")
    return manifest


def _arrays(manifest: dict[str, Any], ids: list[str]) -> tuple[list[str], list[str], list[int]]:
    records = manifest["records_by_id"]
    return (
        list(ids),
        [records[rid]["text"] for rid in ids],
        [int(records[rid]["label"]) for rid in ids],
    )


def split_arrays(manifest: dict[str, Any], split: str) -> tuple[list[str], list[str], list[int]]:
    """Return (ids, texts, labels) of a locked split: train, val or test."""
    if split not in manifest["splits"]:
        raise KeyError(f"Unknown split '{split}'; available: {sorted(manifest['splits'])}.")
    return _arrays(manifest, manifest["splits"][split])


def cv_fold_arrays(
    manifest: dict[str, Any], fold: int, part: str
) -> tuple[list[str], list[str], list[int]]:
    """Return (ids, texts, labels) of one cross-validation fold part (train or val)."""
    if part not in ("train", "val"):
        raise ValueError(f"part must be 'train' or 'val', got '{part}'.")
    for info in manifest.get("cv_folds", []):
        if int(info["fold"]) == int(fold):
            return _arrays(manifest, info[part])
    raise KeyError(f"Fold {fold} not found in manifest cv_folds.")


def load_external(path: str) -> tuple[list[str], list[str], list[int]]:
    """Load the deduplicated SEntFiN holdout as (ids, texts, labels)."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = [r for r in payload["records"] if "text" in r and "label" in r]
    return (
        [str(r["id"]) for r in records],
        [r["text"] for r in records],
        [int(r["label"]) for r in records],
    )
