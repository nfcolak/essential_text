"""Data ingestion, deduplication, split locking, and leakage verification package."""

from __future__ import annotations

from src.data.dedup import (
    cluster_near_duplicates,
    deduplicate_external_holdout,
    normalize_text,
    sentence_sha256,
    token_jaccard,
    tokenize_for_jaccard,
)
from src.data.download import (
    generate_synthetic_dataset,
    load_real_datasets,
)
from src.data.splits import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_records,
    create_stratified_splits,
    verify_no_leakage,
)

__all__ = [
    "normalize_text",
    "sentence_sha256",
    "tokenize_for_jaccard",
    "token_jaccard",
    "cluster_near_duplicates",
    "deduplicate_external_holdout",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "build_records",
    "create_stratified_splits",
    "verify_no_leakage",
    "generate_synthetic_dataset",
    "load_real_datasets",
]
