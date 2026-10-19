"""Near-duplicate clustering and text deduplication for financial sentiment data.

Implements text normalization, SHA-256 sentence hashing, token-level Jaccard similarity,
connected-component near-duplicate clustering (threshold >= 0.80), and external holdout
cross-dataset deduplication to prevent data leakage across splits and datasets.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, replacing non-alphanumeric chars with whitespace,

    and collapsing multiple whitespace characters.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    lowered = text.lower()
    alphanumeric_only = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    normalized = re.sub(r"\s+", " ", alphanumeric_only).strip()
    return normalized


def sentence_sha256(text: str) -> str:
    """Compute the SHA-256 hex digest of normalized sentence text."""
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def tokenize_for_jaccard(text: str) -> set[str]:
    """Tokenize normalized text into a set of unique whitespace-delimited tokens."""
    norm = normalize_text(text)
    if not norm:
        return set()
    return set(norm.split())


def token_jaccard(text_a: str | set[str], text_b: str | set[str]) -> float:
    """Compute token-level Jaccard similarity between two texts or token sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|.
    Returns 1.0 if both sets are empty, 0.0 if one is empty and the other is not.
    """
    tokens_a = text_a if isinstance(text_a, set) else tokenize_for_jaccard(text_a)
    tokens_b = text_b if isinstance(text_b, set) else tokenize_for_jaccard(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def cluster_near_duplicates(texts: list[str], threshold: float = 0.80) -> list[int]:
    """Cluster sentences using token Jaccard similarity >= threshold into connected components.

    Sentences with pairwise Jaccard >= threshold are connected by an edge. Connected
    components form clusters such that near-duplicates share the same cluster ID.

    Returns a list of integer cluster IDs of length len(texts).
    """
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [0]

    token_sets = [tokenize_for_jaccard(t) for t in texts]

    # Disjoint Set Union (Union-Find) with path compression and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

    for i in range(n):
        set_i = token_sets[i]
        len_i = len(set_i)
        for j in range(i + 1, n):
            set_j = token_sets[j]
            len_j = len(set_j)

            # Mathematical pruning: |A ∩ B| / |A ∪ B| <= min(|A|, |B|) / max(|A|, |B|)
            if len_i == 0 and len_j == 0:
                union(i, j)
                continue
            if len_i == 0 or len_j == 0:
                continue

            min_len = min(len_i, len_j)
            max_len = max(len_i, len_j)
            if (min_len / max_len) < threshold:
                continue

            # Full Jaccard calculation
            intersection_size = len(set_i & set_j)
            union_size = len(set_i | set_j)
            if union_size > 0 and (intersection_size / union_size) >= threshold:
                union(i, j)

    # Assign deterministic dense 0-indexed cluster IDs based on first appearance
    root_to_id: dict[int, int] = {}
    cluster_ids: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids.append(root_to_id[root])

    return cluster_ids


def deduplicate_external_holdout(
    external_records: list[dict[str, Any]],
    reference_texts: list[str],
    threshold: float = 0.80,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter external holdout records to ensure zero overlap and no near-duplicates (Jaccard >= threshold)

    against a reference dataset (e.g. Financial PhraseBank).

    A record is removed if:
    1. Its normalized SHA-256 matches any reference sentence SHA-256.
    2. Its token Jaccard similarity with ANY reference sentence is >= threshold.

    Returns:
        (filtered_records, stats_dict) where stats_dict contains:
        {"total_input": int, "removed_overlap": int, "kept": int}
    """
    total_input = len(external_records)
    if total_input == 0 or not reference_texts:
        return list(external_records), {
            "total_input": total_input,
            "removed_overlap": 0,
            "kept": total_input,
        }

    # Pre-index reference hashes and token sets
    ref_hashes = {sentence_sha256(t) for t in reference_texts}
    ref_tokens = [tokenize_for_jaccard(t) for t in reference_texts]
    ref_tokens_nonempty = [s for s in ref_tokens if s]
    has_empty_ref = any(len(s) == 0 for s in ref_tokens)

    kept_records: list[dict[str, Any]] = []

    for record in external_records:
        rec_text = (
            record.get("text")
            or record.get("sentence")
            or record.get("norm_text")
            or ""
        )
        rec_hash = record.get("sha256") or sentence_sha256(rec_text)

        # 1. Exact SHA-256 check
        if rec_hash in ref_hashes:
            continue

        rec_tokens = tokenize_for_jaccard(rec_text)
        if not rec_tokens:
            if has_empty_ref:
                continue
            kept_records.append(record)
            continue

        # 2. Token Jaccard check with length pruning
        len_rec = len(rec_tokens)
        is_duplicate = False

        for r_tokens in ref_tokens_nonempty:
            len_ref = len(r_tokens)
            if min(len_rec, len_ref) / max(len_rec, len_ref) < threshold:
                continue

            inter = len(rec_tokens & r_tokens)
            union = len(rec_tokens | r_tokens)
            if union > 0 and (inter / union) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_records.append(record)

    removed = total_input - len(kept_records)
    stats = {
        "total_input": total_input,
        "removed_overlap": removed,
        "kept": len(kept_records),
    }
    return kept_records, stats
