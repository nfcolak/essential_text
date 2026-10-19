"""Split creation, cluster locking, and leakage verification for financial sentiment datasets.

Implements canonical 3-class label mapping, structured record building, cluster-aware
stratified 70/15/15 train/val/test splits and 5-fold cross-validation locking, and
rigorous leakage verification (zero ID, zero SHA-256, zero cluster overlap, and
cross-split token Jaccard < 0.80).
"""

from __future__ import annotations

import random
from typing import Any

from src.data.dedup import (
    cluster_near_duplicates,
    normalize_text,
    sentence_sha256,
    tokenize_for_jaccard,
)

LABEL_TO_ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def build_records(
    texts: list[str],
    labels: list[int],
    prefix: str = "fpb",
    jaccard_threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Build normalized sentence records with SHA-256 hashes and near-duplicate cluster IDs.

    Args:
        texts: Raw sentence strings.
        labels: Integer sentiment labels (0, 1, or 2).
        prefix: Dataset identifier prefix for record IDs.
        jaccard_threshold: Threshold for near-duplicate clustering (default: 0.80).

    Returns:
        List of record dictionaries conforming to the schema in design.md.
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"Mismatched lengths: {len(texts)} texts and {len(labels)} labels."
        )

    cluster_ids = cluster_near_duplicates(texts, threshold=jaccard_threshold)
    id_padding = 5 if len(texts) >= 10000 else 4

    records: list[dict[str, Any]] = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        lbl = int(label)
        if lbl not in ID_TO_LABEL:
            raise ValueError(f"Invalid label {lbl}; expected 0, 1, or 2.")

        norm = normalize_text(text)
        rec_id = f"{prefix}_{i:0{id_padding}d}"
        rec = {
            "id": rec_id,
            "text": str(text),
            "norm_text": norm,
            "sha256": sentence_sha256(text),
            "label": lbl,
            "label_name": ID_TO_LABEL[lbl],
            "cluster_id": cluster_ids[i],
        }
        records.append(rec)

    return records


def create_stratified_splits(
    records: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    n_folds: int = 5,
) -> dict[str, Any]:
    """Create cluster-aware stratified train/val/test splits and cross-validation folds.

    Records belonging to the same cluster_id are guaranteed to remain in the same partition,
    completely preventing near-duplicate leakage across train, val, and test.

    Returns:
        JSON-serializable split manifest dictionary.
    """
    if not records:
        raise ValueError("Cannot create splits from empty records list.")

    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise ValueError(f"Invalid split ratios {ratios}; expected 3 non-negative floats.")

    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        ratios = (ratios[0] / total_ratio, ratios[1] / total_ratio, ratios[2] / total_ratio)

    records_by_id = {r["id"]: r for r in records}
    prefix = records[0]["id"].split("_")[0] if "_" in records[0]["id"] else "data"

    # Compute total class counts
    class_counts = {ID_TO_LABEL[i]: 0 for i in (0, 1, 2)}
    for r in records:
        class_counts[ID_TO_LABEL[r["label"]]] += 1

    # Group records by cluster_id
    clusters: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        cid = r["cluster_id"]
        clusters.setdefault(cid, []).append(r)

    # Determine representative label for each cluster (majority vote)
    def cluster_label(cluster_records: list[dict[str, Any]]) -> int:
        counts: dict[int, int] = {}
        for cr in cluster_records:
            lbl = cr["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    # Group clusters by label
    clusters_by_label: dict[int, list[list[dict[str, Any]]]] = {0: [], 1: [], 2: []}
    for cid, cl_records in clusters.items():
        lbl = cluster_label(cl_records)
        clusters_by_label[lbl].append(cl_records)

    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    # Allocate clusters per class label
    for lbl in (0, 1, 2):
        cl_list = clusters_by_label[lbl]
        # Deterministic shuffle
        cl_list = list(cl_list)
        # Use a reproducible sort key first to ensure cross-platform consistency
        cl_list.sort(key=lambda c: (len(c), c[0]["id"]))
        rng.shuffle(cl_list)

        total_label_records = sum(len(c) for c in cl_list)
        target_train = round(total_label_records * ratios[0])
        target_val = round(total_label_records * ratios[1])
        target_test = total_label_records - target_train - target_val

        # Ensure non-empty allocations if there are enough records
        if total_label_records >= 3:
            if target_val < 1:
                target_val = 1
                if target_train > 1:
                    target_train -= 1
            if target_test < 1:
                target_test = 1
                if target_train > 1:
                    target_train -= 1

        targets = {"train": target_train, "val": target_val, "test": target_test}
        curr_counts = {"train": 0, "val": 0, "test": 0}

        for cluster in cl_list:
            c_len = len(cluster)
            c_ids = [r["id"] for r in cluster]

            # Choose partition with largest deficit (target - current)
            best_part = "train"
            best_deficit = -10**9
            for part in ("train", "val", "test"):
                deficit = targets[part] - curr_counts[part]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_part = part

            split_ids[best_part].extend(c_ids)
            curr_counts[best_part] += c_len

    # Sort split IDs for clean determinism
    for part in ("train", "val", "test"):
        split_ids[part].sort()

    hashes = {
        part: [records_by_id[rid]["sha256"] for rid in split_ids[part]]
        for part in ("train", "val", "test")
    }

    # Generate n_folds CV folds on the train partition
    cv_folds: list[dict[str, Any]] = []
    if n_folds >= 2 and len(split_ids["train"]) >= n_folds:
        # Group train records by cluster
        train_clusters: dict[int, list[dict[str, Any]]] = {}
        for rid in split_ids["train"]:
            r = records_by_id[rid]
            train_clusters.setdefault(r["cluster_id"], []).append(r)

        train_clusters_by_label: dict[int, list[list[dict[str, Any]]]] = {0: [], 1: [], 2: []}
        for cid, cl_records in train_clusters.items():
            lbl = cluster_label(cl_records)
            train_clusters_by_label[lbl].append(cl_records)

        fold_val_ids: list[list[str]] = [[] for _ in range(n_folds)]
        fold_counts = [0] * n_folds

        for lbl in (0, 1, 2):
            cl_list = list(train_clusters_by_label[lbl])
            cl_list.sort(key=lambda c: (len(c), c[0]["id"]))
            rng_cv = random.Random(seed + 100 + lbl)
            rng_cv.shuffle(cl_list)

            for cluster in cl_list:
                c_len = len(cluster)
                c_ids = [r["id"] for r in cluster]
                # Pick fold with smallest record count
                min_fold = min(range(n_folds), key=lambda f: fold_counts[f])
                fold_val_ids[min_fold].extend(c_ids)
                fold_counts[min_fold] += c_len

        # Construct each fold's train and val sets
        train_ids_set = set(split_ids["train"])
        for k in range(n_folds):
            val_k = sorted(fold_val_ids[k])
            val_k_set = set(val_k)
            train_k = sorted(list(train_ids_set - val_k_set))
            cv_folds.append({
                "fold": k,
                "train": train_k,
                "val": val_k,
            })

    manifest = {
        "dataset_prefix": prefix,
        "seed": seed,
        "total_records": len(records),
        "class_counts": class_counts,
        "splits": split_ids,
        "hashes": hashes,
        "cv_folds": cv_folds,
        "records_by_id": records_by_id,
    }
    return manifest


def verify_no_leakage(manifest: dict[str, Any], jaccard_threshold: float = 0.80) -> dict[str, Any]:
    """Verify that there is zero data leakage across train, val, and test splits.

    Checks:
    1. Zero ID overlap across splits.
    2. Zero SHA-256 hash overlap across splits.
    3. Zero cluster_id overlap across splits.
    4. Maximum pairwise token Jaccard similarity between any two sentences across
       different splits is strictly less than jaccard_threshold.
    5. Zero ID / hash / cluster overlap in CV folds.

    Raises:
        ValueError: If any leakage or constraint violation is found.

    Returns:
        Summary dictionary with verification metrics.
    """
    splits = manifest.get("splits", {})
    train_ids = set(splits.get("train", []))
    val_ids = set(splits.get("val", []))
    test_ids = set(splits.get("test", []))
    records_by_id = manifest.get("records_by_id", {})

    # 1. ID Overlap
    if train_ids & val_ids:
        raise ValueError(f"Leakage detected: {len(train_ids & val_ids)} IDs overlap between train and val.")
    if train_ids & test_ids:
        raise ValueError(f"Leakage detected: {len(train_ids & test_ids)} IDs overlap between train and test.")
    if val_ids & test_ids:
        raise ValueError(f"Leakage detected: {len(val_ids & test_ids)} IDs overlap between val and test.")

    # 2. SHA-256 Hash Overlap
    hashes = manifest.get("hashes", {})
    train_hashes = set(hashes.get("train", []))
    val_hashes = set(hashes.get("val", []))
    test_hashes = set(hashes.get("test", []))

    if train_hashes & val_hashes:
        raise ValueError(f"Leakage detected: SHA-256 hash overlap between train and val.")
    if train_hashes & test_hashes:
        raise ValueError(f"Leakage detected: SHA-256 hash overlap between train and test.")
    if val_hashes & test_hashes:
        raise ValueError(f"Leakage detected: SHA-256 hash overlap between val and test.")

    # 3. Cluster ID Overlap
    train_clusters = {records_by_id[rid]["cluster_id"] for rid in train_ids if rid in records_by_id}
    val_clusters = {records_by_id[rid]["cluster_id"] for rid in val_ids if rid in records_by_id}
    test_clusters = {records_by_id[rid]["cluster_id"] for rid in test_ids if rid in records_by_id}

    if train_clusters & val_clusters:
        raise ValueError("Leakage detected: cluster_id overlap between train and val.")
    if train_clusters & test_clusters:
        raise ValueError("Leakage detected: cluster_id overlap between train and test.")
    if val_clusters & test_clusters:
        raise ValueError("Leakage detected: cluster_id overlap between val and test.")

    # 4. Cross-split Token Jaccard check
    split_tokens = {
        "train": [(rid, tokenize_for_jaccard(records_by_id[rid]["text"])) for rid in train_ids if rid in records_by_id],
        "val": [(rid, tokenize_for_jaccard(records_by_id[rid]["text"])) for rid in val_ids if rid in records_by_id],
        "test": [(rid, tokenize_for_jaccard(records_by_id[rid]["text"])) for rid in test_ids if rid in records_by_id],
    }

    max_jaccard = 0.0
    pairs_to_check = [("train", "val"), ("train", "test"), ("val", "test")]

    for p1, p2 in pairs_to_check:
        list1 = split_tokens[p1]
        list2 = split_tokens[p2]
        for id1, tok1 in list1:
            len1 = len(tok1)
            for id2, tok2 in list2:
                len2 = len(tok2)
                if len1 == 0 or len2 == 0:
                    continue
                if min(len1, len2) / max(len1, len2) < jaccard_threshold:
                    # Could still be > max_jaccard, but cannot exceed threshold
                    continue

                inter = len(tok1 & tok2)
                union = len(tok1 | tok2)
                jacc = inter / union if union > 0 else 0.0
                if jacc > max_jaccard:
                    max_jaccard = jacc
                if jacc >= jaccard_threshold:
                    raise ValueError(
                        f"Leakage detected: Cross-split token Jaccard {jacc:.4f} >= {jaccard_threshold} "
                        f"between {p1} ({id1}) and {p2} ({id2})."
                    )

    # 5. Cross-validation fold verification
    for fold_info in manifest.get("cv_folds", []):
        f_idx = fold_info.get("fold", -1)
        f_train = set(fold_info.get("train", []))
        f_val = set(fold_info.get("val", []))
        if f_train & f_val:
            raise ValueError(f"Leakage detected in CV fold {f_idx}: train and val IDs overlap.")
        f_train_clusters = {records_by_id[rid]["cluster_id"] for rid in f_train if rid in records_by_id}
        f_val_clusters = {records_by_id[rid]["cluster_id"] for rid in f_val if rid in records_by_id}
        if f_train_clusters & f_val_clusters:
            raise ValueError(f"Leakage detected in CV fold {f_idx}: cluster_id overlap between train and val.")

    return {
        "passed": True,
        "max_cross_split_jaccard": round(float(max_jaccard), 6),
        "id_overlap_count": 0,
        "hash_overlap_count": 0,
        "cluster_overlap_count": 0,
        "split_sizes": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }
