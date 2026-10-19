"""Unit tests for leakage-free data splits, near-duplicate clustering, and deduplication.

Verifies:
- Text normalization and deterministic SHA-256 sentence hashing
- Token-level Jaccard similarity and near-duplicate clustering (threshold >= 0.80)
- Cluster-aware stratified 70/15/15 train/val/test splits and 5-fold CV
- Strict cluster locking (near-duplicates never cross partition boundaries)
- verify_no_leakage passing on clean splits and detecting intentional leaks
- Cross-dataset deduplication against external holdout (SEntFiN 1.0)
- Execution gating on load_real_datasets
"""

from __future__ import annotations

import copy
import unittest

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


class TestDeduplicationAndHashing(unittest.TestCase):
    def test_normalize_text(self):
        raw = "  Operating Profit Rose by 15.4% -- Reaching €45M in Q3!  "
        expected = "operating profit rose by 15 4 reaching 45m in q3"
        self.assertEqual(normalize_text(raw), expected)

        # None / empty handling
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

    def test_sentence_sha256_deterministic_and_normalized(self):
        s1 = "Operating profit rose by 15% in the third quarter."
        s2 = "  operating profit rose by 15% in the third quarter!  "
        s3 = "Operating profit fell by 15% in the third quarter."

        h1 = sentence_sha256(s1)
        h2 = sentence_sha256(s2)
        h3 = sentence_sha256(s3)

        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_token_jaccard(self):
        # Identical
        s1 = "Operating profit grew substantially"
        s2 = "operating profit grew substantially"
        self.assertAlmostEqual(token_jaccard(s1, s2), 1.0)

        # Disjoint
        s3 = "Different unrelated financial terms"
        self.assertAlmostEqual(token_jaccard(s1, s3), 0.0)

        # Empty sets
        self.assertEqual(token_jaccard("", ""), 1.0)
        self.assertEqual(token_jaccard(s1, ""), 0.0)

        # Near duplicates: differ by only 1 token out of 10
        t1 = "operating profit rose by 15 to 45 million in previous year"
        t2 = "operating profit rose by 15 to 45 million in prior year"
        jacc = token_jaccard(t1, t2)
        # 9 shared tokens, 11 union tokens -> 9/11 ≈ 0.818 >= 0.80
        self.assertGreaterEqual(jacc, 0.80)

    def test_cluster_near_duplicates(self):
        texts = [
            "Operating profit rose by 15 to 45 million EUR compared with previous year.",
            "Operating profit rose by 15 to 45 million EUR compared with prior year.",  # near-dup to 0
            "Company declared a total dividend of 2 euros per share.",
            "Company declared a regular dividend of 2 euros per share.",  # near-dup to 2 (9/11 = 0.818 >= 0.80)
            "Unrelated sentence regarding debt liabilities and amortization schedules.",
        ]
        clusters = cluster_near_duplicates(texts, threshold=0.80)
        self.assertEqual(len(clusters), 5)
        # 0 and 1 share cluster
        self.assertEqual(clusters[0], clusters[1])
        # 2 and 3 share cluster
        self.assertEqual(clusters[2], clusters[3])
        # 0, 2, 4 are in distinct clusters
        self.assertNotEqual(clusters[0], clusters[2])
        self.assertNotEqual(clusters[0], clusters[4])
        self.assertNotEqual(clusters[2], clusters[4])

    def test_deduplicate_external_holdout(self):
        ref_texts = [
            "Nokia signed a 5G multi-year patent license agreement with Apple.",
            "Operating margin rose from 8% to 12% in the second quarter.",
        ]
        external_records = [
            {"id": "ext_1", "text": "Nokia signed a 5G multi-year patent license agreement with Apple.", "label": 2},
            {"id": "ext_2", "text": "Operating margin rose from 8% to 12% in the second quarter!", "label": 2},
            {"id": "ext_3", "text": "Nokia signed a 5G multi-year patent licensing agreement with Apple.", "label": 2},  # near-dup >= 0.80
            {"id": "ext_4", "text": "Sampo completed acquisition of Hastings insurance group.", "label": 2},  # novel
            {"id": "ext_5", "text": "Fortum announced temporary shutdown of nuclear reactor unit.", "label": 0},  # novel
        ]

        filtered, stats = deduplicate_external_holdout(external_records, ref_texts, threshold=0.80)
        self.assertEqual(stats["total_input"], 5)
        self.assertEqual(stats["removed_overlap"], 3)
        self.assertEqual(stats["kept"], 2)

        kept_ids = [r["id"] for r in filtered]
        self.assertEqual(kept_ids, ["ext_4", "ext_5"])


class TestSplitsAndLeakageVerification(unittest.TestCase):
    def setUp(self):
        data = generate_synthetic_dataset(n_per_class=30, seed=42)
        self.pb_data = data["phrasebank_synthetic"]
        self.sentfin_data = data["sentfin_synthetic"]

    def test_build_records(self):
        texts = self.pb_data["texts"][:10]
        labels = self.pb_data["labels"][:10]
        records = build_records(texts, labels, prefix="test_fpb", jaccard_threshold=0.80)

        self.assertEqual(len(records), 10)
        for r in records:
            self.assertTrue(r["id"].startswith("test_fpb_"))
            self.assertIn("text", r)
            self.assertIn("norm_text", r)
            self.assertIn("sha256", r)
            self.assertIn("label", r)
            self.assertIn("label_name", r)
            self.assertIn("cluster_id", r)
            self.assertEqual(r["label_name"], ID_TO_LABEL[r["label"]])

        # Mismatched lengths
        with self.assertRaises(ValueError):
            build_records(texts[:5], labels[:10])

        # Invalid label
        with self.assertRaises(ValueError):
            build_records(["Sample text"], [99])

    def test_create_stratified_splits_cluster_locking(self):
        records = build_records(
            self.pb_data["texts"],
            self.pb_data["labels"],
            prefix="fpb",
            jaccard_threshold=0.80,
        )
        manifest = create_stratified_splits(records, seed=42, ratios=(0.70, 0.15, 0.15), n_folds=5)

        self.assertIn("splits", manifest)
        self.assertIn("cv_folds", manifest)
        self.assertEqual(len(manifest["cv_folds"]), 5)

        train_ids = set(manifest["splits"]["train"])
        val_ids = set(manifest["splits"]["val"])
        test_ids = set(manifest["splits"]["test"])

        # No ID overlap
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)

        # Cluster locking: check every cluster belongs to exactly one split
        cluster_to_split: dict[int, str] = {}
        for part, ids in manifest["splits"].items():
            for rid in ids:
                cid = manifest["records_by_id"][rid]["cluster_id"]
                if cid in cluster_to_split:
                    self.assertEqual(
                        cluster_to_split[cid],
                        part,
                        f"Cluster {cid} leaked across splits {cluster_to_split[cid]} and {part}!",
                    )
                else:
                    cluster_to_split[cid] = part

        # Proportions approximately 70/15/15
        total = len(records)
        self.assertTrue(0.60 <= len(train_ids) / total <= 0.80)
        self.assertTrue(0.10 <= len(val_ids) / total <= 0.25)
        self.assertTrue(0.10 <= len(test_ids) / total <= 0.25)

    def test_verify_no_leakage_clean(self):
        records = build_records(
            self.pb_data["texts"],
            self.pb_data["labels"],
            prefix="fpb",
            jaccard_threshold=0.80,
        )
        manifest = create_stratified_splits(records, seed=42, ratios=(0.70, 0.15, 0.15), n_folds=5)
        audit = verify_no_leakage(manifest, jaccard_threshold=0.80)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["id_overlap_count"], 0)
        self.assertEqual(audit["hash_overlap_count"], 0)
        self.assertEqual(audit["cluster_overlap_count"], 0)
        self.assertLess(audit["max_cross_split_jaccard"], 0.80)

    def test_verify_no_leakage_detects_id_leak(self):
        records = build_records(self.pb_data["texts"], self.pb_data["labels"], prefix="fpb")
        manifest = create_stratified_splits(records, seed=42)

        # Inject ID leak: copy a val ID into train
        corrupted = copy.deepcopy(manifest)
        corrupted["splits"]["train"].append(corrupted["splits"]["val"][0])

        with self.assertRaises(ValueError) as ctx:
            verify_no_leakage(corrupted)
        self.assertIn("IDs overlap", str(ctx.exception))

    def test_verify_no_leakage_detects_hash_leak(self):
        records = build_records(self.pb_data["texts"], self.pb_data["labels"], prefix="fpb")
        manifest = create_stratified_splits(records, seed=42)

        # Inject hash leak: copy a test hash into train hashes
        corrupted = copy.deepcopy(manifest)
        corrupted["hashes"]["train"].append(corrupted["hashes"]["test"][0])

        with self.assertRaises(ValueError) as ctx:
            verify_no_leakage(corrupted)
        self.assertIn("SHA-256 hash overlap", str(ctx.exception))

    def test_verify_no_leakage_detects_cluster_leak(self):
        records = build_records(self.pb_data["texts"], self.pb_data["labels"], prefix="fpb")
        manifest = create_stratified_splits(records, seed=42)

        # Inject cluster leak by moving a record between splits without changing cluster ID
        corrupted = copy.deepcopy(manifest)
        rid = corrupted["splits"]["train"].pop()
        corrupted["splits"]["val"].append(rid)

        # If rid belongs to a multi-sentence cluster, this triggers cluster overlap
        # Let's explicitly force cluster overlap
        c_id = corrupted["records_by_id"][corrupted["splits"]["train"][0]]["cluster_id"]
        val_first = corrupted["splits"]["val"][0]
        corrupted["records_by_id"][val_first]["cluster_id"] = c_id

        with self.assertRaises(ValueError) as ctx:
            verify_no_leakage(corrupted)
        self.assertIn("cluster_id overlap", str(ctx.exception))

    def test_verify_no_leakage_detects_jaccard_leak(self):
        # Create records where sentence A in train and sentence B in test have Jaccard >= 0.85
        # but with artificial different cluster IDs
        t_a = "Operating profit rose by 15% to 45 million EUR compared to 39 million in the previous year."
        t_b = "Operating profit rose by 15% to 45 million EUR compared to 39 million in the prior year."

        records = [
            {"id": "r_1", "text": t_a, "norm_text": normalize_text(t_a), "sha256": sentence_sha256(t_a), "label": 2, "label_name": "positive", "cluster_id": 1},
            {"id": "r_2", "text": t_b, "norm_text": normalize_text(t_b), "sha256": sentence_sha256(t_b), "label": 2, "label_name": "positive", "cluster_id": 2},
            {"id": "r_3", "text": "Neutral balance sheet disclosure on lease liability.", "norm_text": "neutral balance sheet disclosure on lease liability", "sha256": sentence_sha256("x"), "label": 1, "label_name": "neutral", "cluster_id": 3},
        ]
        manifest = {
            "splits": {"train": ["r_1"], "val": ["r_3"], "test": ["r_2"]},
            "hashes": {"train": [records[0]["sha256"]], "val": [records[2]["sha256"]], "test": [records[1]["sha256"]]},
            "records_by_id": {r["id"]: r for r in records},
        }

        with self.assertRaises(ValueError) as ctx:
            verify_no_leakage(manifest, jaccard_threshold=0.80)
        self.assertIn("Cross-split token Jaccard", str(ctx.exception))

    def test_load_real_datasets_gated(self):
        # Locally without execute=True, must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            load_real_datasets(execute=False)
        self.assertIn("--execute", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
