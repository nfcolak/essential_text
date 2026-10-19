"""Integration tests for Step 1 end-to-end pipeline and Colab notebook.

Tests:
1. `colab/run_step1_colab.py --mode smoke` in a temporary directory.
2. Verification of generated locked split manifests and zero leakage.
3. Verification of metrics JSON schema, 4 baseline models, and statistical tests.
4. Verification of external holdout cross-dataset deduplication.
5. Verification of Markdown evaluation report structure and RQ1 sections.
6. Verification of `notebooks/step1_data_and_baselines_colab.ipynb` NBFormat v4 structure.
7. Verification of `--mode full` gate requiring `--execute`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

from colab.run_step1_colab import parse_args, run_step1
from src.data.splits import verify_no_leakage

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end pipeline integration test suite."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = os.path.join(self.temp_dir.name, "results")
        self.splits_dir = os.path.join(self.temp_dir.name, "splits")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_smoke_pipeline_end_to_end(self) -> None:
        """Run --mode smoke and assert all outputs conform to specification."""
        args = argparse.Namespace(
            mode="smoke",
            execute=False,
            out_dir=self.out_dir,
            splits_dir=self.splits_dir,
            bootstrap_samples=100,
            seed=42,
            fpb_config="sentences_allagree",
        )

        results = run_step1(args)
        self.assertIsInstance(results, dict)

        # 1. Verify Manifests in splits_dir
        manifest_path = os.path.join(self.splits_dir, "phrasebank_synthetic_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "Split manifest was not created.")
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("splits", manifest)
        self.assertIn("hashes", manifest)
        self.assertIn("cv_folds", manifest)
        self.assertIn("records_by_id", manifest)
        self.assertEqual(len(manifest["cv_folds"]), 5)

        # Verify zero leakage on generated manifest
        audit = verify_no_leakage(manifest, jaccard_threshold=0.80)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["id_overlap_count"], 0)
        self.assertEqual(audit["hash_overlap_count"], 0)
        self.assertEqual(audit["cluster_overlap_count"], 0)
        self.assertLess(audit["max_cross_split_jaccard"], 0.80)

        # 2. Verify Deduplicated External Holdout
        holdout_path = os.path.join(self.splits_dir, "sentfin_synthetic_dedup.json")
        self.assertTrue(os.path.exists(holdout_path), "Holdout file was not created.")
        with open(holdout_path, mode="r", encoding="utf-8") as f:
            holdout_data = json.load(f)
        self.assertIn("stats", holdout_data)
        self.assertIn("records", holdout_data)
        # Check that intentional synthetic duplicates were removed
        self.assertGreater(holdout_data["stats"]["removed_overlap"], 0)
        self.assertEqual(
            holdout_data["stats"]["kept"],
            len(holdout_data["records"]),
        )

        # 3. Verify summary_metrics.json
        summary_path = os.path.join(self.out_dir, "summary_metrics.json")
        self.assertTrue(os.path.exists(summary_path), "summary_metrics.json was not created.")
        with open(summary_path, mode="r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertIn("metadata", summary)
        self.assertIn("leakage_audit", summary)
        self.assertIn("models", summary)

        expected_models = [
            "VADER",
            "Loughran-McDonald",
            "TF-IDF + Logistic Regression",
            "TF-IDF + Linear SVM",
        ]
        for m_name in expected_models:
            self.assertIn(m_name, summary["models"], f"Model {m_name} missing from summary.")
            m_res = summary["models"][m_name]

            # Check test metrics
            tm = m_res["test_metrics"]
            self.assertTrue(0.0 <= tm["accuracy"] <= 1.0)
            self.assertTrue(0.0 <= tm["macro_f1"] <= 1.0)
            self.assertTrue(0.0 <= tm["neutral_to_negative_rate"] <= 1.0)
            self.assertEqual(len(tm["confusion_matrix"]), 3)
            self.assertEqual(len(tm["confusion_matrix"][0]), 3)

            for cls_name in ("negative", "neutral", "positive"):
                self.assertIn(cls_name, tm["per_class"])
                cls_m = tm["per_class"][cls_name]
                self.assertTrue(0.0 <= cls_m["precision"] <= 1.0)
                self.assertTrue(0.0 <= cls_m["recall"] <= 1.0)
                self.assertTrue(0.0 <= cls_m["f1"] <= 1.0)

            # Check 5-fold CV metrics
            cv_m = m_res["cv_metrics"]
            self.assertIn("accuracy_mean", cv_m)
            self.assertIn("macro_f1_mean", cv_m)
            self.assertIn("neutral_to_negative_rate_mean", cv_m)
            self.assertEqual(len(cv_m["folds"]), 5)

        # 4. Verify statistical_comparisons.json
        stats_path = os.path.join(self.out_dir, "statistical_comparisons.json")
        self.assertTrue(os.path.exists(stats_path), "statistical_comparisons.json was not created.")
        with open(stats_path, mode="r", encoding="utf-8") as f:
            stats = json.load(f)

        self.assertIn("comparisons", stats)
        # 4 models -> 6 pairwise comparisons
        self.assertEqual(len(stats["comparisons"]), 6)
        for comp_name, comp_data in stats["comparisons"].items():
            mc = comp_data["mcnemar"]
            bs = comp_data["bootstrap"]
            self.assertIn("p_value", mc)
            self.assertTrue(0.0 <= mc["p_value"] <= 1.0)
            self.assertIn("delta_macro_f1", bs)
            self.assertIn("ci_low", bs)
            self.assertIn("ci_high", bs)
            self.assertLessEqual(bs["ci_low"], bs["ci_high"])
            self.assertTrue(0.0 <= bs["p_value"] <= 1.0)

        # 5. Verify STEP1_REPORT.md
        report_path = os.path.join(self.out_dir, "STEP1_REPORT.md")
        self.assertTrue(os.path.exists(report_path), "STEP1_REPORT.md was not created.")
        with open(report_path, mode="r", encoding="utf-8") as f:
            report_text = f.read()

        self.assertIn("# Step 1 Baseline Evaluation Report", report_text)
        self.assertIn("## 1. Dataset Partitioning & Leakage Audit", report_text)
        self.assertIn("## 2. Held-Out Test Set Performance", report_text)
        self.assertIn("## 3. 5-Fold Cross-Validation Performance", report_text)
        self.assertIn("## 4. Deduplicated External Holdout Performance", report_text)
        self.assertIn("## 5. RQ1 Deep Dive: Neutral-to-Negative Misclassification Analysis", report_text)
        self.assertIn("## 6. Statistical Significance Comparisons", report_text)
        self.assertIn("## 7. Transition to Step 2", report_text)
        for m_name in expected_models:
            self.assertIn(m_name, report_text)

    def test_full_mode_safety_gate(self) -> None:
        """Verify that --mode full without --execute is strictly blocked."""
        args = argparse.Namespace(
            mode="full",
            execute=False,
            out_dir=self.out_dir,
            splits_dir=self.splits_dir,
            bootstrap_samples=100,
            seed=42,
            fpb_config="sentences_allagree",
        )
        with self.assertRaises(RuntimeError) as ctx:
            run_step1(args)
        self.assertIn("requires --execute", str(ctx.exception))

    def test_argument_parser_defaults(self) -> None:
        """Verify CLI argument defaults."""
        args = parse_args([])
        self.assertEqual(args.mode, "smoke")
        self.assertFalse(args.execute)
        self.assertEqual(args.out_dir, "results/step1")
        self.assertEqual(args.splits_dir, "data/splits")
        self.assertIsNone(args.bootstrap_samples)
        self.assertEqual(args.seed, 42)


class TestNotebookStructure(unittest.TestCase):
    """Test Colab notebook validity and cell requirements."""

    def setUp(self) -> None:
        self.notebook_path = REPO_ROOT / "notebooks" / "step1_data_and_baselines_colab.ipynb"

    def test_notebook_file_exists_and_valid_json(self) -> None:
        """Verify notebook exists and conforms to NBFormat v4 schema."""
        self.assertTrue(self.notebook_path.exists(), f"Notebook missing at {self.notebook_path}")
        with open(self.notebook_path, mode="r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4)
        self.assertIn("cells", nb)
        self.assertGreaterEqual(len(nb["cells"]), 7)

    def test_notebook_contains_required_workflow_cells(self) -> None:
        """Verify presence of badge, clone, install, data, audit, baselines, viz, and export."""
        with open(self.notebook_path, mode="r", encoding="utf-8") as f:
            nb = json.load(f)

        all_cell_text = "".join(
            "".join(c.get("source", [])) for c in nb.get("cells", [])
        )

        # 1. Open in Colab badge
        self.assertIn("colab.research.google.com/assets/colab-badge.svg", all_cell_text)
        # 2. Git clone / pull
        self.assertIn("essential_text.git", all_cell_text)
        # 3. Pip install
        self.assertIn("requirements.txt", all_cell_text)
        # 4. Real dataset download & split locking
        self.assertIn("load_real_datasets", all_cell_text)
        self.assertIn("create_stratified_splits", all_cell_text)
        # 5. Leakage audit
        self.assertIn("verify_no_leakage", all_cell_text)
        # 6. Baseline training & evaluation
        self.assertIn("run_step1_colab.py", all_cell_text)
        # 7. RQ1 Neutral-vs-Negative Confusion visualization
        self.assertIn("confusion_matrix", all_cell_text)
        self.assertIn("neutral_to_negative_rate", all_cell_text)
        # 8. Artifact zip export
        self.assertIn("step1_artifacts", all_cell_text)


if __name__ == "__main__":
    unittest.main()
