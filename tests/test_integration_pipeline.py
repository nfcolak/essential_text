"""Integration tests for Step 1 end-to-end pipeline and Colab notebook.

Tests:
1. `colab/run_step1_colab.py --mode smoke` in a temporary directory.
2. Verification of generated locked split manifests and zero leakage.
3. Verification of metrics JSON schema, 4 baseline models, and statistical tests.
4. Verification of external holdout cross-dataset deduplication.
5. Verification of Markdown evaluation report structure and RQ1 sections.
6. Verification of `notebooks/step1_data_and_baselines_colab.ipynb` NBFormat v4 structure.
7. Verification of `--mode full` gate requiring `--execute`.
8. Verification of `notebooks/tuning_hyperparameters_colab.ipynb` structure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
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


class TestTuningNotebookStructure(unittest.TestCase):
    """The tuning notebook must call scripts/run_tuning.py and back up completed runs."""

    def setUp(self) -> None:
        self.notebook_path = REPO_ROOT / "notebooks" / "tuning_hyperparameters_colab.ipynb"

    def test_notebook_file_exists_and_valid_json(self) -> None:
        self.assertTrue(self.notebook_path.exists(), f"Notebook missing at {self.notebook_path}")
        with open(self.notebook_path, mode="r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        self.assertEqual(notebook.get("metadata", {}).get("accelerator"), "GPU")

    def test_notebook_follows_setup_and_reuses_tuning_runner(self) -> None:
        with open(self.notebook_path, mode="r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        all_cell_text = "".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        code_text = "".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

        self.assertIn("colab.research.google.com/assets/colab-badge.svg", all_cell_text)
        self.assertIn("essential_text.git", code_text)
        self.assertIn('REPO_DIR = "/content/essential_text"', code_text)
        self.assertIn('BRANCH = "qiyao/hyperparameter-tuning"', code_text)
        self.assertIn('os.path.join(REPO_DIR, ".git")', code_text)
        self.assertIn('["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR]', code_text)
        self.assertIn('["git", "-C", REPO_DIR, "fetch", "origin", BRANCH]', code_text)
        self.assertIn('["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH]', code_text)
        self.assertIn("os.chdir(REPO_DIR)", code_text)
        self.assertIn('Expected git branch {BRANCH} in {REPO_DIR}', code_text)
        self.assertIn('os.path.join(REPO_DIR, "scripts", "run_tuning.py")', code_text)
        self.assertNotIn("origin main", code_text)
        self.assertNotIn("%cd", code_text)
        self.assertNotIn("!git", code_text)
        self.assertIn("requirements.txt", code_text)
        self.assertIn("load_real_datasets", code_text)
        self.assertIn("create_stratified_splits", code_text)
        self.assertIn("verify_no_leakage", code_text)
        self.assertIn("SHA256SUMS", code_text)
        self.assertIn("verify_split_hashes", code_text)
        self.assertIn("drive.mount", code_text)
        self.assertIn("MyDrive/essential_text_tuning", code_text)
        self.assertIn("result.json", code_text)
        self.assertIn(
            '[sys.executable, "scripts/run_tuning.py", "--phase", "all", "--execute"]',
            code_text,
        )
        for override in ("--learning-rate", "--batch-size", "--val-only", "--seeds"):
            self.assertNotIn(override, code_text)

    def _setup_cell_source(self, repo_dir: str, repo_url: str) -> str:
        with open(self.notebook_path, mode="r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn('REPO_DIR = "/content/essential_text"', source)
        source = source.replace('REPO_DIR = "/content/essential_text"', f"REPO_DIR = {repo_dir!r}")
        source = source.replace(
            'REPO_URL = "https://github.com/nfcolak/essential_text.git"',
            f"REPO_URL = {repo_url!r}",
        )
        return source

    def _git(self, *args: str, cwd: str | None = None) -> None:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            self.fail(completed.stderr.strip() or completed.stdout.strip() or f"git {' '.join(args)} failed")

    def _publish_branch(self, remote: str, branch: str, files: dict[str, str]) -> None:
        self._git("init", "--bare", "-b", branch, remote)
        seed = remote + "-seed"
        os.makedirs(seed, exist_ok=True)
        self._git("init", "-b", branch, seed)
        self._git("config", "user.email", "tuning-test@example.com", cwd=seed)
        self._git("config", "user.name", "Tuning Test", cwd=seed)
        for relative, text in files.items():
            path = os.path.join(seed, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            self._git("add", relative, cwd=seed)
        self._git("commit", "-m", "seed", cwd=seed)
        self._git("remote", "add", "origin", remote, cwd=seed)
        self._git("push", "-u", "origin", branch, cwd=seed)

    def test_setup_cell_stays_on_the_absolute_directory(self) -> None:
        branch = "qiyao/hyperparameter-tuning"
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            remote = os.path.join(tmp, "remote.git")
            repo_dir = os.path.join(tmp, "content", "essential_text")
            self._publish_branch(remote, branch, {"scripts/run_tuning.py": "# tuning\n"})
            source = self._setup_cell_source(repo_dir, remote)
            nested = os.path.join(tmp, "decoy", "essential_text")
            os.makedirs(nested)
            nested_import = os.path.join(repo_dir, "essential_text")
            previous_cwd = os.getcwd()
            previous_path = list(sys.path)
            sys.path.append(nested_import)
            try:
                os.chdir(nested)
                exec(compile(source, "<tuning-cell-1>", "exec"), {"__name__": "__main__"})
                self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(repo_dir))
                self.assertEqual(sys.path[0], repo_dir)
                self.assertNotIn(nested_import, sys.path)
                editor = os.path.join(tmp, "editor")
                self._git("clone", "--branch", branch, remote, editor)
                self._git("config", "user.email", "tuning-test@example.com", cwd=editor)
                self._git("config", "user.name", "Tuning Test", cwd=editor)
                with open(os.path.join(editor, "MARKER"), "w", encoding="utf-8") as handle:
                    handle.write("updated\n")
                self._git("add", "MARKER", cwd=editor)
                self._git("commit", "-m", "update", cwd=editor)
                self._git("push", "origin", branch, cwd=editor)
                os.chdir(repo_dir)
                exec(compile(source, "<tuning-cell-1>", "exec"), {"__name__": "__main__"})
                self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(repo_dir))
                self.assertTrue(os.path.isfile(os.path.join(repo_dir, "scripts", "run_tuning.py")))
                self.assertTrue(os.path.isfile(os.path.join(repo_dir, "MARKER")))
                self.assertFalse(os.path.exists(os.path.join(repo_dir, "essential_text")))
                self.assertFalse(os.path.exists(os.path.join(nested, "essential_text")))
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_path

    def test_setup_cell_reports_a_missing_checkout_or_script(self) -> None:
        branch = "qiyao/hyperparameter-tuning"
        previous_cwd = os.getcwd()
        previous_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
                repo_dir = os.path.join(tmp, "content", "essential_text")
                os.makedirs(repo_dir)
                source = self._setup_cell_source(repo_dir, os.path.join(tmp, "unused.git"))
                with self.assertRaises(RuntimeError) as missing_git:
                    exec(compile(source, "<tuning-cell-1>", "exec"), {"__name__": "__main__"})
                self.assertIn(".git", str(missing_git.exception))
                self.assertIn("absolute path", str(missing_git.exception))
                os.rmdir(repo_dir)

                remote = os.path.join(tmp, "remote.git")
                self._publish_branch(remote, branch, {"README.md": "no tuning script\n"})
                source = self._setup_cell_source(repo_dir, remote)
                with self.assertRaises(RuntimeError) as missing_script:
                    exec(compile(source, "<tuning-cell-1>", "exec"), {"__name__": "__main__"})
                self.assertIn("scripts/run_tuning.py", str(missing_script.exception))
                self.assertIn(branch, str(missing_script.exception))

                wrong_remote = os.path.join(tmp, "wrong.git")
                wrong_dir = os.path.join(tmp, "content", "wrong_repo")
                self._publish_branch(wrong_remote, "main", {"scripts/run_tuning.py": "# tuning\n"})
                wrong_source = self._setup_cell_source(wrong_dir, wrong_remote)
                with self.assertRaises(RuntimeError) as wrong_branch:
                    exec(compile(wrong_source, "<tuning-cell-1>", "exec"), {"__name__": "__main__"})
                self.assertIn(branch, str(wrong_branch.exception))
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path


if __name__ == "__main__":
    unittest.main()
