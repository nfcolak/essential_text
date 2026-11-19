"""Figures for the tuning study, using synthetic tables only."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import unittest

from scripts import run_step2
from src.analysis.tuning_figures import (
    EXPECTED_FINAL_ROWS,
    EXPECTED_TUNING_ROWS,
    TuningResultsError,
    TuningRow,
    generate_report,
    load_original_fixed_recipe,
    load_study,
    mean_and_sample_sd,
    resolve_results_root,
    select_recorded_winners,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TUNING_GRID = {
    ("bert_base", 2e-5, 16): 0.10,
    ("bert_base", 2e-5, 32): 0.40,
    ("bert_base", 3e-5, 16): 0.70,
    ("bert_base", 3e-5, 32): 0.95,
    ("finbert_mlm", 2e-5, 16): 0.20,
    ("finbert_mlm", 2e-5, 32): 0.93,
    ("finbert_mlm", 3e-5, 16): 0.55,
    ("finbert_mlm", 3e-5, 32): 0.30,
}
FINAL_SCORES = {
    "bert_base": {
        42: (0.90, 0.50),
        100: (0.92, 0.60),
        2026: (0.94, 0.70),
    },
    "finbert_mlm": {
        42: (0.40, 0.20),
        100: (0.50, 0.30),
        2026: (0.60, 0.40),
    },
}


def _rate(value: float) -> str:
    return run_step2.format_learning_rate(value)


def write_tables(root: str, *, archive_run: bool = False, drop_final: bool = False) -> None:
    os.makedirs(root, exist_ok=True)
    winners = {("bert_base", 3e-5, 32), ("finbert_mlm", 2e-5, 32)}
    tuning_fields = [
        "model_key", "learning_rate", "batch_size", "seed",
        "val_macro_f1", "val_accuracy", "selected", "run_dir",
    ]
    tuning_rows = []
    for (model, learning_rate, batch_size), score in TUNING_GRID.items():
        run_dir = f"selection/sentences_allagree/{model}/lr{_rate(learning_rate)}_bs{batch_size}_seed42"
        if archive_run and model == "finbert_mlm":
            run_dir = "archive/invalid_finbert_tokenizer/old/" + run_dir
        tuning_rows.append({
            "model_key": model,
            "learning_rate": _rate(learning_rate),
            "batch_size": str(batch_size),
            "seed": "42",
            "val_macro_f1": f"{score:.6f}",
            "val_accuracy": f"{score:.6f}",
            "selected": "1" if (model, learning_rate, batch_size) in winners else "0",
            "run_dir": run_dir,
        })
    with open(os.path.join(root, "tuning_results.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuning_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(tuning_rows)
    final_fields = [
        "model_key", "learning_rate", "batch_size", "seed", "best_epoch", "val_macro_f1",
        "test_accuracy", "test_macro_f1", "test_f1_negative", "test_f1_neutral", "test_f1_positive",
        "external_accuracy", "external_macro_f1",
        "external_f1_negative", "external_f1_neutral", "external_f1_positive", "run_dir",
    ]
    final_rows = []
    if not drop_final:
        for model, by_seed in FINAL_SCORES.items():
            learning_rate, batch_size = (3e-5, 32) if model == "bert_base" else (2e-5, 32)
            for seed, (test_score, external_score) in by_seed.items():
                final_rows.append({
                    "model_key": model,
                    "learning_rate": _rate(learning_rate),
                    "batch_size": str(batch_size),
                    "seed": str(seed),
                    "best_epoch": "2",
                    "val_macro_f1": "0.500000",
                    "test_accuracy": f"{test_score:.6f}",
                    "test_macro_f1": f"{test_score:.6f}",
                    "test_f1_negative": f"{test_score:.6f}",
                    "test_f1_neutral": f"{test_score:.6f}",
                    "test_f1_positive": f"{test_score:.6f}",
                    "external_accuracy": f"{external_score:.6f}",
                    "external_macro_f1": f"{external_score:.6f}",
                    "external_f1_negative": f"{external_score:.6f}",
                    "external_f1_neutral": f"{external_score:.6f}",
                    "external_f1_positive": f"{external_score:.6f}",
                    "run_dir": f"final/sentences_allagree/{model}/seed{seed}",
                })
    with open(os.path.join(root, "final_results.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)
    selection = {
        "selected": {
            "bert_base": {"learning_rate": 3e-5, "batch_size": 32, "val_macro_f1": 0.95},
            "finbert_mlm": {"learning_rate": 2e-5, "batch_size": 32, "val_macro_f1": 0.93},
        }
    }
    with open(os.path.join(root, "selection.json"), "w", encoding="utf-8") as handle:
        json.dump(selection, handle)


def write_step2(root: str, *, learning_rate: float = 2e-5, batch_size: int = 16) -> None:
    fixed = {
        "bert_base": (0.70, 0.40),
        "finbert_mlm": (0.65, 0.45),
    }
    for model, (test_score, external_score) in fixed.items():
        for seed in (42, 100, 2026):
            run_dir = os.path.join(root, "sentences_allagree", model, f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            payload = {
                "status": "completed",
                "config": "sentences_allagree",
                "model_key": model,
                "seed": seed,
                "hyperparams": {"learning_rate": learning_rate, "batch_size": batch_size},
                "metrics": {
                    "test": {"macro_f1": test_score},
                    "external": {"macro_f1": external_score},
                },
            }
            with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)


class SummaryMathTests(unittest.TestCase):
    def test_sample_standard_deviation_uses_n_minus_one(self):
        summary = mean_and_sample_sd([0.90, 0.92, 0.94])
        self.assertAlmostEqual(summary.mean, 0.92)
        self.assertAlmostEqual(summary.sample_sd, math.sqrt(0.0008 / 2))

    def test_selection_prefers_validation_score_then_lower_rate_then_smaller_batch(self):
        rows = [
            TuningRow("bert_base", 3e-5, 32, 42, 0.50, False, "a"),
            TuningRow("bert_base", 2e-5, 32, 42, 0.50, False, "b"),
            TuningRow("bert_base", 2e-5, 16, 42, 0.50, False, "c"),
            TuningRow("finbert_mlm", 2e-5, 16, 42, 0.20, False, "d"),
            TuningRow("finbert_mlm", 3e-5, 16, 42, 0.80, False, "e"),
        ]
        chosen = select_recorded_winners(rows)
        self.assertEqual((chosen["bert_base"].learning_rate, chosen["bert_base"].batch_size), (2e-5, 16))
        self.assertEqual((chosen["finbert_mlm"].learning_rate, chosen["finbert_mlm"].batch_size), (3e-5, 16))

    def test_archived_rows_and_incomplete_tables_are_rejected(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        archived = os.path.join(tmp, "archived")
        write_tables(archived, archive_run=True)
        with self.assertRaises(TuningResultsError) as caught:
            load_study(archived)
        self.assertIn("archived", str(caught.exception).lower())
        short = os.path.join(tmp, "short")
        write_tables(short, drop_final=True)
        with self.assertRaises(TuningResultsError) as caught:
            load_study(short)
        self.assertIn(str(EXPECTED_FINAL_ROWS), str(caught.exception))

    def test_drive_backup_is_used_only_when_local_tables_do_not_validate(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        local = os.path.join(tmp, "local")
        drive = os.path.join(tmp, "drive")
        write_tables(drive)
        with self.assertRaises(TuningResultsError):
            resolve_results_root(local, None)
        root, note = resolve_results_root(local, drive)
        self.assertEqual(os.path.realpath(root), os.path.realpath(drive))
        self.assertIsNone(note)
        write_tables(local, drop_final=True)
        root, note = resolve_results_root(local, drive)
        self.assertEqual(os.path.realpath(root), os.path.realpath(drive))
        self.assertIn(str(EXPECTED_FINAL_ROWS), note)


class FigureGenerationTests(unittest.TestCase):
    def test_synthetic_results_write_figures_and_the_summary(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        results = os.path.join(tmp, "tuning")
        step2 = os.path.join(tmp, "step2")
        figures = os.path.join(results, "figures")
        write_tables(results)
        write_step2(step2)
        study = load_study(results)
        self.assertEqual(len(study.tuning), EXPECTED_TUNING_ROWS)
        self.assertEqual(len(study.final), EXPECTED_FINAL_ROWS)
        self.assertEqual(study.selected["bert_base"].batch_size, 32)
        self.assertAlmostEqual(study.selected["bert_base"].learning_rate, 3e-5)
        self.assertAlmostEqual(study.selected["finbert_mlm"].val_macro_f1, 0.93)
        report = generate_report(results, figures, step2)
        self.assertTrue(report["original_comparison"])
        for name in (
            "hyperparameter_heatmaps.png",
            "hyperparameter_heatmaps.pdf",
            "final_model_comparison.png",
            "final_model_comparison.pdf",
            "original_vs_tuned.png",
            "original_vs_tuned.pdf",
            "per_seed_performance.png",
            "per_seed_performance.pdf",
            "RESULTS_SUMMARY.md",
        ):
            path = os.path.join(figures, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertGreater(os.path.getsize(path), 500, name)
        with open(os.path.join(figures, "hyperparameter_heatmaps.png"), "rb") as handle:
            self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(figures, "final_model_comparison.pdf"), "rb") as handle:
            self.assertTrue(handle.read(5).startswith(b"%PDF"))
        with open(os.path.join(figures, "RESULTS_SUMMARY.md"), encoding="utf-8") as handle:
            summary = handle.read()
        self.assertIn("0.920 ± 0.020", summary)
        self.assertIn("0.600 ± 0.100", summary)
        self.assertIn("3e-05", summary)
        self.assertIn("![Hyperparameter tuning](results/tuning/figures/hyperparameter_heatmaps.png)", summary)
        self.assertIn("0.050 under the fixed recipe and 0.420 after independent tuning", summary)
        self.assertIn("not a significance test", summary)
        self.assertNotIn("archive/", summary)

    def test_unmatched_step2_files_skip_the_original_figure(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        results = os.path.join(tmp, "tuning")
        step2 = os.path.join(tmp, "step2")
        figures = os.path.join(tmp, "figures")
        write_tables(results)
        write_step2(step2, learning_rate=3e-5, batch_size=32)
        original, reason = load_original_fixed_recipe(step2)
        self.assertIsNone(original)
        self.assertIn("fixed recipe", reason)
        report = generate_report(results, figures, step2)
        self.assertFalse(report["original_comparison"])
        self.assertIn("Figure 3 was not drawn", report["summary_markdown"])
        self.assertFalse(os.path.exists(os.path.join(figures, "original_vs_tuned.png")))
        self.assertTrue(os.path.isfile(os.path.join(figures, "per_seed_performance.png")))

    def test_repository_step2_files_match_the_fixed_recipe_when_present(self):
        step2 = os.path.join(REPO_ROOT, "results", "step2", "runs")
        sample = os.path.join(step2, "sentences_allagree", "bert_base", "seed42", "result.json")
        if not os.path.isfile(sample):
            self.skipTest("Step 2 result files are not in this checkout")
        original, reason = load_original_fixed_recipe(step2)
        self.assertIsNone(reason)
        self.assertIsNotNone(original)
        for model in ("bert_base", "finbert_mlm"):
            self.assertEqual(len(original.test[model]), 3)
            self.assertEqual(len(original.external[model]), 3)
            for seed, test_score, external_score in zip(
                (42, 100, 2026), original.test[model], original.external[model]
            ):
                path = os.path.join(step2, "sentences_allagree", model, f"seed{seed}", "result.json")
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertAlmostEqual(test_score, payload["metrics"]["test"]["macro_f1"])
                self.assertAlmostEqual(external_score, payload["metrics"]["external"]["macro_f1"])
                self.assertTrue(run_step2.is_default_recipe(
                    payload["hyperparams"]["learning_rate"],
                    payload["hyperparams"]["batch_size"],
                ))


if __name__ == "__main__":
    unittest.main()
