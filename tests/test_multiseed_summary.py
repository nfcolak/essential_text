"""Tests for multi-seed statistics helpers and the Step 2 summarizer (synthetic runs only)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts import summarize_step2 as summ
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.multiseed import holm_bonferroni, load_runs, mean_sd, seed_averaged_bootstrap

N_ROWS = 60
SEEDS = (42, 100, 2026)


def _probs(y: np.ndarray, accuracy: float, rng: np.random.Generator) -> np.ndarray:
    """Probabilities whose argmax is right for about ``accuracy`` of the rows."""
    pred = y.copy()
    wrong = rng.random(len(y)) > accuracy
    pred[wrong] = (y[wrong] + rng.integers(1, 3, size=int(wrong.sum()))) % 3
    prob = np.full((len(y), 3), 0.1, dtype=np.float32)
    prob[np.arange(len(y)), pred] = 0.8
    return prob


def _write_run(runs_dir, config, model, tag, splits, val_f1=None, seed=42, fold=None, best_epoch=None):
    """Write result.json + predictions.npz in the exact design.md layout.

    ``splits`` maps split name -> (ids, y, prob).
    """
    run_dir = os.path.join(runs_dir, config, model, tag)
    os.makedirs(run_dir)
    metrics = {"val": None, "test": None, "external": None}
    arrays = {}
    for split, (ids, y, prob) in splits.items():
        metrics[split] = compute_classification_metrics(y, np.argmax(prob, axis=1), prob)
        arrays[f"{split}_ids"] = np.asarray(ids)
        arrays[f"{split}_y"] = np.asarray(y, dtype=np.int64)
        arrays[f"{split}_prob"] = np.asarray(prob, dtype=np.float32)
    if val_f1 is not None:
        metrics["val"] = dict(metrics["val"] or compute_classification_metrics([0, 1, 2], [0, 1, 2]))
        metrics["val"]["macro_f1"] = val_f1
    result = {"status": "completed", "stage": "x", "config": config, "model_key": model, "model_name": model,
              "seed": seed, "fold": fold, "hyperparams": {}, "best_epoch": best_epoch, "history": [],
              "metrics": metrics, "n": {}, "device": "cpu", "versions": {}, "manifest_sha256": None,
              "external_sha256": None, "probe": False, "wall_seconds": 1.0, "timestamp": "t"}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    np.savez_compressed(os.path.join(run_dir, "predictions.npz"), **arrays)


def build_full_matrix(runs_dir, logreg_val=0.9, svm_val=0.8, svm_test_acc=0.99, logreg_test_acc=0.5):
    rng = np.random.default_rng(0)
    y_test = rng.integers(0, 3, size=N_ROWS)
    y_ext = rng.integers(0, 3, size=N_ROWS)
    y_val = rng.integers(0, 3, size=N_ROWS)
    test_ids = np.array([f"t{i}" for i in range(N_ROWS)])
    ext_ids = np.array([f"e{i}" for i in range(N_ROWS)])
    val_ids = np.array([f"v{i}" for i in range(N_ROWS)])
    accs = {"vader": 0.4, "loughran_mcdonald": 0.45, "tfidf_logreg": logreg_test_acc, "tfidf_svm": svm_test_acc,
            "bert_base": 0.8, "finbert_mlm": 0.9}

    def splits(model, with_val=True):
        a = accs[model]
        out = {"test": (test_ids, y_test, _probs(y_test, a, rng)), "external": (ext_ids, y_ext, _probs(y_ext, a, rng))}
        if with_val:
            out["val"] = (val_ids, y_val, _probs(y_val, a, rng))
        return out

    for config in summ.CONFIGS:
        for model in summ.BASELINE_MODELS:
            val = {"tfidf_logreg": logreg_val, "tfidf_svm": svm_val}.get(model, 0.5)
            _write_run(runs_dir, config, model, "seed42", splits(model), val_f1=val)
        for model in summ.TRANSFORMER_MODELS:
            for seed in SEEDS:
                _write_run(runs_dir, config, model, f"seed{seed}", splits(model), seed=seed, best_epoch=3)
    for model in summ.MODELS:
        for fold in summ.CV_FOLDS:
            _write_run(runs_dir, summ.CV_CONFIG, model, f"cv_fold{fold}",
                       {"val": (val_ids, y_val, _probs(y_val, accs[model], rng))}, fold=fold)
    _write_run(runs_dir, "external_only", "prosus_finbert", "zeroshot",
               {"external": (ext_ids, y_ext, _probs(y_ext, 0.7, rng))})


class HolmTests(unittest.TestCase):
    def test_hand_computed_example(self):
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
        # sorted: d .005*4=.02, a .01*3=.03, c .03*2=.06, b .04*1=.04 -> monotone .06
        self.assertAlmostEqual(adjusted["d"], 0.02)
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_cap_at_one(self):
        self.assertEqual(holm_bonferroni({"a": 0.6, "b": 0.9})["b"], 1.0)

    def test_mean_sd(self):
        self.assertEqual(mean_sd([0.5]), {"mean": 0.5, "sd": 0.0, "n": 1})
        stats = mean_sd([1.0, 3.0])
        self.assertAlmostEqual(stats["sd"], 2 ** 0.5)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.integers(0, 3, size=200)
        self.right = self.y.copy()
        self.wrong = (self.y + 1) % 3

    def test_always_right_beats_always_wrong(self):
        for metric in ("macro_f1", "accuracy"):
            res = seed_averaged_bootstrap(self.y, [self.right], [self.wrong], metric=metric, B=200)
            self.assertGreater(res["delta"], 0)
            self.assertGreater(res["ci_low"], 0)
            self.assertLess(res["p_value"], 0.05)
            self.assertEqual((res["B"], res["n"]), (200, 200))

    def test_reversed_sign_and_multi_seed_list(self):
        res = seed_averaged_bootstrap(self.y, [self.wrong, self.wrong], [self.right], B=200)
        self.assertLess(res["delta"], 0)
        self.assertLess(res["ci_high"], 0)

    def test_identical_models_have_zero_delta(self):
        res = seed_averaged_bootstrap(self.y, [self.right], [self.right], B=200)
        self.assertEqual(res["delta"], 0.0)
        self.assertEqual(res["p_value"], 1.0)

    def test_matches_sklearn_macro_f1_delta(self):
        res = seed_averaged_bootstrap(self.y, [self.right], [self.wrong], B=10)
        ref = (compute_classification_metrics(self.y, self.right)["macro_f1"]
               - compute_classification_metrics(self.y, self.wrong)["macro_f1"])
        self.assertAlmostEqual(res["delta"], ref)

    def test_rejects_bad_metric(self):
        with self.assertRaises(ValueError):
            seed_averaged_bootstrap(self.y, [self.right], [self.wrong], metric="auc", B=10)


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runs = os.path.join(self.tmp, "runs")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.runs)

    def run_main(self, *extra):
        return summ.main(["--runs-dir", self.runs, "--out-dir", self.out, "--bootstrap-samples", "200", *extra])

    def test_load_runs_reads_layout_and_skips_tmp(self):
        build_full_matrix(self.runs)
        os.makedirs(os.path.join(self.runs, "sentences_allagree", "bert_base", "seed7.tmp"))
        runs = load_runs(self.runs)
        self.assertEqual(len(runs), len(summ.planned_runs(SEEDS)))
        run = next(r for r in runs if r["tag"] == "seed42" and r["model_key"] == "bert_base")
        self.assertEqual(run["predictions"]["test"]["prob"].shape, (N_ROWS, 3))

    def test_full_run_writes_all_outputs(self):
        build_full_matrix(self.runs)
        self.assertEqual(self.run_main(), 0)
        for name in ("summary_metrics.json", "statistical_comparisons.json", "attribution.json",
                     "STEP2_REPORT.md", *summ.FIGURES):
            path = os.path.join(self.out, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertGreater(os.path.getsize(path), 0, name)
        with open(os.path.join(self.out, "statistical_comparisons.json"), encoding="utf-8") as handle:
            comps = json.load(handle)
        test_pairs = comps["configs"]["sentences_allagree"]["test"]["pairs"]
        self.assertEqual(len(test_pairs), 15)
        pair = test_pairs["bert_base__vs__finbert_mlm"]
        self.assertEqual(sorted(pair["mcnemar_by_seed"]), ["100", "2026", "42"])
        self.assertGreaterEqual(pair["p_holm"], pair["p_primary"])
        self.assertEqual(len(comps["configs"]["sentences_allagree"]["external"]["pairs"]), 21)
        with open(os.path.join(self.out, "STEP2_REPORT.md"), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("## Protocol", report)
        self.assertIn("prosus_finbert", report)

    def test_single_seed_baseline_list_case(self):
        build_full_matrix(self.runs)
        self.assertEqual(self.run_main(), 0)
        with open(os.path.join(self.out, "statistical_comparisons.json"), encoding="utf-8") as handle:
            comps = json.load(handle)
        pair = comps["configs"]["sentences_allagree"]["test"]["pairs"]["vader__vs__tfidf_svm"]
        self.assertEqual(list(pair["mcnemar_by_seed"]), ["42"])
        self.assertEqual((pair["n_seeds_a"], pair["n_seeds_b"]), (1, 1))

    def test_tfidf_best_chosen_by_validation_not_test(self):
        # svm is far better on test, logreg is better on validation: validation must win.
        build_full_matrix(self.runs, logreg_val=0.9, svm_val=0.8, svm_test_acc=0.99, logreg_test_acc=0.4)
        self.assertEqual(self.run_main(), 0)
        with open(os.path.join(self.out, "attribution.json"), encoding="utf-8") as handle:
            attribution = json.load(handle)
        for config in summ.CONFIGS:
            self.assertEqual(attribution["configs"][config]["tfidf_best"], "tfidf_logreg")
            entry = attribution["configs"][config]["test"]
            self.assertIn("delta_arch", entry)
            self.assertIn("delta_domain", entry)
            self.assertGreater(entry["delta_arch"]["macro_f1"]["delta"], 0)

    def test_missing_run_is_refused_unless_allowed(self):
        build_full_matrix(self.runs)
        shutil.rmtree(os.path.join(self.runs, "sentences_75agree", "bert_base", "seed2026"))
        self.assertEqual(self.run_main(), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out, "summary_metrics.json")))
        self.assertEqual(self.run_main("--allow-partial"), 0)
        with open(os.path.join(self.out, "STEP2_REPORT.md"), encoding="utf-8") as handle:
            self.assertIn("sentences_75agree/bert_base/seed2026", handle.read())

    def test_missing_runs_dir_exits_2(self):
        self.assertEqual(summ.main(["--runs-dir", os.path.join(self.tmp, "nope"), "--out-dir", self.out]), 2)


if __name__ == "__main__":
    unittest.main()
