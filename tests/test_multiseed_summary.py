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
from src.evaluation.multiseed import (
    holm_bonferroni,
    load_runs,
    mean_sd,
    seed_averaged_bootstrap,
    seed_sentence_bootstrap,
)

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
    accs = {"vader_official": 0.35, "vader_official_tuned": 0.37, "lm_official": 0.38, "lm_official_tuned": 0.41, "vader": 0.4, "loughran_mcdonald": 0.45, "tfidf_logreg": logreg_test_acc, "tfidf_svm": svm_test_acc,
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


class BoundaryMetricTests(unittest.TestCase):
    def test_hand_made_confusion_matrix(self):
        # rows = gold, cols = predicted; neutral recall 6/8, negative precision 3/5
        cells = {(0, 0): 3, (0, 1): 1, (1, 0): 2, (1, 1): 6, (1, 2): 0, (2, 2): 4}
        y, a = [], []
        for (gold, pred), count in cells.items():
            y += [gold] * count
            a += [pred] * count
        y, a = np.array(y), np.array(a)
        b = y.copy()  # perfect: R_neu = 1, P_neg = 1
        for metric, expected in (("neutral_recall", 6 / 8 - 1.0), ("negative_precision", 3 / 5 - 1.0)):
            for fn in (seed_averaged_bootstrap, seed_sentence_bootstrap):
                res = fn(y, [a], [b], metric=metric, B=20)
                self.assertAlmostEqual(res["delta"], expected, msg=f"{fn.__name__} {metric}")

    def test_zero_denominator_is_zero(self):
        y = np.array([0, 2, 2, 0])
        never_neg = np.array([2, 2, 2, 2])
        # no gold neutral -> R_neu 0 for both; model never predicts negative -> P_neg 0
        for metric in ("neutral_recall", "negative_precision"):
            res = seed_averaged_bootstrap(y, [never_neg], [never_neg], metric=metric, B=10)
            self.assertEqual(res["delta"], 0.0)
        res = seed_sentence_bootstrap(y, [y], [never_neg], metric="negative_precision", B=10)
        self.assertEqual(res["delta"], 1.0)


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


class SeedSentenceBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = rng.integers(0, 3, size=300)
        self.right = self.y.copy()
        self.wrong = (self.y + 1) % 3

    def _noisy(self, accuracy, seed):
        rng = np.random.default_rng(seed)
        pred = self.y.copy()
        flip = rng.random(len(self.y)) > accuracy
        pred[flip] = (self.y[flip] + 1) % 3
        return pred

    def test_single_seed_reduces_to_sentence_bootstrap(self):
        a, b = self._noisy(0.8, 10), self._noisy(0.7, 11)
        two = seed_sentence_bootstrap(self.y, [a], [b], B=300, seed=5)
        one = seed_averaged_bootstrap(self.y, [a], [b], B=300, seed=5)
        for key in ("delta", "ci_low", "ci_high", "p_value", "B", "n", "metric"):
            self.assertAlmostEqual(two[key], one[key], places=12, msg=key)
        self.assertEqual(two["method"], "seed+sentence")
        self.assertAlmostEqual(two["per_seed_delta"][0], two["delta"])

    def test_disagreeing_seeds_widen_ci_and_contain_zero(self):
        # model A beats B under two seeds and loses under the third: seed noise dominates
        good, bad = self._noisy(0.85, 1), self._noisy(0.55, 2)
        a_list = [good, bad, good]
        b_list = [self._noisy(0.7, 3)] * 3
        wide = seed_sentence_bootstrap(self.y, a_list, b_list, B=600, seed=7)
        narrow = seed_averaged_bootstrap(self.y, a_list, b_list, B=600, seed=7)
        self.assertGreater(wide["ci_high"] - wide["ci_low"], narrow["ci_high"] - narrow["ci_low"])
        self.assertLess(wide["ci_low"], 0.0)
        self.assertGreater(wide["ci_high"], 0.0)
        self.assertEqual(len(wide["per_seed_delta"]), 3)
        self.assertGreater(max(wide["per_seed_delta"]), 0)
        self.assertLess(min(wide["per_seed_delta"]), 0)

    def test_per_seed_delta_pairing_rules(self):
        a_list = [self.right, self.wrong]
        equal = seed_sentence_bootstrap(self.y, a_list, [self.wrong, self.wrong], B=20)
        self.assertEqual(len(equal["per_seed_delta"]), 2)
        self.assertGreater(equal["per_seed_delta"][0], 0)
        self.assertEqual(equal["per_seed_delta"][1], 0.0)
        unequal = seed_sentence_bootstrap(self.y, a_list, [self.wrong], B=20)
        self.assertEqual(unequal["per_seed_delta"], equal["per_seed_delta"])

    def test_deterministic_and_validates_input(self):
        a, b = self._noisy(0.8, 10), self._noisy(0.7, 11)
        first = seed_sentence_bootstrap(self.y, [a, b], [b], B=50, seed=9)
        self.assertEqual(first, seed_sentence_bootstrap(self.y, [a, b], [b], B=50, seed=9))
        with self.assertRaises(ValueError):
            seed_sentence_bootstrap(self.y, [a], [b], metric="auc", B=10)
        with self.assertRaises(ValueError):
            seed_sentence_bootstrap(self.y, [], [b], B=10)
        with self.assertRaises(ValueError):
            seed_sentence_bootstrap(self.y, [a[:-1]], [b], B=10)


class FamilyAndFormatTests(unittest.TestCase):
    def test_models_list_has_ten_in_design_order(self):
        self.assertEqual(
            list(summ.MODELS),
            ["vader_official", "vader_official_tuned", "lm_official", "lm_official_tuned",
             "vader", "loughran_mcdonald", "tfidf_logreg", "tfidf_svm", "bert_base", "finbert_mlm"],
        )

    def test_holm_is_applied_per_seed_family(self):
        def pair(p42, p100, p2026):
            return {"mcnemar_by_seed": {"42": {"p_value": p42}, "100": {"p_value": p100},
                                        "2026": {"p_value": p2026}}}

        single = {"mcnemar_by_seed": {"42": {"p_value": 0.01}}}
        results = {"x": pair(0.001, 0.5, 0.02), "y": pair(0.02, 0.001, 0.04), "base": single}
        seeds = summ.apply_seed_families(results)
        self.assertEqual(seeds, [42, 100, 2026])
        # family 42: ordered x .001*3, base .01*2, y .02*1 -> .003, .02, .02
        self.assertAlmostEqual(results["x"]["p_holm_by_seed"]["42"], 0.003)
        self.assertAlmostEqual(results["base"]["p_holm_by_seed"]["42"], 0.02)
        # family 100: y .001*3, base .01*2, x .5 -> base keeps its single run in this family
        self.assertAlmostEqual(results["y"]["p_holm_by_seed"]["100"], 0.003)
        self.assertAlmostEqual(results["base"]["p_holm_by_seed"]["100"], 0.02)
        self.assertAlmostEqual(results["x"]["p_holm_by_seed"]["100"], 0.5)
        self.assertEqual((results["x"]["p_min"], results["x"]["p_max"]), (0.001, 0.5))
        self.assertEqual(results["x"]["n_seeds_significant"], 2)  # 0.003 (42), 2026: .02*2 =.04 < .05
        self.assertEqual(results["base"]["n_families"], 3)
        self.assertEqual(results["base"]["p_min"], results["base"]["p_max"])

    def test_single_run_prints_na_sd_and_zero_p_uses_resolution(self):
        self.assertEqual(summ._pm(mean_sd([0.5])), "0.500 (sd n/a)")
        self.assertEqual(summ._pm(mean_sd([0.5, 0.7])), "0.600 ± 0.141")
        self.assertEqual(summ._p_txt(0.0, 10000), "p < 0.0002")
        self.assertEqual(summ._p_txt(0.0123, 10000), "p=0.0123")


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
                     "rq1_boundary.json", "STEP2_REPORT.md", *summ.FIGURES):
            path = os.path.join(self.out, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertGreater(os.path.getsize(path), 0, name)
        with open(os.path.join(self.out, "statistical_comparisons.json"), encoding="utf-8") as handle:
            comps = json.load(handle)
        test_pairs = comps["configs"]["sentences_allagree"]["test"]["pairs"]
        self.assertEqual(len(test_pairs), 45)
        pair = test_pairs["bert_base__vs__finbert_mlm"]
        self.assertEqual(sorted(pair["mcnemar_by_seed"]), ["100", "2026", "42"])
        self.assertNotIn("p_primary", pair)
        self.assertLessEqual(pair["p_min"], pair["p_max"])
        per_seed_p = [v["p_value"] for v in pair["mcnemar_by_seed"].values()]
        self.assertAlmostEqual(pair["p_min"], min(per_seed_p))
        self.assertAlmostEqual(pair["p_max"], max(per_seed_p))
        self.assertEqual(sorted(pair["p_holm_by_seed"]), ["100", "2026", "42"])
        self.assertTrue(0 <= pair["n_seeds_significant"] <= 3)
        self.assertEqual(pair["bootstrap_macro_f1"]["method"], "seed+sentence")
        self.assertEqual(len(comps["configs"]["sentences_allagree"]["external"]["pairs"]), 55)
        with open(os.path.join(self.out, "STEP2_REPORT.md"), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("## RQ1: neutral/negative boundary", report)
        self.assertIn("restricted to gold-neutral sentences", report)
        self.assertIn("## Per-class precision / recall / F1", report)
        with open(os.path.join(self.out, "rq1_boundary.json"), encoding="utf-8") as handle:
            boundary = json.load(handle)
        for config in summ.CONFIGS:
            self.assertEqual(len(boundary["configs"][config]["test"]["pairs"]), 4)
            self.assertEqual(len(boundary["configs"][config]["external"]["pairs"]), 6)
        bpair = boundary["configs"]["sentences_allagree"]["external"]["pairs"]["prosus_finbert__vs__bert_base"]
        self.assertEqual(bpair["delta_neutral_recall"]["metric"], "neutral_recall")
        self.assertEqual(bpair["delta_negative_precision"]["method"], "seed+sentence")
        self.assertLessEqual(bpair["p_min"], bpair["p_max"])
        self.assertGreater(bpair["n_neutral"], 0)
        with open(os.path.join(self.out, "summary_metrics.json"), encoding="utf-8") as handle:
            summary = json.load(handle)
        cell = summary["configs"]["sentences_allagree"]["bert_base"]["test"]["per_class"]["neutral"]
        self.assertEqual(sorted(cell), ["f1", "precision", "recall"])
        self.assertEqual(cell["recall"]["n"], 3)
        self.assertIn("## Protocol", report)
        self.assertIn("prosus_finbert", report)
        self.assertIn("never mixed across levels", report)
        self.assertIn("p_min", report)
        self.assertIn("(sd n/a)", report)
        vader_rows = [ln for ln in report.splitlines() if ln.startswith(("| VADER (official lexicon) | 1 |", "| vader_official | 1 |"))]
        self.assertTrue(vader_rows)
        self.assertTrue(all("±" not in ln and "(sd n/a)" in ln for ln in vader_rows))

    def test_single_seed_baseline_list_case(self):
        build_full_matrix(self.runs)
        self.assertEqual(self.run_main(), 0)
        with open(os.path.join(self.out, "statistical_comparisons.json"), encoding="utf-8") as handle:
            comps = json.load(handle)
        pair = comps["configs"]["sentences_allagree"]["test"]["pairs"]["vader__vs__tfidf_svm"]
        self.assertEqual(list(pair["mcnemar_by_seed"]), ["42"])
        self.assertEqual((pair["n_seeds_a"], pair["n_seeds_b"]), (1, 1))
        # a baseline-only pair enters every seed family with its single run
        self.assertEqual(pair["n_families"], 3)
        self.assertEqual(pair["p_min"], pair["p_max"])
        # the raw p is the same in every family; Holm ranks it against seed-dependent transformer pairs
        self.assertEqual(sorted(pair["p_holm_by_seed"]), ["100", "2026", "42"])

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
            cell = entry["delta_domain"]["macro_f1"]
            self.assertEqual(cell["method"], "seed+sentence")
            self.assertEqual(len(cell["per_seed_delta"]), 3)
            self.assertAlmostEqual(cell["per_seed_delta_mean"], float(np.mean(cell["per_seed_delta"])))
            self.assertLessEqual(cell["n_seeds_same_sign"], 3)
            self.assertEqual(cell["conditional_on_seeds"]["B"], cell["B"])
            self.assertNotIn("per_seed_delta", cell["conditional_on_seeds"])

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
