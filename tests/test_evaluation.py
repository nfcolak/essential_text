"""Unit tests for classification metrics, neutral/negative confusion, and statistical tests."""

from __future__ import annotations

import unittest
import numpy as np

from src.evaluation.metrics import (
    CLASS_NAMES,
    compute_classification_metrics,
    compute_ece,
)
from src.evaluation.stats import (
    mcnemar_test,
    paired_bootstrap_ci,
)


class TestClassificationMetrics(unittest.TestCase):
    """Tests for compute_classification_metrics and compute_ece."""

    def test_compute_classification_metrics_basic(self) -> None:
        # Synthetic dataset with known distributions across 3 classes:
        # 0: negative, 1: neutral, 2: positive
        # True:  [0, 0, 1, 1, 1, 1, 2, 2]
        # Pred:  [0, 1, 0, 0, 1, 2, 2, 2]
        y_true = [0, 0, 1, 1, 1, 1, 2, 2]
        y_pred = [0, 1, 0, 0, 1, 2, 2, 2]

        metrics = compute_classification_metrics(y_true, y_pred)

        # Expected confusion matrix:
        # True 0 (neg): [1 pred neg, 1 pred neu, 0 pred pos]
        # True 1 (neu): [2 pred neg, 1 pred neu, 1 pred pos]
        # True 2 (pos): [0 pred neg, 0 pred neu, 2 pred pos]
        expected_cm = [
            [1, 1, 0],
            [2, 1, 1],
            [0, 0, 2],
        ]
        self.assertEqual(metrics["confusion_matrix"], expected_cm)

        # Accuracy: 4 correct out of 8 = 0.50
        self.assertAlmostEqual(metrics["accuracy"], 0.50, places=5)

        # Neutral-to-negative rate (RQ1):
        # 2 out of 4 true neutrals misclassified as negative -> 2/4 = 0.50
        self.assertAlmostEqual(metrics["neutral_to_negative_rate"], 0.50, places=5)

        # Per-class metrics
        self.assertIn("per_class", metrics)
        for name in CLASS_NAMES:
            self.assertIn(name, metrics["per_class"])
            cls_stat = metrics["per_class"][name]
            self.assertIn("precision", cls_stat)
            self.assertIn("recall", cls_stat)
            self.assertIn("f1", cls_stat)
            self.assertIn("support", cls_stat)

        # Positive class: precision = 2/3, recall = 2/2 = 1.0, f1 = 4/5 = 0.80
        self.assertAlmostEqual(metrics["per_class"]["positive"]["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["per_class"]["positive"]["precision"], 2.0 / 3.0, places=5)
        self.assertEqual(metrics["per_class"]["positive"]["support"], 2)

        # Macro-F1 should be average of class F1s
        f1_neg = metrics["per_class"]["negative"]["f1"]
        f1_neu = metrics["per_class"]["neutral"]["f1"]
        f1_pos = metrics["per_class"]["positive"]["f1"]
        expected_macro = (f1_neg + f1_neu + f1_pos) / 3.0
        self.assertAlmostEqual(metrics["macro_f1"], expected_macro, places=5)

        # Without probabilities, ece is None
        self.assertIsNone(metrics["ece"])

    def test_compute_classification_metrics_with_ece(self) -> None:
        y_true = [0, 1, 2, 1]
        y_pred = [0, 1, 2, 0]
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
        ])

        metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)
        self.assertIsNotNone(metrics["ece"])
        self.assertIsInstance(metrics["ece"], float)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)

    def test_ece_perfect_and_worst_calibration(self) -> None:
        # Perfect calibration: 100% confidence on all correct labels
        y_true = [0, 1, 2]
        prob_perfect = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        ece_perf = compute_ece(y_true, prob_perfect)
        self.assertAlmostEqual(ece_perf, 0.0, places=5)

        # Worst calibration: 100% confidence on incorrect labels
        prob_wrong = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        ece_wrong = compute_ece(y_true, prob_wrong)
        self.assertAlmostEqual(ece_wrong, 1.0, places=5)

    def test_zero_division_safety(self) -> None:
        # Class 2 (positive) is never in ground truth and never predicted
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["per_class"]["positive"]["support"], 0)
        self.assertEqual(metrics["per_class"]["positive"]["f1"], 0.0)
        self.assertIsInstance(metrics["macro_f1"], float)

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            compute_classification_metrics([0, 1], [0])

        with self.assertRaises(ValueError):
            compute_classification_metrics([], [])

        with self.assertRaises(ValueError):
            compute_classification_metrics([0, 1], [0, 1], y_prob=np.zeros((3, 3)))


class TestMcNemarTest(unittest.TestCase):
    """Tests for mcnemar_test."""

    def test_identical_models(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        res = mcnemar_test(y_true, y_pred, y_pred)
        self.assertEqual(res["n10"], 0)
        self.assertEqual(res["n01"], 0)
        self.assertEqual(res["chi2"], 0.0)
        self.assertEqual(res["p_value"], 1.0)

    def test_symmetric_discordant(self) -> None:
        # 2 cases where A is right and B is wrong
        # 2 cases where A is wrong and B is right
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred_a = [0, 0, 0, 0, 2, 2]  # A correct on 0, 1, 4, 5 (wrong on 2, 3)
        y_pred_b = [1, 1, 1, 1, 2, 2]  # B correct on 2, 3, 4, 5 (wrong on 0, 1)

        res = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertEqual(res["n10"], 2)
        self.assertEqual(res["n01"], 2)
        self.assertEqual(res["chi2"], 0.0)
        self.assertEqual(res["p_value"], 1.0)

    def test_asymmetric_discordant_known_statistic(self) -> None:
        # Construct synthetic arrays with n10=25, n01=2, n11=50, n00=23 (total N=100)
        y_true = [0] * 100
        y_pred_a = [0] * 75 + [1] * 25  # correct on first 75
        y_pred_b = [0] * 50 + [1] * 25 + [0] * 2 + [1] * 23

        # Breakdown:
        # idx 0..49 (50): A=0 (right), B=0 (right) -> n11 = 50
        # idx 50..74 (25): A=0 (right), B=1 (wrong) -> n10 = 25
        # idx 75..76 (2): A=1 (wrong), B=0 (right) -> n01 = 2
        # idx 77..99 (23): A=1 (wrong), B=1 (wrong) -> n00 = 23

        res = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertEqual(res["n10"], 25)
        self.assertEqual(res["n01"], 2)
        self.assertEqual(res["n11"], 50)
        self.assertEqual(res["n00"], 23)

        # chi2 = (|25 - 2| - 1)^2 / 27 = 22^2 / 27 = 484 / 27 = 17.9259259...
        expected_chi2 = (22.0 ** 2) / 27.0
        self.assertAlmostEqual(res["chi2"], expected_chi2, places=5)
        self.assertAlmostEqual(res["statistic"], expected_chi2, places=5)
        self.assertLess(res["p_value"], 0.001)

    def test_input_validation_and_empty(self) -> None:
        with self.assertRaises(ValueError):
            mcnemar_test([0, 1], [0], [0, 1])

        res_empty = mcnemar_test([], [], [])
        self.assertEqual(res_empty["chi2"], 0.0)
        self.assertEqual(res_empty["p_value"], 1.0)


class TestPairedBootstrapCI(unittest.TestCase):
    """Tests for paired_bootstrap_ci."""

    def test_identical_predictions(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        res = paired_bootstrap_ci(y_true, y_pred, y_pred, B=500, seed=42)

        self.assertEqual(res["delta_macro_f1"], 0.0)
        self.assertEqual(res["ci_low"], 0.0)
        self.assertEqual(res["ci_high"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["b_samples"], 500)

    def test_superior_model_significance(self) -> None:
        # Model A gets 100% correct, Model B gets 0% correct
        y_true = [0, 1, 2] * 20
        y_pred_a = [0, 1, 2] * 20
        y_pred_b = [1, 2, 0] * 20

        res = paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, B=1000, seed=42)
        self.assertGreater(res["delta_macro_f1"], 0.9)
        self.assertGreater(res["ci_low"], 0.8)
        self.assertLess(res["p_value"], 0.01)

    def test_reproducibility_with_seed(self) -> None:
        rng = np.random.default_rng(123)
        y_true = rng.integers(0, 3, size=60)
        y_pred_a = rng.integers(0, 3, size=60)
        y_pred_b = rng.integers(0, 3, size=60)

        res1 = paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, B=500, seed=42)
        res2 = paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, B=500, seed=42)

        self.assertEqual(res1["delta_macro_f1"], res2["delta_macro_f1"])
        self.assertEqual(res1["ci_low"], res2["ci_low"])
        self.assertEqual(res1["ci_high"], res2["ci_high"])
        self.assertEqual(res1["p_value"], res2["p_value"])

    def test_default_b_samples(self) -> None:
        # Verify default B=10000 runs fast (< 0.5s) and reports b_samples=10000
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 3, size=50)
        y_pred_a = rng.integers(0, 3, size=50)
        y_pred_b = rng.integers(0, 3, size=50)

        res = paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, seed=42)
        self.assertEqual(res["b_samples"], 10000)
        self.assertIn("ci_95", res)
        self.assertEqual(len(res["ci_95"]), 2)
        self.assertLessEqual(res["ci_low"], res["ci_high"])

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([0, 1], [0], [0, 1])

        res_empty = paired_bootstrap_ci([], [], [], B=100)
        self.assertEqual(res_empty["delta_macro_f1"], 0.0)
        self.assertEqual(res_empty["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
