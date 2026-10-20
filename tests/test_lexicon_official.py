"""Tests for src/baselines/lexicon_official.py and src/baselines/registry.py (synthetic inputs only)."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.baselines.lexicon_official import (
    LM_CSV_DEFAULT,
    LMOfficialBaseline,
    LMOfficialTunedBaseline,
    VaderOfficialBaseline,
    VaderOfficialTunedBaseline,
    load_lm_master,
    tune_thresholds,
)
from src.baselines.lexicon import LoughranMcDonaldBaseline
from src.baselines.registry import BASELINE_SPECS, DISPLAY_NAMES, make_baseline

SENTENCES = [
    "Operating profit rose sharply and the outlook is excellent.",
    "The company reported a terrible loss and a dreadful decline.",
    "The meeting will be held on Tuesday at the head office.",
    "",
    "Net sales were EUR 12.5 mn in the first quarter.",
    "Great news: we are happy and delighted!",
    "Awful, horrible, bad results.",
]


class VaderOfficialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = VaderOfficialBaseline().fit(["x"], [1], val_texts=["x"], val_labels=[1])

    def test_fit_returns_self_and_does_not_tune(self):
        self.assertIsInstance(self.model, VaderOfficialBaseline)
        params = self.model.best_params_
        self.assertFalse(params["tuned"])
        self.assertEqual(params["thresholds"], [-0.05, 0.05])
        self.assertEqual(params["lexicon_size"], 7506)

    def test_probabilities_form_a_simplex_and_argmax_equals_predict(self):
        prob = self.model.predict_proba(SENTENCES)
        self.assertEqual(prob.shape, (len(SENTENCES), 3))
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.argmax(prob, axis=1), self.model.predict(SENTENCES))
        np.testing.assert_array_equal(
            np.argmax(prob.astype(np.float32), axis=1), self.model.predict(SENTENCES)
        )

    def test_fixed_thresholds_on_known_signs(self):
        preds = self.model.predict(SENTENCES)
        scores = self.model.score_texts(SENTENCES)
        expected = np.where(scores >= 0.05, 2, np.where(scores <= -0.05, 0, 1))
        np.testing.assert_array_equal(preds, expected)
        self.assertEqual(int(preds[0]), 2)  # positive compound
        self.assertEqual(int(preds[1]), 0)  # negative compound
        self.assertEqual(int(preds[2]), 1)  # no lexicon word
        self.assertEqual(int(preds[3]), 1)  # empty string

    def test_threshold_boundaries_are_inclusive(self):
        scores = np.array([-0.0501, -0.05, -0.0499, 0.0, 0.0499, 0.05, 0.0501])
        from src.baselines.lexicon_official import _rule_probabilities

        prob = _rule_probabilities(scores, -0.05, 0.05)
        np.testing.assert_array_equal(np.argmax(prob.astype(np.float32), axis=1), [0, 0, 1, 1, 1, 2, 2])

    def test_deterministic(self):
        np.testing.assert_array_equal(self.model.predict_proba(SENTENCES), self.model.predict_proba(SENTENCES))


class LMOfficialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LMOfficialBaseline(words=({"profit", "gain", "strong"}, {"loss", "decline", "weak"}))

    def test_plain_counting(self):
        self.assertEqual(self.model.count_sentence("Profit and gain, but a loss."), (2, 1))
        texts = [
            "Profit and gain were strong",          # P=3, N=0 -> positive
            "A loss and a decline, weak demand",    # P=0, N=3 -> negative
            "Profit but also a loss",               # P=1, N=1 -> neutral
            "No sentiment words here",              # neutral
            "",                                     # neutral
            "Profit and a gain despite one loss",   # P=2, N=1 -> positive
        ]
        np.testing.assert_array_equal(self.model.predict(texts), [2, 0, 1, 1, 1, 2])

    def test_no_negation_rule(self):
        # "not" does not flip the count: plain counting, as in the published method
        self.assertEqual(int(self.model.predict(["no profit"])[0]), 2)

    def test_tokenisation_is_case_insensitive_and_alphabetic(self):
        self.assertEqual(self.model.count_sentence("PROFIT, Profit; profit-loss (LOSS)"), (3, 2))

    def test_probabilities_form_a_simplex_and_argmax_equals_predict(self):
        texts = ["profit gain", "loss", "nothing", "", "profit loss", "weak weak decline profit"]
        prob = self.model.predict_proba(texts)
        self.assertEqual(prob.shape, (len(texts), 3))
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.argmax(prob, axis=1), self.model.predict(texts))
        np.testing.assert_array_equal(np.argmax(prob.astype(np.float32), axis=1), self.model.predict(texts))

    def test_best_params_and_fit(self):
        self.assertIs(self.model.fit(["a"], [1]), self.model)
        params = self.model.best_params_
        self.assertFalse(params["tuned"])
        self.assertEqual(params["lexicon_size"], {"positive": 3, "negative": 3})


class LoadLMMasterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lm_master_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "lm.csv")
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            fh.write("Word,Negative,Positive,Uncertainty\n")
            fh.write("ABANDON,2009,0,0\n")
            fh.write("ABLE,0,2009,0\n")
            fh.write("AMBIGUOUS,2009,2011,0\n")
            fh.write("TABLE,0,0,0\n")
            fh.write(",2009,0,0\n")
        with open(self.path, "rb") as fh:
            self.sha = hashlib.sha256(fh.read()).hexdigest()

    def test_loads_members_by_nonzero_column(self):
        positive, negative = load_lm_master(self.path, self.sha)
        self.assertEqual(positive, {"ABLE", "AMBIGUOUS"})
        self.assertEqual(negative, {"ABANDON", "AMBIGUOUS"})

    def test_sha_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            load_lm_master(self.path, "0" * 64)
        self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_sha_check_can_be_disabled(self):
        positive, _ = load_lm_master(self.path, None)
        self.assertIn("ABLE", positive)

    def test_missing_file_error_names_the_url(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_lm_master(os.path.join(self.tmp, "absent.csv"))
        self.assertIn("drive.usercontent.google.com", str(ctx.exception))

    def test_model_loads_from_csv_path(self):
        model = LMOfficialBaseline(lm_path=self.path, expected_sha256=self.sha)
        np.testing.assert_array_equal(model.predict(["able", "abandon", "table"]), [2, 0, 1])

    def test_default_path_constant_is_under_data_raw(self):
        self.assertTrue(LM_CSV_DEFAULT.startswith("data/raw/"))


class TuneThresholdsTest(unittest.TestCase):
    def test_reproduces_loughran_mcdonald_fit(self):
        rng = np.random.default_rng(0)
        scores = np.round(rng.normal(0.0, 0.2, 120), 3)
        y = np.where(scores < -0.1, 0, np.where(scores > 0.12, 2, 1))
        y[::7] = rng.integers(0, 3, len(y[::7]))
        reference = LoughranMcDonaldBaseline()
        reference.score_texts = lambda texts: scores
        reference.fit(["x"], [1], ["t"] * len(scores), list(y))
        t_neg, t_pos, f1 = tune_thresholds(scores, y)
        self.assertEqual(t_neg, reference.best_params_["threshold_neg"])
        self.assertEqual(t_pos, reference.best_params_["threshold_pos"])
        self.assertAlmostEqual(f1, reference.best_params_["val_macro_f1"])

    def test_empty_scores_keep_default(self):
        t_neg, t_pos, _ = tune_thresholds(np.array([]), np.array([], dtype=int))
        self.assertLess(t_neg, t_pos)


class VaderOfficialTunedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = VaderOfficialTunedBaseline()

    def test_no_validation_keeps_published_thresholds(self):
        model = VaderOfficialTunedBaseline().fit(SENTENCES, [1] * len(SENTENCES))
        self.assertEqual((model.threshold_neg_, model.threshold_pos_), (-0.05, 0.05))
        self.assertFalse(model.best_params_["tuned"])

    def test_thresholds_move_toward_a_separable_validation_split(self):
        # Mildly positive/negative compounds labelled neutral force wider thresholds than +/-0.05.
        texts = ["good", "bad", "good result", "bad result", "great", "terrible", "nice", "awful"]
        scores = VaderOfficialTunedBaseline().score_texts(texts)
        self.assertGreater(float(np.abs(scores).min()), 0.05)
        y = [1] * len(texts)
        model = VaderOfficialTunedBaseline().fit(None, None, texts, y)
        self.assertTrue(model.best_params_["tuned"])
        self.assertLess(model.threshold_neg_, -0.05)
        self.assertGreater(model.threshold_pos_, 0.05)
        default_f1 = tune_thresholds(scores, y)[2]
        self.assertGreaterEqual(model.best_params_["val_macro_f1"], default_f1)
        self.assertGreater(float(np.mean(model.predict(texts) == 1)), float(np.mean(scores.__abs__() < 0.05)))

    def test_best_params_schema(self):
        model = VaderOfficialTunedBaseline().fit(None, None, ["good", "bad", "plain text"], [2, 0, 1])
        for key in ("threshold_neg", "threshold_pos", "val_macro_f1", "tuned", "lexicon", "lexicon_size", "negation"):
            self.assertIn(key, model.best_params_)
        self.assertFalse(model.best_params_["negation"])

    def test_probabilities_form_a_simplex_and_argmax_equals_predict(self):
        model = VaderOfficialTunedBaseline().fit(None, None, SENTENCES, [2, 0, 1, 1, 1, 2, 0])
        prob = model.predict_proba(SENTENCES)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.argmax(prob, axis=1), model.predict(SENTENCES))


class LMOfficialTunedTest(unittest.TestCase):
    WORDS = ({"profit", "gain", "strong"}, {"loss", "decline", "weak"})

    def make(self):
        return LMOfficialTunedBaseline(words=self.WORDS)

    def test_negation_flips_a_hit(self):
        model = self.make()
        self.assertEqual(model.count_sentence("not a profit"), (0, 1))
        self.assertEqual(model.count_sentence("no loss"), (1, 0))
        self.assertEqual(model.count_sentence("a profit"), (1, 0))
        # negation further than 3 tokens away does not apply
        self.assertEqual(model.count_sentence("not one two three profit"), (1, 0))
        self.assertEqual(int(model.predict(["no profit"])[0]), 0)

    def test_score_formula(self):
        np.testing.assert_allclose(self.make().score_texts(["profit gain", "loss", "nothing"]), [2 / 3, -0.5, 0.0])

    def test_thresholds_tuned_on_validation_only(self):
        model = self.make()
        self.assertFalse(model.best_params_["tuned"])
        texts = ["profit", "profit loss decline", "nothing here", "loss", "strong gain", "weak decline loss"]
        y = [1, 0, 1, 1, 2, 0]
        self.assertIs(model.fit(["ignored"], [0], texts, y), model)
        self.assertTrue(model.best_params_["tuned"])
        self.assertTrue(model.best_params_["negation"])
        self.assertEqual(model.best_params_["lexicon_size"], {"positive": 3, "negative": 3})
        self.assertEqual(model.best_params_["threshold_neg"], model.threshold_neg_)
        self.assertEqual(model.best_params_["threshold_pos"], model.threshold_pos_)
        self.assertLess(model.threshold_neg_, model.threshold_pos_)
        self.assertGreater(model.best_params_["val_macro_f1"], 0.0)

    def test_no_validation_keeps_default_thresholds(self):
        model = self.make().fit(["a"], [1])
        self.assertEqual((model.threshold_neg_, model.threshold_pos_), (-0.05, 0.05))
        self.assertFalse(model.best_params_["tuned"])

    def test_probabilities_form_a_simplex_and_argmax_equals_predict(self):
        model = self.make().fit(None, None, ["profit gain", "loss", "nothing", "not profit"], [2, 0, 1, 0])
        texts = ["profit gain", "loss", "nothing", "", "profit loss", "not weak decline profit"]
        prob = model.predict_proba(texts)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.argmax(prob, axis=1), model.predict(texts))
        np.testing.assert_array_equal(np.argmax(prob.astype(np.float32), axis=1), model.predict(texts))

    def test_registry_builds_it(self):
        self.assertIsInstance(LMOfficialTunedBaseline(words=self.WORDS), LMOfficialTunedBaseline)


class RegistryTest(unittest.TestCase):
    def test_specs_and_display_names(self):
        self.assertEqual(
            list(BASELINE_SPECS),
            ["vader_official", "vader_official_tuned", "lm_official", "lm_official_tuned", "vader",
             "loughran_mcdonald", "tfidf_logreg", "tfidf_svm"],
        )
        self.assertEqual(BASELINE_SPECS["vader_official_tuned"]["variant"], "official_tuned")
        self.assertEqual(BASELINE_SPECS["lm_official_tuned"]["variant"], "official_tuned")
        self.assertEqual(
            DISPLAY_NAMES["vader_official_tuned"], "VADER (official lexicon, val-tuned thresholds)"
        )
        self.assertEqual(
            DISPLAY_NAMES["lm_official_tuned"],
            "Loughran-McDonald (official dictionary, negation + val-tuned thresholds)",
        )
        self.assertEqual(BASELINE_SPECS["vader_official"]["variant"], "official")
        self.assertEqual(BASELINE_SPECS["vader"]["variant"], "finance_adjusted")
        self.assertEqual(DISPLAY_NAMES["vader_official"], "VADER (official lexicon)")
        for key in ("bert_base", "finbert_mlm", "prosus_finbert"):
            self.assertIn(key, DISPLAY_NAMES)

    def test_make_baseline_builds_each_key(self):
        self.assertIsInstance(make_baseline("vader_official"), VaderOfficialBaseline)
        self.assertIsInstance(make_baseline("vader_official_tuned"), VaderOfficialTunedBaseline)
        for key in ("vader", "loughran_mcdonald", "tfidf_logreg", "tfidf_svm"):
            model = make_baseline(key, seed=7)
            self.assertTrue(hasattr(model, "fit") and hasattr(model, "predict_proba"), key)

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            make_baseline("nope")


if __name__ == "__main__":
    unittest.main()
