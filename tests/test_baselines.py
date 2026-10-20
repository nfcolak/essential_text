"""Unit and smoke tests for Tier 1 and Tier 2 baselines on synthetic financial sentences.

Verifies:
1. Train-only vocabulary fitting (TfidfMLBaseline vocabulary contains no test-only tokens).
2. 3-token window negation handling (negation inversion of positive and negative polarity).
3. Domain-specific neutral accounting term handling (cost, liability, depreciation, debt).
4. Valid probability simplex outputs (shape (N, 3), values in [0, 1], rows sum to 1.0).
5. Validation threshold and hyperparameter tuning for Macro-F1.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.baselines.lexicon import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    NEUTRAL_ACCOUNTING_TERMS,
    LoughranMcDonaldBaseline,
    VaderBaseline,
)
from src.baselines.tfidf_ml import TfidfMLBaseline


class TestLexiconBaselines(unittest.TestCase):
    """Test suite for Tier 1 Lexicon baselines (VADER and Loughran-McDonald)."""

    def setUp(self) -> None:
        self.lm = LoughranMcDonaldBaseline()
        self.vader = VaderBaseline()

    def test_neutral_accounting_terms_not_misclassified_as_negative(self) -> None:
        """Verify cost, liability, depreciation, and debt are classified as neutral (1)."""
        accounting_sentences = [
            "Operating cost was 10 million euros in the second quarter.",
            "The company recorded a depreciation expense on its equipment.",
            "Long-term debt and liabilities remained on the balance sheet.",
            "Amortization and general expenses were reported in the fiscal statement.",
        ]

        # Loughran-McDonald checks
        for s in accounting_sentences:
            score = self.lm.score_sentence(s)
            self.assertEqual(
                score,
                0.0,
                f"LM scored neutral accounting sentence as non-neutral ({score}): '{s}'",
            )
        lm_preds = self.lm.predict(accounting_sentences)
        for pred, s in zip(lm_preds, accounting_sentences):
            self.assertEqual(
                pred,
                LABEL_TO_ID["neutral"],
                f"LM misclassified neutral accounting sentence: '{s}', got {ID_TO_LABEL[pred]}",
            )

        # VADER checks
        vader_preds = self.vader.predict(accounting_sentences)
        for pred, s in zip(vader_preds, accounting_sentences):
            self.assertEqual(
                pred,
                LABEL_TO_ID["neutral"],
                f"VADER misclassified neutral accounting sentence: '{s}', got {ID_TO_LABEL[pred]}",
            )

    def test_negation_inversion_positive(self) -> None:
        """Verify negation within 3 tokens inverts positive polarity."""
        pos_sentence = "The division achieved impressive profit growth."
        negated_sentence = "The division did not achieve impressive profit growth."

        # Loughran-McDonald
        lm_pos_score = self.lm.score_sentence(pos_sentence)
        lm_negated_score = self.lm.score_sentence(negated_sentence)
        self.assertGreater(
            lm_pos_score,
            0.0,
            f"Expected positive LM score for '{pos_sentence}', got {lm_pos_score}",
        )
        self.assertLess(
            lm_negated_score,
            0.0,
            f"Expected negative LM score for negated positive '{negated_sentence}', got {lm_negated_score}",
        )

        # VADER
        vader_pos_score = self.vader.score_sentence(pos_sentence)
        vader_negated_score = self.vader.score_sentence(negated_sentence)
        self.assertGreater(
            vader_pos_score,
            0.0,
            f"Expected positive VADER score for '{pos_sentence}', got {vader_pos_score}",
        )
        self.assertLess(
            vader_negated_score,
            vader_pos_score,
            f"Expected lower VADER score after negation: {vader_negated_score} vs {vader_pos_score}",
        )

    def test_negation_inversion_negative(self) -> None:
        """Verify negation within 3 tokens inverts negative polarity."""
        neg_sentence = "The business suffered severe losses."
        negated_sentence = "The business did not suffer severe losses."

        # Loughran-McDonald
        lm_neg_score = self.lm.score_sentence(neg_sentence)
        lm_negated_score = self.lm.score_sentence(negated_sentence)
        self.assertLess(
            lm_neg_score,
            0.0,
            f"Expected negative LM score for '{neg_sentence}', got {lm_neg_score}",
        )
        self.assertGreater(
            lm_negated_score,
            lm_neg_score,
            f"Expected negated negative to have higher score than '{neg_sentence}', got {lm_negated_score}",
        )

        # VADER
        vader_neg_score = self.vader.score_sentence(neg_sentence)
        vader_negated_score = self.vader.score_sentence(negated_sentence)
        self.assertLess(
            vader_neg_score,
            0.0,
            f"Expected negative VADER score for '{neg_sentence}', got {vader_neg_score}",
        )
        self.assertGreater(
            vader_negated_score,
            vader_neg_score,
            f"Expected negated negative to have higher score than '{neg_sentence}', got {vader_negated_score}",
        )

    def test_clause_boundary_resets_negation(self) -> None:
        """Verify punctuation clause boundaries prevent negation leak."""
        sentence = "Production did not stop; profits increased substantially."
        # Even though 'not' appears in the sentence, it is before the semicolon,
        # so 'profits increased' should remain positive.
        lm_score = self.lm.score_sentence(sentence)
        self.assertGreater(
            lm_score,
            0.0,
            f"LM score across clause break should be positive, got {lm_score}",
        )
        vader_score = self.vader.score_sentence(sentence)
        self.assertGreater(
            vader_score,
            0.0,
            f"VADER score across clause break should be positive, got {vader_score}",
        )

    def test_probability_simplex_outputs(self) -> None:
        """Verify predict_proba returns a valid probability simplex (sum to 1.0)."""
        texts = [
            "Profits rose and revenues soared.",
            "The firm reported standard debt and costs.",
            "Severe drop in sales and catastrophic losses.",
        ]

        for model in (self.lm, self.vader):
            probs = model.predict_proba(texts)
            self.assertEqual(probs.shape, (3, 3))
            self.assertTrue(np.all(probs >= 0.0), f"Negative probabilities found in {model}")
            self.assertTrue(np.all(probs <= 1.0), f"Probabilities > 1.0 found in {model}")

            row_sums = probs.sum(axis=1)
            np.testing.assert_allclose(
                row_sums,
                np.ones(3),
                rtol=1e-5,
                err_msg=f"Probabilities do not sum to 1.0 in {model}",
            )

            preds = model.predict(texts)
            expected_argmax = np.argmax(probs, axis=1)
            np.testing.assert_array_equal(
                preds,
                expected_argmax,
                err_msg=f"predict() does not match argmax(predict_proba()) in {model}",
            )

    def test_validation_threshold_tuning(self) -> None:
        """Verify fit() tunes decision thresholds on validation data."""
        val_texts = [
            "Severe losses and bankruptcies occurred.",
            "Revenues dropped and earnings fell.",
            "The company noted balance sheet debt and expenses.",
            "Depreciation and cost were listed in the report.",
            "Record profits and outstanding growth were achieved.",
            "Revenues soared and quarterly earnings exceeded forecasts.",
        ]
        val_labels = [0, 0, 1, 1, 2, 2]

        lm = LoughranMcDonaldBaseline()
        lm.fit(train_texts=[], train_labels=[], val_texts=val_texts, val_labels=val_labels)
        self.assertIsNotNone(lm.best_params_["val_macro_f1"])
        self.assertLess(lm.threshold_neg_, lm.threshold_pos_)

        vader = VaderBaseline()
        vader.fit(train_texts=[], train_labels=[], val_texts=val_texts, val_labels=val_labels)
        self.assertIsNotNone(vader.best_params_["val_macro_f1"])
        self.assertLess(vader.threshold_neg_, vader.threshold_pos_)


class TestTfidfMLBaseline(unittest.TestCase):
    """Test suite for Tier 2 TF-IDF ML baseline (Logistic Regression & Linear SVM)."""

    def setUp(self) -> None:
        self.train_texts = [
            "Operating profit increased significantly across divisions.",
            "Strong revenue growth and improved margins were achieved.",
            "Record quarterly earnings exceeded expectations comfortably.",
            "The firm reported severe losses and declining sales.",
            "Revenues plummeted and bankruptcy risk escalated sharply.",
            "Earnings fell and customer demand weakened substantially.",
            "The corporation held cash and balance sheet debt unchanged.",
            "Depreciation expenses and total costs were accounted for.",
            "Neutral financial report filed with standard regulatory disclosures.",
        ]
        self.train_labels = [2, 2, 2, 0, 0, 0, 1, 1, 1]

        self.val_texts = [
            "Profits rose and net margins grew.",
            "Heavy losses and falling demand hit the division.",
            "Operating cost and debt were filed in line with expectations.",
        ]
        self.val_labels = [2, 0, 1]

    def test_train_only_vocabulary_fitting(self) -> None:
        """Verify vectorizer vocabulary is fitted strictly on training texts."""
        model = TfidfMLBaseline(model_type="logreg")
        model.fit(
            self.train_texts,
            self.train_labels,
            val_texts=self.val_texts,
            val_labels=self.val_labels,
        )

        # Confirm vectorizer vocabulary only contains train tokens
        test_only_tokens = ["unseen_special_token_xyz", "futuristic_crypto_yield_404"]
        test_texts = [
            "Operating profit grew with unseen_special_token_xyz.",
            "Disaster struck with futuristic_crypto_yield_404.",
        ]

        vocab = model.vectorizer.vocabulary_
        for tok in test_only_tokens:
            self.assertNotIn(
                tok,
                vocab,
                f"Leakage detected: test token '{tok}' present in training vocabulary!",
            )

        # Verify model handles test-only tokens cleanly
        probs = model.predict_proba(test_texts)
        self.assertEqual(probs.shape, (2, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-5)

    def test_valid_probability_simplex_logreg(self) -> None:
        """Verify Logistic Regression outputs valid probability simplex."""
        model = TfidfMLBaseline(model_type="logreg")
        model.fit(
            self.train_texts,
            self.train_labels,
            val_texts=self.val_texts,
            val_labels=self.val_labels,
        )

        test_texts = [
            "The division reported massive profit growth.",
            "Severe drop in revenues resulted in net loss.",
            "Debt and depreciation cost were recorded.",
        ]
        probs = model.predict_proba(test_texts)
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-5)

        preds = model.predict(test_texts)
        self.assertEqual(preds.shape, (3,))
        np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))

    def test_valid_probability_simplex_svm(self) -> None:
        """Verify calibrated Linear SVM outputs valid probability simplex."""
        model = TfidfMLBaseline(model_type="svm")
        model.fit(
            self.train_texts,
            self.train_labels,
            val_texts=self.val_texts,
            val_labels=self.val_labels,
        )

        test_texts = [
            "The division reported massive profit growth.",
            "Severe drop in revenues resulted in net loss.",
            "Debt and depreciation cost were recorded.",
        ]
        probs = model.predict_proba(test_texts)
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-5)

        preds = model.predict(test_texts)
        self.assertEqual(preds.shape, (3,))
        np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))

    def test_hyperparameter_sweep_and_best_params(self) -> None:
        """Verify hyperparameter sweep selects C and class_weight on validation."""
        for model_type in ("logreg", "svm"):
            model = TfidfMLBaseline(model_type=model_type)
            model.fit(
                self.train_texts,
                self.train_labels,
                val_texts=self.val_texts,
                val_labels=self.val_labels,
            )

            self.assertIn("C", model.best_params_)
            self.assertIn("class_weight", model.best_params_)
            self.assertIn(model.best_params_["C"], {0.01, 0.1, 1.0, 10.0})
            self.assertIn(model.best_params_["class_weight"], {None, "balanced"})
            self.assertIsNotNone(model.best_params_["val_macro_f1"])

    def test_unfitted_model_raises_error(self) -> None:
        """Verify calling predict on unfitted model raises RuntimeError."""
        model = TfidfMLBaseline(model_type="logreg")
        with self.assertRaises(RuntimeError):
            model.predict(["Profit rose."])


if __name__ == "__main__":
    unittest.main()
