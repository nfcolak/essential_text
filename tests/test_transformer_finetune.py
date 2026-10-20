from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.transformer.finetune import (
    FinetuneConfig,
    canonical_label_permutation,
    finetune_and_predict,
    predict_proba,
    zero_shot_predict,
)

WORDS = ["profit", "rose", "fell", "loss", "sales", "the", "company", "net", "operating", "costs",
         "increased", "decreased", "quarter", "revenue", "is", "was", "not", "expected", "to",
         "grow", "in", "year", "eur", "mn", "from", "a", "of", "and", "will", "remain", "stable"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ","] + WORDS

TRAIN = ["profit rose in the quarter", "net loss increased", "sales is stable", "costs decreased",
         "revenue was not expected to grow", "operating profit will remain stable",
         "the company loss", "sales rose a year", "net sales fell", "the quarter is stable",
         "profit of eur mn", "revenue and sales grew"]
TRAIN_Y = [2, 0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 1]
VAL = ["profit rose", "loss increased", "sales is stable", "revenue fell", "costs rose", "stable year"]
VAL_Y = [2, 0, 1, 0, 0, 1]


class TinyEnv:
    def __init__(self, tmpdir, id2label=None):
        self.vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(self.vocab_path, "w") as fh:
            fh.write("\n".join(VOCAB) + "\n")
        self.id2label = id2label

    def loader(self, name, revision, for_zero_shot):
        cfg = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64, num_labels=3,
                         max_position_embeddings=64)
        if self.id2label is not None:
            cfg.id2label = dict(self.id2label)
            cfg.label2id = {v: k for k, v in self.id2label.items()}
        model = BertForSequenceClassification(cfg)
        tok = BertTokenizerFast(vocab_file=self.vocab_path, do_lower_case=True)
        return model, tok


def make_config(**kw):
    base = dict(batch_size=4, eval_batch_size=8, learning_rate=5e-3, max_epochs=3, patience=2,
                seed=7, device="cpu", max_length=16)
    base.update(kw)
    return FinetuneConfig(**base)


class FinetuneTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.env = TinyEnv(self._tmp.name)

    def run_ft(self, config):
        return finetune_and_predict("tiny", TRAIN, TRAIN_Y, VAL, VAL_Y,
                                    {"test": VAL, "ext": VAL[:3]}, config,
                                    model_loader=self.env.loader)

    def test_same_seed_is_reproducible(self):
        a = self.run_ft(make_config())
        b = self.run_ft(make_config())
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        np.testing.assert_array_equal(a["val_prob"], b["val_prob"])
        np.testing.assert_array_equal(a["eval_probs"]["test"], b["eval_probs"]["test"])

    def test_result_keys_and_probabilities(self):
        res = self.run_ft(make_config())
        self.assertEqual(set(res), {"best_epoch", "history", "val_prob", "eval_probs",
                                    "wall_seconds", "device", "config"})
        self.assertEqual(res["device"], "cpu")
        self.assertEqual(res["config"], make_config().to_dict())
        self.assertGreaterEqual(res["best_epoch"], 1)
        self.assertEqual(set(res["eval_probs"]), {"test", "ext"})
        self.assertEqual(res["val_prob"].shape, (len(VAL), 3))
        self.assertEqual(res["val_prob"].dtype, np.float32)
        self.assertEqual(res["eval_probs"]["ext"].shape, (3, 3))
        for prob in [res["val_prob"]] + list(res["eval_probs"].values()):
            np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)
        for row in res["history"]:
            self.assertEqual(set(row), {"epoch", "train_loss", "val_macro_f1", "seconds"})

    def test_early_stopping_honours_patience(self):
        # lr=0 keeps weights fixed, so validation F1 never improves after epoch 1.
        res = self.run_ft(make_config(learning_rate=0.0, max_epochs=5, patience=2))
        self.assertEqual(len(res["history"]), 3)
        self.assertEqual(res["best_epoch"], 1)
        res = self.run_ft(make_config(learning_rate=0.0, max_epochs=5, patience=1))
        self.assertEqual(len(res["history"]), 2)
        self.assertEqual(res["best_epoch"], 1)

    def test_full_run_without_stopping(self):
        res = self.run_ft(make_config(max_epochs=2, patience=5))
        self.assertEqual(len(res["history"]), 2)


class LabelPermutationTests(unittest.TestCase):
    def test_prosus_order(self):
        self.assertEqual(canonical_label_permutation({0: "positive", 1: "negative", 2: "neutral"}),
                         [1, 2, 0])

    def test_string_keys_and_case(self):
        self.assertEqual(canonical_label_permutation({"0": "Negative", "1": "NEUTRAL", "2": "positive"}),
                         [0, 1, 2])

    def test_invalid_names_raise(self):
        with self.assertRaises(ValueError):
            canonical_label_permutation({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        with self.assertRaises(ValueError):
            canonical_label_permutation({0: "positive", 1: "negative", 2: "mixed"})
        with self.assertRaises(ValueError):
            canonical_label_permutation({0: "positive", 1: "negative", 2: "negative"})


class ZeroShotTests(unittest.TestCase):
    def test_columns_are_reordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            id2label = {0: "positive", 1: "negative", 2: "neutral"}
            env = TinyEnv(tmp, id2label)
            config = make_config()
            texts = VAL
            torch.manual_seed(0)
            model, tok = env.loader("tiny", None, True)
            raw = predict_proba(model, tok, texts, config, torch.device("cpu"))

            def fixed_loader(name, revision, for_zero_shot):
                self.assertTrue(for_zero_shot)
                return model, tok

            out = zero_shot_predict("tiny", texts, config, model_loader=fixed_loader)
            self.assertEqual(out.shape, (len(texts), 3))
            np.testing.assert_allclose(out[:, 0], raw[:, 1], atol=1e-6)  # negative
            np.testing.assert_allclose(out[:, 1], raw[:, 2], atol=1e-6)  # neutral
            np.testing.assert_allclose(out[:, 2], raw[:, 0], atol=1e-6)  # positive
            np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
