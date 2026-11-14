from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertTokenizerFast

from src.transformer import finetune as finetune_module
from src.transformer.finetune import (
    FinetuneConfig,
    _build_tokenizer,
    canonical_label_permutation,
    finetune_and_predict,
    load_model_and_tokenizer,
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


def _tiny_bert_config() -> BertConfig:
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )


def _write_vocab(directory: str) -> None:
    with open(os.path.join(directory, "vocab.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(VOCAB) + "\n")


def _checkpoint_tensors(directory: str) -> dict[str, torch.Tensor]:
    safetensors_path = os.path.join(directory, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path)
    return torch.load(os.path.join(directory, "pytorch_model.bin"), map_location="cpu", weights_only=True)


def _strip_model_type(directory: str) -> None:
    config_path = os.path.join(directory, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.pop("model_type", None)
    payload["architectures"] = ["BertForMaskedLM"]
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class CheckpointLoadTests(unittest.TestCase):
    """Loaders must keep checkpoint encoder weights and still run a forward pass."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _legacy_finbert_dir(self) -> str:
        directory = os.path.join(self._tmp.name, "finbert-pretrain")
        os.makedirs(directory)
        model = BertForMaskedLM(_tiny_bert_config())
        with torch.no_grad():
            model.bert.encoder.layer[0].output.dense.weight.fill_(0.37)
            model.cls.predictions.bias.fill_(4.5)
        model.save_pretrained(directory)
        _strip_model_type(directory)
        # The published MLM checkpoint also carries the BERT pooler. Current BertForMaskedLM
        # does not save one, so add a known pooler and require the loader to keep it.
        saved = _checkpoint_tensors(directory)
        hidden = model.config.hidden_size
        saved["bert.pooler.dense.weight"] = torch.full((hidden, hidden), 0.61)
        saved["bert.pooler.dense.bias"] = torch.full((hidden,), -0.2)
        safetensors_path = os.path.join(directory, "model.safetensors")
        bin_path = os.path.join(directory, "pytorch_model.bin")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import save_file

            save_file(saved, safetensors_path)
            if os.path.isfile(bin_path):
                os.remove(bin_path)
        else:
            torch.save(saved, bin_path)
        _write_vocab(directory)
        return directory

    def _bert_base_dir(self) -> str:
        directory = os.path.join(self._tmp.name, "bert-base")
        os.makedirs(directory)
        config = _tiny_bert_config()
        config.num_labels = 3
        model = BertForSequenceClassification(config)
        with torch.no_grad():
            model.bert.encoder.layer[0].output.dense.weight.fill_(0.11)
            model.classifier.weight.fill_(0.22)
        model.save_pretrained(directory)
        _write_vocab(directory)
        return directory

    def _assert_forward(self, model, tokenizer) -> None:
        encoded = tokenizer(["profit rose"], return_tensors="pt")
        self.assertNotIn(tokenizer.unk_token_id, encoded["input_ids"][0].tolist())
        self.assertEqual(int(encoded["attention_mask"].sum()), int(encoded["attention_mask"].numel()))
        logits = model(**encoded).logits
        self.assertEqual(tuple(logits.shape), (1, model.config.num_labels))
        self.assertTrue(torch.isfinite(logits).all())

    def test_legacy_finbert_loads_encoder_weights_and_a_new_head(self):
        directory = self._legacy_finbert_dir()
        saved = _checkpoint_tensors(directory)
        with patch.object(
            finetune_module,
            "_load_legacy_bert_for_classification",
            wraps=finetune_module._load_legacy_bert_for_classification,
        ) as legacy_loader:
            model, tokenizer = load_model_and_tokenizer(directory, num_labels=3)
        legacy_loader.assert_called_once()
        self.assertIsInstance(model, BertForSequenceClassification)
        self.assertEqual(model.config.num_labels, 3)
        loaded = dict(model.named_parameters())
        encoder_names = [name for name in saved if name.startswith("bert.encoder.")]
        self.assertGreater(len(encoder_names), 0)
        for name in encoder_names:
            self.assertIn(name, loaded)
            torch.testing.assert_close(loaded[name], saved[name], atol=0, rtol=0)
        for name in ("bert.pooler.dense.weight", "bert.pooler.dense.bias",
                     "bert.embeddings.word_embeddings.weight"):
            torch.testing.assert_close(loaded[name], saved[name], atol=0, rtol=0)
        self.assertNotIn("classifier.weight", saved)
        self.assertNotIn("classifier.bias", saved)
        self.assertEqual(tuple(loaded["classifier.weight"].shape), (3, model.config.hidden_size))
        fresh = BertForSequenceClassification(model.config)
        self.assertFalse(torch.allclose(
            fresh.bert.encoder.layer[0].output.dense.weight,
            loaded["bert.encoder.layer.0.output.dense.weight"],
        ))
        self.assertFalse(torch.allclose(
            loaded["classifier.weight"],
            torch.full_like(loaded["classifier.weight"], 0.37),
        ))
        self._assert_forward(model, tokenizer)

    def test_bert_base_checkpoint_keeps_auto_loading(self):
        directory = self._bert_base_dir()
        saved = _checkpoint_tensors(directory)
        with patch.object(
            finetune_module, "_load_legacy_bert_for_classification"
        ) as legacy_loader:
            model, tokenizer = load_model_and_tokenizer(directory, num_labels=3)
        legacy_loader.assert_not_called()
        for name, tensor in model.named_parameters():
            self.assertIn(name, saved)
            torch.testing.assert_close(tensor, saved[name], atol=0, rtol=0)
        self._assert_forward(model, tokenizer)


class TokenizerCompatibilityTests(unittest.TestCase):
    """A legacy vocab.txt must stay aligned with the checkpoint embedding matrix."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.vocab_path = os.path.join(self._tmp.name, "vocab.txt")
        self.tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "profit", "ebitda"]
        with open(self.vocab_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.tokens) + "\n")

    def test_transformers5_vocab_argument_loads_the_wordpiece_file(self):
        import transformers

        class VocabArgumentTokenizer:
            def __init__(self, vocab=None, do_lower_case=True, **kwargs):
                if vocab is None:
                    self._tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                else:
                    with open(vocab, encoding="utf-8") as handle:
                        self._tokens = [line.strip() for line in handle if line.strip()]
                self.unk_token_id = self._tokens.index("[UNK]")

            def __len__(self):
                return len(self._tokens)

            def encode(self, text, add_special_tokens=False):
                ids = []
                for word in text.split():
                    ids.append(self._tokens.index(word) if word in self._tokens else self.unk_token_id)
                return ids

        def fail_auto(*args, **kwargs):
            raise ValueError("no tokenizer backend")

        with patch.object(transformers, "AutoTokenizer") as auto, patch.object(
            transformers, "BertTokenizerFast", VocabArgumentTokenizer
        ):
            auto.from_pretrained.side_effect = fail_auto
            tokenizer = _build_tokenizer(
                self._tmp.name, None, expected_vocab_size=len(self.tokens)
            )
        self.assertEqual(len(tokenizer), len(self.tokens))
        self.assertNotEqual(tokenizer.encode("ebitda profit"), [tokenizer.unk_token_id, tokenizer.unk_token_id])
        self.assertEqual(tokenizer.encode("ebitda profit"), [6, 5])

    def test_rejects_a_five_token_tokenizer_when_the_checkpoint_is_larger(self):
        import transformers

        class TinyTokenizer:
            def __len__(self):
                return 5

        def return_tiny(*args, **kwargs):
            return TinyTokenizer()

        with patch.object(transformers, "AutoTokenizer") as auto, patch(
            "src.transformer.finetune._bert_tokenizer_from_vocab_file", return_value=TinyTokenizer()
        ):
            auto.from_pretrained.side_effect = return_tiny
            with self.assertRaises(RuntimeError) as caught:
                _build_tokenizer(self._tmp.name, None, expected_vocab_size=len(self.tokens))
        self.assertIn(str(len(self.tokens)), str(caught.exception))
        self.assertIn("does not match", str(caught.exception))


class FinbertTokenizerProbeTests(unittest.TestCase):
    def _tokenizer(self, size, mapping, unk=1):
        class _Tokenizer:
            unk_token_id = unk

            def __len__(self):
                return size

            def encode(self, text, add_special_tokens=False):
                return list(mapping.get(text, [unk]))

        return _Tokenizer()

    def test_vocab_size_must_be_30873(self):
        from src.transformer.finetune import validate_finbert_tokenizer

        tokenizer = self._tokenizer(5, {"profit": [1], "ebitda": [1], "depreciation": [1]})
        with self.assertRaises(RuntimeError) as caught:
            validate_finbert_tokenizer(tokenizer)
        self.assertIn("30873", str(caught.exception))

    def test_all_three_financial_terms_must_not_be_unk(self):
        from src.transformer.finetune import validate_finbert_tokenizer

        tokenizer = self._tokenizer(30873, {}, unk=100)
        with self.assertRaises(RuntimeError) as caught:
            validate_finbert_tokenizer(tokenizer)
        message = str(caught.exception)
        self.assertIn("profit", message)
        self.assertIn("[UNK]", message)

    def test_a_matching_vocabulary_with_real_terms_is_accepted(self):
        from src.transformer.finetune import (
            tokenizer_check_payload_is_valid,
            validate_finbert_tokenizer,
        )

        tokenizer = self._tokenizer(
            30873,
            {"profit": [358], "ebitda": [515], "depreciation": [785]},
            unk=100,
        )
        payload = validate_finbert_tokenizer(tokenizer)
        self.assertEqual(payload["vocab_size"], 30873)
        self.assertTrue(tokenizer_check_payload_is_valid(payload))
        self.assertEqual(payload["terms"]["ebitda"], [515])

    def test_one_unknown_probe_term_does_not_reject_the_tokenizer(self):
        from src.transformer.finetune import validate_finbert_tokenizer

        tokenizer = self._tokenizer(
            30873,
            {"profit": [358], "depreciation": [785]},
            unk=100,
        )
        payload = validate_finbert_tokenizer(tokenizer)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["terms"]["ebitda"], [100])


if __name__ == "__main__":
    unittest.main()
