"""Tests for the extension analyses: SEntFiN entity subsets and the tokenizer comparison (synthetic data only)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from transformers import BertTokenizerFast

from scripts import analyze_extra_external as ax
from scripts import analyze_sentfin_entities as ent
from scripts import analyze_tokenizers as tok
from scripts import prepare_extra_external as prep
from scripts import run_extra_external as rx
from scripts import summarize_step2 as summ
from src.evaluation.multiseed import load_runs
from tests.test_multiseed_summary import N_ROWS, build_full_matrix


class EntitySubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.runs = os.path.join(self.tmp, "runs")
        build_full_matrix(self.runs)
        # e0..e59: every third headline names two entities
        self.counts = {f"e{i}": (2 if i % 3 == 0 else 1) for i in range(N_ROWS)}
        self.texts = {f"e{i}": ("Company reports loss" if i % 4 == 0 else "Company opens new plant")
                      for i in range(N_ROWS)}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_subset_masks_partition_the_holdout(self):
        ids = np.array(sorted(self.counts))
        masks = ent.subset_masks(ids, self.counts)
        self.assertTrue(masks["all"].all())
        self.assertFalse((masks["single"] & masks["multi"]).any())
        self.assertEqual(int(masks["single"].sum() + masks["multi"].sum()), N_ROWS)
        self.assertEqual(int(masks["multi"].sum()), 20)

    def test_subset_masks_reject_unknown_or_empty(self):
        with self.assertRaises(ValueError):
            ent.subset_masks(np.array(["e0", "missing"]), self.counts)
        with self.assertRaises(ValueError):
            ent.subset_masks(np.array(["x"]), {"x": 0})

    def test_build_matches_full_set_and_subsets(self):
        table = summ.RunTable(load_runs(self.runs))
        out = ent.build(table, self.counts, self.texts, B=50, seed=0)
        self.assertEqual(out["n"], {"all": N_ROWS, "single": 40, "multi": 20})
        cfg = out["configs"]["sentences_allagree"]
        # the "all" subset reproduces the stored external Macro-F1 of the seed42 run
        run = table.get("sentences_allagree", "tfidf_svm", "seed42")
        self.assertAlmostEqual(cfg["models"]["tfidf_svm"]["all"]["macro_f1"]["mean"],
                               run["metrics"]["external"]["macro_f1"], places=6)
        self.assertEqual(cfg["models"]["bert_base"]["single"]["n_seeds"], 3)
        self.assertIn("prosus_finbert", cfg["models"])
        for subset in ("single", "multi"):
            self.assertIn("delta_arch", cfg["attribution"][subset])
            self.assertIn("delta_domain", cfg["attribution"][subset])
        self.assertIn("with_loss", cfg["rq1_single_entity"])
        self.assertLessEqual(cfg["rq1_single_entity"]["n_neutral_term_loss_only"],
                             cfg["rq1_single_entity"]["n_neutral_term"])


class TokenizerTests(unittest.TestCase):
    VOCAB_A = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]", "cost", "profit", "de", "##pre",
               "##ciation", "rose"]
    VOCAB_B = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cost", "profit", "depreciation", "rose", "ebitda"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _tokenizer(self, vocab):
        path = os.path.join(self.tmp, f"vocab{len(vocab)}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(vocab) + "\n")
        return BertTokenizerFast(vocab_file=path, do_lower_case=True)

    def test_vocab_overlap_ignores_special_and_unused(self):
        ov = tok.vocab_overlap(self.VOCAB_A, self.VOCAB_B)
        self.assertEqual(ov["all"]["a"], 6)  # cost profit de ##pre ##ciation rose
        self.assertEqual(ov["all"]["b"], 5)
        self.assertEqual(ov["all"]["shared"], 3)
        self.assertEqual(ov["continuation_pieces"]["a"], 2)
        self.assertEqual(ov["continuation_pieces"]["shared"], 0)
        self.assertAlmostEqual(ov["all"]["jaccard"], 3 / 8)

    def test_tokenize_stats_counts_pieces_splits_and_unk(self):
        a = self._tokenizer(self.VOCAB_A)
        b = self._tokenizer(self.VOCAB_B)
        texts = ["Depreciation rose", "cost xyz"]
        sa, na = tok.tokenize_stats(a, texts)
        sb, nb = tok.tokenize_stats(b, texts)
        self.assertEqual(na.tolist(), [4, 2])  # de ##pre ##ciation rose | cost [UNK]
        self.assertEqual(nb.tolist(), [2, 2])
        self.assertAlmostEqual(sa["share_words_split"], 1 / 4)
        self.assertAlmostEqual(sb["share_words_split"], 0.0)
        self.assertAlmostEqual(sa["unk_rate"], 1 / 6)
        self.assertAlmostEqual(sb["pieces_per_word"], 1.0)
        groups = tok.tokenization_groups(na, nb)
        self.assertEqual(groups["finbert_fewer"].tolist(), [True, False])
        self.assertEqual(groups["same"].tolist(), [False, True])


class ExtraExternalPrepTests(unittest.TestCase):
    def test_fiqa_labels_map_and_conflicts_drop(self):
        rows = [
            {"sentence": "Shares jump", "label": "0", "format": "headline", "_split": "train"},
            {"sentence": "Shares jump", "label": "0", "format": "headline", "_split": "train"},  # 2nd aspect
            {"sentence": "Mixed view", "label": "0", "format": "post", "_split": "test"},
            {"sentence": "Mixed view", "label": "2", "format": "post", "_split": "test"},
            {"sentence": "Flat day", "label": "1", "format": "post", "_split": "test"},
            {"sentence": "Stock slumps", "label": "2", "format": "headline", "_split": "test"},
        ]
        records, stats = prep.parse_fiqa(rows)
        self.assertEqual(stats["conflicting_label_dropped"], 1)
        self.assertEqual({r["text"]: r["label"] for r in records},
                         {"Shares jump": 2, "Flat day": 1, "Stock slumps": 0})

    def test_twitter_label_map_and_url_removal(self):
        rows = [{"text": "$X - X cut https://t.co/abc", "label": "0"},
                {"text": "$Y beats https://t.co/d", "label": "1"},
                {"text": "https://t.co/only", "label": "2"},
                {"text": "$Z flat", "label": "2"}]
        records, stats = prep.parse_twitter(rows)
        self.assertEqual(stats["empty_after_url_removal"], 1)
        self.assertEqual([(r["text"], r["label"]) for r in records],
                         [("$X - X cut", 0), ("$Y beats", 2), ("$Z flat", 1)])

    def test_internal_duplicates(self):
        recs = [{"text": "A b", "label": 1}, {"text": "a  B!", "label": 1}, {"text": "c", "label": 0},
                {"text": "C", "label": 2}, {"text": "d", "label": 2}]
        kept, dup, conflict = prep.drop_internal_duplicates(recs)
        self.assertEqual([r["text"] for r in kept], ["A b", "d"])
        self.assertEqual((dup, conflict), (1, 2))

    def test_runner_plan_matches_step2_without_cv(self):
        plan = rx.build_plan()
        self.assertEqual(len(plan), 24 + 18 + 1)
        self.assertEqual({r.stage for r in plan}, {"baselines", "finetune", "zeroshot"})


class ExtraExternalAnalysisTests(unittest.TestCase):
    def test_build_on_synthetic_runs_with_extra_splits(self):
        tmp = tempfile.mkdtemp()
        try:
            runs = os.path.join(tmp, "runs")
            build_full_matrix(runs)
            # add fiqa / twitter blocks (copies of the external block) to every stored run
            for root, _, files in os.walk(runs):
                if "predictions.npz" in files:
                    path = os.path.join(root, "predictions.npz")
                    with np.load(path) as data:
                        arrays = {k: data[k] for k in data.files}
                    if "external_prob" in arrays:
                        for name in ("fiqa", "twitter"):
                            ids = np.array([f"{name}_{i:05d}" for i in range(len(arrays["external_ids"]))])
                            arrays[f"{name}_ids"] = ids
                            arrays[f"{name}_y"] = arrays["external_y"]
                            arrays[f"{name}_prob"] = arrays["external_prob"]
                    np.savez_compressed(path, **arrays)
            fmt = {f"fiqa_{i:05d}": ("headline" if i % 2 else "post") for i in range(N_ROWS)}
            table = summ.RunTable(ax.load_runs(runs))
            out = ax.build(table, fmt, B=30, seed=0)
            cfg = out["configs"]["sentences_allagree"]
            self.assertEqual(cfg["sets"]["fiqa"]["n"], N_ROWS)
            self.assertEqual(cfg["sets"]["fiqa_headline"]["n"] + cfg["sets"]["fiqa_post"]["n"], N_ROWS)
            ext = cfg["sets"]["sentfin"]["models"]["tfidf_svm"]["macro_f1"]["mean"]
            self.assertAlmostEqual(cfg["sets"]["fiqa"]["models"]["tfidf_svm"]["macro_f1"]["mean"], ext)
            self.assertIn("prosus_finbert", cfg["sets"]["twitter"]["models"])
            self.assertIn("delta_domain", cfg["sets"]["twitter"]["attribution"])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
