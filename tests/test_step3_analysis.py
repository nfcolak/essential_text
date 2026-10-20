"""Tests for Step 3 agreement, consensus, per-phenomenon statistics and the analysis script."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest

import numpy as np

from scripts import run_step3_analysis
from src.analysis.attribution import (
    PHENOMENA,
    agreement_table,
    cohen_kappa,
    consensus_tags,
    holm_adjust,
    loss_only_ids,
    normalise_cue,
    phenomenon_attribution,
    phenomenon_error_rates,
    read_annotations,
    rq1_cue_breakdown,
    rq1_terminology_comparison,
)

CONFIG = "sentences_50agree"
ZERO = {name: 0 for name in PHENOMENA}


def _dump(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)


def _tags(**on: int) -> dict:
    return {**ZERO, **on}


def _write_ann(path: str, rows: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["diag_id", *PHENOMENA, "note"])
        for diag_id, tags in rows.items():
            writer.writerow([diag_id, *[tags[p] for p in PHENOMENA], ""])


class TestKappa(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        self.assertAlmostEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_both_constant_equal_is_one(self) -> None:
        self.assertEqual(cohen_kappa([0, 0, 0], [0, 0, 0]), 1.0)

    def test_chance_agreement_is_zero(self) -> None:
        # observed 0.5, expected 0.5
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_hand_computed_value(self) -> None:
        # po = 3/4, pe = 0.5*0.25 + 0.5*0.75 = 0.5 -> kappa = 0.5
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)

    def test_complete_disagreement_is_negative(self) -> None:
        self.assertAlmostEqual(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]), -1.0)

    def test_empty_is_nan(self) -> None:
        self.assertTrue(math.isnan(cohen_kappa([], [])))


class TestAgreementAndConsensus(unittest.TestCase):
    def setUp(self) -> None:
        self.a = {"d1": _tags(negation=1), "d2": _tags(negation=1), "d3": _tags(), "d4": _tags()}
        self.b = {"d1": _tags(negation=1), "d2": _tags(), "d3": _tags(hedging=1), "d4": _tags()}

    def test_consensus_both_vs_either(self) -> None:
        both = consensus_tags(self.a, self.b, "both")
        either = consensus_tags(self.a, self.b, "either")
        self.assertEqual([both[d]["negation"] for d in ("d1", "d2", "d3", "d4")], [1, 0, 0, 0])
        self.assertEqual([either[d]["negation"] for d in ("d1", "d2", "d3", "d4")], [1, 1, 0, 0])
        self.assertEqual(sum(t["hedging"] for t in both.values()), 0)
        self.assertEqual(sum(t["hedging"] for t in either.values()), 1)
        with self.assertRaises(ValueError):
            consensus_tags(self.a, self.b, "majority")

    def test_agreement_table_values(self) -> None:
        rules = {"d1": _tags(negation=1), "d2": _tags(negation=1), "d3": _tags(), "d4": _tags()}
        table = agreement_table(self.a, self.b, rules)
        neg = table["negation"]
        self.assertAlmostEqual(neg["kappa_ab"], 0.5)
        self.assertAlmostEqual(neg["kappa_a_rule"], 1.0)
        self.assertAlmostEqual(neg["raw_agreement_ab"], 0.75)
        self.assertAlmostEqual(neg["prevalence_a"], 0.5)
        self.assertAlmostEqual(neg["prevalence_b"], 0.25)
        self.assertAlmostEqual(neg["prevalence_rule"], 0.5)
        self.assertEqual(table["numerical_comparison"]["kappa_ab"], 1.0)  # both constant 0
        # both-consensus negation = d1 only; the rules mark d1 and d2 -> one difference
        self.assertEqual(neg["n_consensus_vs_rule_diff"], 1)
        self.assertEqual(table["hedging"]["n_consensus_vs_rule_diff"], 0)

    def test_read_annotations_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.csv")
            _write_ann(good, self.a)
            self.assertEqual(read_annotations(good)["d1"]["negation"], 1)
            bad = os.path.join(tmp, "bad.csv")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("diag_id," + ",".join(PHENOMENA) + "\nd1,2,0,0,0,0\n")
            with self.assertRaises(ValueError):
                read_annotations(bad)
            short = os.path.join(tmp, "short.csv")
            with open(short, "w", encoding="utf-8") as handle:
                handle.write("diag_id,negation\nd1,1\n")
            with self.assertRaises(ValueError):
                read_annotations(short)


def _preds(ids: list, wrong: set) -> dict:
    """Gold is 1 for every id; wrong ids get prediction 0."""
    return {i: (0 if i in wrong else 1) for i in ids}


class TestErrorRatesAndAttribution(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"r{i}" for i in range(10)]
        self.y = {i: 1 for i in self.ids}
        # negation tag on r0..r3 (n=4); hedging on r0..r24 impossible, so none
        self.tags = {i: _tags(negation=int(k < 4)) for k, i in enumerate(self.ids)}
        self.preds = {
            "tfidf_svm": [_preds(self.ids, {"r0", "r1", "r2", "r5"})],
            "bert_base": [_preds(self.ids, {"r0", "r5"}), _preds(self.ids, {"r0", "r1", "r5"})],
            "finbert_mlm": [_preds(self.ids, {"r5"}), _preds(self.ids, {"r5"})],
        }

    def test_error_rate_and_ci(self) -> None:
        rates = phenomenon_error_rates(self.tags, self.y, self.preds, B=500, seed=1)
        neg = rates["negation"]
        self.assertEqual((neg["n_tag1"], neg["n_tag0"]), (4, 6))
        self.assertTrue(neg["small_n"])
        svm = neg["models"]["tfidf_svm"]["tag1"]
        self.assertAlmostEqual(svm["error_rate"], 0.75)  # r0, r1, r2 wrong of r0..r3
        self.assertLessEqual(svm["ci_low"], svm["error_rate"])
        self.assertGreaterEqual(svm["ci_high"], svm["error_rate"])
        bert = neg["models"]["bert_base"]["tag1"]
        self.assertAlmostEqual(bert["error_rate"], 0.375)  # seed mean of 1/4 and 2/4
        self.assertEqual(len(bert["per_seed"]), 2)
        finbert = neg["models"]["finbert_mlm"]["tag1"]
        self.assertEqual((finbert["error_rate"], finbert["ci_low"], finbert["ci_high"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(neg["models"]["tfidf_svm"]["tag0"]["error_rate"], 1 / 6)
        # phenomenon with no tagged sentence
        self.assertEqual(rates["hedging"]["n_tag1"], 0)
        self.assertIsNone(rates["hedging"]["models"]["bert_base"]["tag1"]["error_rate"])

    def test_attribution_deltas_and_small_n(self) -> None:
        attr = phenomenon_attribution(self.tags, self.y, self.preds, "tfidf_svm", B=500, seed=1)
        neg = attr["negation"]
        # accuracy on r0..r3: svm 0.25, bert 0.625, finbert 1.0
        self.assertAlmostEqual(neg["delta_arch"]["delta"], 0.625 - 0.25)
        self.assertAlmostEqual(neg["delta_domain"]["delta"], 1.0 - 0.625)
        self.assertTrue(neg["small_n"])
        self.assertLessEqual(neg["delta_arch"]["ci_low"], neg["delta_arch"]["delta"])
        self.assertGreaterEqual(neg["delta_arch"]["ci_high"], neg["delta_arch"]["delta"])
        # n = 4 < 20: CI only, no p-value and no Holm p
        self.assertIsNone(neg["delta_arch"]["p_value"])
        self.assertIsNone(neg["delta_arch"]["p_holm"])
        self.assertEqual(len(neg["delta_arch"]["per_seed_delta"]), 2)
        self.assertEqual(attr["hedging"]["n"], 0)
        self.assertIsNone(attr["hedging"]["delta_arch"]["delta"])

    def test_small_n_clears_at_twenty(self) -> None:
        ids = [f"s{i}" for i in range(20)]
        tags = {i: _tags(hedging=1) for i in ids}
        y = {i: 1 for i in ids}
        preds = {m: [_preds(ids, set())] for m in ("tfidf_svm", "bert_base", "finbert_mlm")}
        attr = phenomenon_attribution(tags, y, preds, "tfidf_svm", B=50, seed=1)
        self.assertFalse(attr["hedging"]["small_n"])
        self.assertEqual(attr["hedging"]["delta_arch"]["p_value"], 1.0)
        self.assertEqual(attr["hedging"]["delta_arch"]["p_holm"], 1.0)

    def test_balanced_error_and_gold_mix(self) -> None:
        # 10 gold-neutral and 2 gold-negative sentences tagged; the model errs on both negatives
        ids = [f"b{i}" for i in range(12)]
        y = {i: (0 if k < 2 else 1) for k, i in enumerate(ids)}
        tags = {i: _tags(hedging=1) for i in ids}
        preds = {"m": [{i: (1 if k < 2 else 1) for k, i in enumerate(ids)}]}
        rates = phenomenon_error_rates(tags, y, preds, B=200, seed=1)["hedging"]
        entry = rates["models"]["m"]["tag1"]
        self.assertAlmostEqual(entry["error_rate"], 2 / 12)
        self.assertAlmostEqual(entry["balanced_error"], 0.5)  # (1.0 + 0.0) / 2 classes present
        self.assertLessEqual(entry["balanced_ci_low"], entry["balanced_error"])
        self.assertGreaterEqual(entry["balanced_ci_high"], entry["balanced_error"])
        self.assertEqual(rates["gold_mix_tag1"], {"n_neg": 2, "n_neu": 10, "n_pos": 0})
        self.assertEqual(rates["gold_mix_tag0"], {"n_neg": 0, "n_neu": 0, "n_pos": 0})
        self.assertIsNone(rates["models"]["m"]["tag0"]["balanced_error"])

    def test_balanced_equals_raw_with_one_gold_class(self) -> None:
        rates = phenomenon_error_rates(self.tags, self.y, self.preds, B=200, seed=1)
        entry = rates["negation"]["models"]["tfidf_svm"]["tag1"]
        self.assertAlmostEqual(entry["balanced_error"], entry["error_rate"])

    def test_two_level_bootstrap_widens_ci_when_seeds_disagree(self) -> None:
        ids = [f"w{i}" for i in range(40)]
        y = {i: 1 for i in ids}
        tags = {i: _tags(hedging=1) for i in ids}
        wrong_sets = [set(), set(ids[:20]), set(ids[20:])]
        preds_multi = {m: [_preds(ids, w) for w in wrong_sets] for m in ("bert_base",)}
        preds_multi["tfidf_svm"] = [_preds(ids, set())]
        preds_multi["finbert_mlm"] = [_preds(ids, set())]
        multi = phenomenon_error_rates(tags, y, preds_multi, B=2000, seed=3)["hedging"]
        bert = multi["models"]["bert_base"]["tag1"]
        # conditional on the seed mean (sentence-only) the CI is much narrower than seed+sentence
        mean_err = np.mean([[1.0 if i in w else 0.0 for i in ids] for w in wrong_sets], axis=0)
        draws = np.random.default_rng(3).integers(0, 40, size=(2000, 40))
        sentence_only = np.percentile(mean_err[draws].mean(axis=1), [2.5, 97.5])
        self.assertGreater(bert["ci_high"] - bert["ci_low"], sentence_only[1] - sentence_only[0])
        attr = phenomenon_attribution(tags, y, preds_multi, "tfidf_svm", B=2000, seed=3)["hedging"]
        arch = attr["delta_arch"]
        self.assertLess(arch["ci_low"], 0.0)  # seeds disagree in sign -> the interval reaches zero side
        self.assertEqual(len(arch["per_seed_delta"]), 3)

    def test_single_seed_matches_plain_sentence_bootstrap(self) -> None:
        ids = [f"s{i}" for i in range(30)]
        y = {i: 1 for i in ids}
        tags = {i: _tags(hedging=1) for i in ids}
        wrong = set(ids[:6])
        preds = {"m": [_preds(ids, wrong)]}
        rates = phenomenon_error_rates(tags, y, preds, B=1000, seed=5)["hedging"]["models"]["m"]
        entry = rates["tag1"]
        err = np.array([1.0 if i in wrong else 0.0 for i in ids])
        self.assertAlmostEqual(entry["error_rate"], 0.2)
        # plain bootstrap interval of a 6/30 rate is about [0.07, 0.37]
        self.assertTrue(0.03 < entry["ci_low"] < 0.12)
        self.assertTrue(0.30 < entry["ci_high"] < 0.43)
        self.assertEqual(entry["per_seed"], [float(err.mean())])

    def test_holm_adjust(self) -> None:
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.5]), [0.5])
        self.assertEqual(holm_adjust([0.4, 0.9]), [0.8, 0.9])

    def test_holm_within_attribution_table_and_no_p_for_small_n(self) -> None:
        ids = [f"h{i}" for i in range(30)]
        y = {i: 1 for i in ids}
        tags = {i: _tags(hedging=1, negation=int(k < 5)) for k, i in enumerate(ids)}
        preds = {"tfidf_svm": [_preds(ids, set(ids[:12]))],
                 "bert_base": [_preds(ids, set(ids[:3]))],
                 "finbert_mlm": [_preds(ids, set())]}
        attr = phenomenon_attribution(tags, y, preds, "tfidf_svm", B=500, seed=2)
        self.assertIsNone(attr["negation"]["delta_arch"]["p_value"])  # n = 5
        self.assertIsNone(attr["negation"]["delta_arch"]["p_holm"])
        self.assertIsNotNone(attr["negation"]["delta_arch"]["ci_low"])
        tested = [attr["hedging"][k] for k in ("delta_arch", "delta_domain")]
        for t in tested:
            self.assertGreaterEqual(t["p_holm"], t["p_value"])
            self.assertLessEqual(t["p_holm"], 1.0)
        # smallest p is multiplied by the number of tested rows (2), a zero p enters as 2/B
        p_min = max(min(t["p_value"] for t in tested), 2 / 500)
        self.assertAlmostEqual(min(t["p_holm"] for t in tested), min(1.0, 2 * p_min))

    def test_normalise_cue_and_loss_only(self) -> None:
        self.assertEqual(normalise_cue("Losses"), "loss")
        self.assertEqual(normalise_cue("costs"), "cost")
        self.assertEqual(normalise_cue("taxes"), "tax")
        self.assertEqual(normalise_cue("liabilities"), "liability")
        self.assertEqual(normalise_cue("charges"), "charge")
        self.assertEqual(normalise_cue("write-downs"), "write-down")
        self.assertEqual(normalise_cue("loss"), "loss")
        cues = {"a": ["loss"], "b": ["losses", "Loss"], "c": ["loss", "cost"], "d": []}
        self.assertEqual(loss_only_ids(cues), {"a", "b"})

    def test_cue_breakdown_and_excluding_loss(self) -> None:
        ids = [f"c{i}" for i in range(12)]
        y = {i: 1 for i in ids}
        cues = {ids[k]: ["losses"] for k in range(6)}
        cues.update({ids[k]: ["cost"] for k in range(6, 12)})
        # model calls every loss sentence negative and no cost sentence negative
        preds = {"m": [{i: (0 if k < 6 else 1) for k, i in enumerate(ids)}]}
        out = rq1_cue_breakdown(cues, y, preds, B=200, seed=1)
        self.assertEqual(list(out), ["cost", "loss"])  # tie on n -> alphabetical
        self.assertEqual(out["loss"]["n"], 6)
        self.assertAlmostEqual(out["loss"]["models"]["m"]["rate"], 1.0)
        self.assertAlmostEqual(out["cost"]["models"]["m"]["rate"], 0.0)
        self.assertFalse(out["loss"]["na"])
        tags = {i: _tags(financial_terminology=1) for i in ids}
        tags.update({f"z{i}": _tags() for i in range(6)})
        y.update({f"z{i}": 1 for i in range(6)})
        preds["m"][0].update({f"z{i}": 1 for i in range(6)})
        full = rq1_terminology_comparison(tags, y, preds, B=200, seed=1)["m"]
        resid = rq1_terminology_comparison(tags, y, preds, B=200, seed=1,
                                           drop_ids=loss_only_ids(cues))["m"]
        self.assertAlmostEqual(full["term1"]["rate"], 0.5)
        self.assertEqual(resid["term1"]["n"], 6)
        self.assertAlmostEqual(resid["term1"]["rate"], 0.0)

    def test_rq1_small_group_is_na(self) -> None:
        ids = [f"n{i}" for i in range(8)]
        y = {i: 1 for i in ids}
        tags = {i: _tags(financial_terminology=int(k < 3)) for k, i in enumerate(ids)}
        preds = {"m": [{i: 0 for i in ids}]}
        out = rq1_terminology_comparison(tags, y, preds, B=100, seed=1)["m"]
        self.assertTrue(out["term1"]["na"])
        self.assertFalse(out["term0"]["na"])

    def test_rq1_comparison(self) -> None:
        ids = [f"n{i}" for i in range(8)]
        y = {i: 1 for i in ids}
        tags = {i: _tags(financial_terminology=int(k < 4)) for k, i in enumerate(ids)}
        preds = {"m": [{i: (0 if k < 3 else 1) for k, i in enumerate(ids)}]}
        out = rq1_terminology_comparison(tags, y, preds, B=200, seed=1)["m"]
        self.assertAlmostEqual(out["term1"]["rate"], 0.75)
        self.assertAlmostEqual(out["term0"]["rate"], 0.0)
        self.assertAlmostEqual(out["difference"]["delta"], 0.75)
        self.assertEqual((out["term1"]["n"], out["term0"]["n"]), (4, 4))


def _write_run(runs: str, config: str, model: str, tag: str, ids_by_split: dict, labels: dict,
               wrong: set, val_f1: float) -> None:
    run_dir = os.path.join(runs, config, model, tag)
    os.makedirs(run_dir)
    arrays = {}
    for split, ids in ids_by_split.items():
        prob = np.full((len(ids), 3), 0.05, dtype=np.float32)
        for row, rid in enumerate(ids):
            gold = labels[rid]
            pred = (gold + 1) % 3 if rid in wrong else gold
            prob[row, pred] = 0.9
        arrays[f"{split}_ids"] = np.array(ids)
        arrays[f"{split}_y"] = np.array([labels[i] for i in ids], dtype=np.int64)
        arrays[f"{split}_prob"] = prob
    np.savez_compressed(os.path.join(run_dir, "predictions.npz"), **arrays)
    _dump({"status": "completed", "metrics": {"val": {"macro_f1": val_f1}}},
          os.path.join(run_dir, "result.json"))


def _fake_tagger(texts: list) -> list:
    return [_tags(financial_terminology=int("cost" in t), negation=int("not" in t),
                  hedging=int("may" in t)) for t in texts]


class TestEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.diag = os.path.join(root, "diagnostic")
        self.runs = os.path.join(root, "runs")
        self.splits = os.path.join(root, "splits")
        self.out = os.path.join(root, "out")
        for d in (self.diag, self.runs, self.splits):
            os.makedirs(d)
        rng = np.random.default_rng(0)
        words = ["cost may rise", "profit did not fall", "sales grew", "a plain note"]
        self.test_ids = [f"t{i}" for i in range(60)]
        records = {i: {"text": words[k % 4] + f" {k}", "label": int(rng.integers(0, 3))}
                   for k, i in enumerate(self.test_ids)}
        _dump({"splits": {"test": self.test_ids}, "records_by_id": records},
              os.path.join(self.splits, f"{CONFIG}_manifest.json"))
        self.ext_ids = [f"e{i}" for i in range(30)]
        ext = [{"id": i, "text": words[k % 4], "label": int(k % 3)}
               for k, i in enumerate(self.ext_ids)]
        _dump({"stats": {}, "records": ext}, os.path.join(self.splits, "sentfin_dedup.json"))
        labels = {i: r["label"] for i, r in records.items()}
        ext_labels = {r["id"]: r["label"] for r in ext}
        wrong = {
            "vader_official": set(self.test_ids[::6]), "lm_official": set(self.test_ids[::8]),
            "vader": set(self.test_ids[::2]), "loughran_mcdonald": set(self.test_ids[::3]),
            "tfidf_logreg": set(self.test_ids[::4]), "tfidf_svm": set(self.test_ids[::5]),
            "bert_base": set(self.test_ids[::7]), "finbert_mlm": set(self.test_ids[::9]),
        }
        f1 = {"tfidf_logreg": 0.80, "tfidf_svm": 0.85}
        for model, bad in wrong.items():
            tags = ["seed42"] if model in run_step3_analysis.BASELINES else ["seed42", "seed100"]
            for tag in tags:
                _write_run(self.runs, CONFIG, model, tag,
                           {"val": self.test_ids[:5], "test": self.test_ids, "external": self.ext_ids},
                           {**labels, **ext_labels}, bad, f1.get(model, 0.5))
        # diagnostic subset: first 40 test records
        picked = self.test_ids[:40]
        items = [{"diag_id": f"d{k:03d}", "record_id": r, "label": labels[r]}
                 for k, r in enumerate(picked, start=1)]
        _dump({"config": CONFIG, "seed": 42, "n": 40, "source_split": "test", "items": items},
              os.path.join(self.diag, "diagnostic_key.json"))
        rule = {it["diag_id"]: _tags(**_fake_tagger([records[it["record_id"]]["text"]])[0])
                for it in items}
        ann_a = {d: dict(t) for d, t in rule.items()}
        ann_b = {d: dict(t) for d, t in rule.items()}
        for k, d in enumerate(sorted(ann_b)):
            if k % 5 == 0:
                ann_b[d]["negation"] = 1 - ann_b[d]["negation"]
        _write_ann(os.path.join(self.diag, "annotations_a.csv"), ann_a)
        _write_ann(os.path.join(self.diag, "annotations_b.csv"), ann_b)
        _write_ann(os.path.join(self.diag, "rule_tags.csv"), rule)
        _dump({"a": "model-alpha", "b": "model-beta"}, os.path.join(self.diag, "annotators.json"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *extra: str) -> int:
        return run_step3_analysis.main(
            ["--diagnostic-dir", self.diag, "--runs-dir", self.runs, "--out-dir", self.out,
             "--splits-dir", self.splits, "--bootstrap-samples", "200", *extra],
            tagger=_fake_tagger)

    def test_full_run_writes_outputs(self) -> None:
        self.assertEqual(self._run(), 0)
        for name in ("annotation_agreement.json", "phenomenon_analysis.json", "STEP3_REPORT.md",
                     "step3_phenomenon_error_heatmap.png", "step3_attribution_by_phenomenon.png",
                     "step3_rq1_accounting_terms.png"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        with open(os.path.join(self.out, "STEP3_REPORT.md"), encoding="utf-8") as handle:
            report = handle.read()
        first_section = report.split("## 2.")[0]
        self.assertIn("two independent LLM annotators, not human", first_section)
        self.assertIn("model-alpha", first_section)
        self.assertIn("model-beta", first_section)
        with open(os.path.join(self.out, "annotation_agreement.json"), encoding="utf-8") as handle:
            agreement = json.load(handle)
        self.assertAlmostEqual(agreement["agreement"]["hedging"]["kappa_ab"], 1.0)
        self.assertLess(agreement["agreement"]["negation"]["kappa_ab"], 1.0)
        with open(os.path.join(self.out, "phenomenon_analysis.json"), encoding="utf-8") as handle:
            analysis = json.load(handle)
        self.assertEqual(analysis["tfidf_best"], "tfidf_svm")
        self.assertIn(CONFIG, analysis["full_split_rule_tags"])
        full = analysis["full_split_rule_tags"][CONFIG]
        self.assertEqual(full["test"]["n"], 60)
        self.assertEqual(full["external"]["n"], 30)
        rq1 = full["test"]["rq1_financial_terminology"]["bert_base"]
        self.assertGreaterEqual(rq1["term1"]["n"] + rq1["term0"]["n"], 1)
        self.assertIn("delta_arch", analysis["diagnostic"]["attribution"]["negation"])
        self.assertEqual(len(run_step3_analysis.MODELS), 8)
        models = analysis["diagnostic"]["error_rates"]["negation"]["models"]
        self.assertEqual(list(models), list(run_step3_analysis.MODELS))
        self.assertIn("balanced_error", models["vader_official"]["tag1"])
        self.assertIn("gold_mix_tag1", analysis["diagnostic"]["error_rates"]["negation"])
        self.assertEqual(agreement["agreement"]["negation"]["n_consensus_vs_rule_diff"], 4)
        # SEntFiN block: cue breakdown and the loss-excluded table are present, diag has none
        ext = full["external"]
        self.assertIn("cost", ext["rq1_cue_breakdown"])
        self.assertIn("rq1_excluding_loss", ext)
        self.assertIn("rq1_cue_breakdown", full["test"])
        self.assertNotIn("rq1_cue_breakdown", analysis["diagnostic"])
        for p in PHENOMENA:
            a = analysis["diagnostic"]["attribution"][p]["delta_arch"]
            if analysis["diagnostic"]["attribution"][p]["small_n"]:
                self.assertIsNone(a["p_value"])
        with open(os.path.join(self.out, "STEP3_REPORT.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Both models may have seen Financial PhraseBank during pretraining", text)
        self.assertIn("label-balanced error", text)
        self.assertIn("gold-label mix", text)
        self.assertIn("excluding sentences whose only cue is loss/losses", text)
        self.assertIn("RQ1 by cue word", text)
        self.assertIn("consensus differs from rule", text)
        self.assertIn("Holm p", text)
        self.assertIn("(n=0*)", text)  # small-n star in headers

    def test_tmp_run_dirs_are_ignored(self) -> None:
        stale = os.path.join(self.runs, CONFIG, "bert_base", "seed7.tmp")
        os.makedirs(stale)  # no result.json: would raise "Missing run" if globbed
        self.assertEqual(self._run(), 0)

    def test_missing_official_baseline_exits_two(self) -> None:
        import shutil
        shutil.rmtree(os.path.join(self.runs, CONFIG, "lm_official"))
        self.assertEqual(self._run(), 2)

    def test_either_consensus_runs(self) -> None:
        self.assertEqual(self._run("--consensus", "either"), 0)
        with open(os.path.join(self.out, "phenomenon_analysis.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["consensus_mode"], "either")

    def test_missing_input_exits_two(self) -> None:
        os.remove(os.path.join(self.diag, "annotations_b.csv"))
        self.assertEqual(self._run(), 2)

    def test_missing_run_exits_two(self) -> None:
        import shutil
        shutil.rmtree(os.path.join(self.runs, CONFIG, "finbert_mlm"))
        self.assertEqual(self._run(), 2)


if __name__ == "__main__":
    unittest.main()
