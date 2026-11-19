"""Tests for scripts/summarize_step4.py on synthetic tuning + Step 2 run trees (no real runs)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts import summarize_step4 as s4
from scripts.run_step2 import finetune_tag
from src.evaluation.multiseed import mean_sd, seed_sentence_bootstrap

N_TEST = 40
N_EXTERNAL = 45
B = 200
VERSIONS = {"python": "3.9.6", "numpy": "1.26.2", "sklearn": "1.6.1", "torch": "2.8.0", "transformers": "4.57.6"}
TUNED = {"bert_base": (3e-5, 32), "finbert_mlm": (2e-5, 32)}
FIXED = (2e-5, 16)
# share of correct rows per recipe/model; seeds shift it a little
ACCURACY = {("fixed", "bert_base"): 0.70, ("fixed", "finbert_mlm"): 0.75,
            ("tuned", "bert_base"): 0.85, ("tuned", "finbert_mlm"): 0.80}


def _labels(n: int) -> np.ndarray:
    return (np.arange(n) % 3).astype(np.int64)


def _predictions(y: np.ndarray, accuracy: float, rng: np.random.Generator) -> np.ndarray:
    pred = y.copy()
    wrong = rng.random(len(y)) > accuracy
    pred[wrong] = (y[wrong] + rng.integers(1, 3, size=int(wrong.sum()))) % 3
    return pred


def _prob(pred: np.ndarray) -> np.ndarray:
    prob = np.full((len(pred), 3), 0.1, dtype=np.float32)
    prob[np.arange(len(pred)), pred] = 0.8
    return prob


def _write_run(run_dir, model, seed, recipe, preds, ids, labels, versions=VERSIONS):
    """result.json + predictions.npz as scripts/run_step2.py writes them. ``preds``: split -> vector."""
    os.makedirs(run_dir)
    arrays = {}
    for split, pred in preds.items():
        arrays[f"{split}_ids"] = np.asarray(ids[split])
        arrays[f"{split}_y"] = np.asarray(labels[split], dtype=np.int64)
        arrays[f"{split}_prob"] = _prob(pred)
    result = {
        "status": "completed", "stage": "finetune", "config": s4.CONFIG, "model_key": model, "seed": seed,
        "hyperparams": {"learning_rate": recipe[0], "batch_size": recipe[1]},
        "metrics": {split: {"macro_f1": s4.macro_f1(labels[split], pred)} for split, pred in preds.items()},
        "device": "mps", "versions": versions,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    np.savez_compressed(os.path.join(run_dir, "predictions.npz"), **arrays)


class Tree:
    """A synthetic ``results/tuning`` + ``results/step2/runs`` pair in a temp dir."""

    def __init__(self, root: str, identical: bool = False):
        self.root = root
        self.tuning = os.path.join(root, "tuning")
        self.step2 = os.path.join(root, "step2", "runs")
        self.out = os.path.join(root, "step4")
        self.ids = {"test": np.array([f"pb_{i:03d}" for i in range(N_TEST)]),
                    "external": np.array([f"sf_{i:03d}" for i in range(N_EXTERNAL)])}
        self.labels = {"test": _labels(N_TEST), "external": _labels(N_EXTERNAL)}
        self.preds: dict = {}
        rng = np.random.default_rng(7)
        shared = {split: _predictions(y, 0.8, rng) for split, y in self.labels.items()}
        for recipe in s4.RECIPES:
            for model in s4.MODELS:
                for index, seed in enumerate(s4.SEEDS):
                    if identical:
                        preds = {split: vec.copy() for split, vec in shared.items()}
                    else:
                        acc = ACCURACY[(recipe, model)] + 0.03 * index
                        preds = {split: _predictions(y, acc, rng) for split, y in self.labels.items()}
                    self.preds[(recipe, model, seed)] = preds
                    _write_run(self.run_dir(recipe, model, seed), model, seed,
                               TUNED[model] if recipe == "tuned" else FIXED, preds, self.ids, self.labels)
        os.makedirs(self.tuning, exist_ok=True)
        selection = {
            "phrasebank_config": s4.CONFIG, "selection_seed": 42, "selection_metric": "val_macro_f1",
            "selected": {m: {"learning_rate": lr, "batch_size": bs, "val_macro_f1": 0.95}
                         for m, (lr, bs) in TUNED.items()},
            "candidates": [],
        }
        with open(os.path.join(self.tuning, "selection.json"), "w", encoding="utf-8") as handle:
            json.dump(selection, handle)

    def run_dir(self, recipe: str, model: str, seed: int) -> str:
        if recipe == "fixed":
            return os.path.join(self.step2, s4.CONFIG, model, f"seed{seed}")
        return os.path.join(self.tuning, "final", s4.CONFIG, model, finetune_tag(seed, *TUNED[model]))

    def argv(self, *extra: str) -> list:
        return ["--tuning-dir", self.tuning, "--step2-runs", self.step2, "--out-dir", self.out,
                "--B", str(B), "--seed", "5", *extra]

    def main(self, *extra: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = s4.main(self.argv(*extra))
        return code, out.getvalue(), err.getvalue()

    def stats(self) -> dict:
        with open(os.path.join(self.out, s4.JSON_NAME), encoding="utf-8") as handle:
            return json.load(handle)

    def rewrite_npz(self, recipe: str, model: str, seed: int, **changes) -> None:
        path = os.path.join(self.run_dir(recipe, model, seed), "predictions.npz")
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays.update(changes)
        np.savez_compressed(path, **arrays)


class TempTreeCase(unittest.TestCase):
    identical = False

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="step4_stats_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tree = Tree(self.tmp, identical=self.identical)


class SummaryTests(TempTreeCase):
    def setUp(self):
        super().setUp()
        code, self.stdout, self.stderr = self.tree.main()
        self.assertEqual(code, 0, self.stderr)
        self.stats = self.tree.stats()

    def test_recipe_read_from_selection_json(self):
        recipes = self.stats["recipes"]
        self.assertEqual(recipes["tuned"]["bert_base"],
                         {"learning_rate": 3e-5, "batch_size": 32, "val_macro_f1": 0.95})
        self.assertEqual(recipes["tuned"]["finbert_mlm"]["batch_size"], 32)
        self.assertEqual(recipes["fixed"]["bert_base"], {"learning_rate": 2e-5, "batch_size": 16})
        tuned_dir = self.stats["run_dirs"]["tuned"]["bert_base"]["100"]
        self.assertTrue(tuned_dir.endswith("final/sentences_allagree/bert_base/lr3e-05_bs32_seed100"), tuned_dir)
        self.assertTrue(self.stats["run_dirs"]["fixed"]["finbert_mlm"]["2026"].endswith(
            "sentences_allagree/finbert_mlm/seed2026"))
        self.assertEqual(self.stats["protocol"]["B"], B)
        self.assertEqual(self.stats["protocol"]["seed"], 5)
        self.assertEqual(self.stats["n"], {"test": N_TEST, "external": N_EXTERNAL})

    def test_means_and_sds(self):
        from sklearn.metrics import accuracy_score, f1_score

        for recipe in s4.RECIPES:
            for model in s4.MODELS:
                for split in s4.SPLITS:
                    y = self.tree.labels[split]
                    f1s = [f1_score(y, self.tree.preds[(recipe, model, s)][split], labels=[0, 1, 2],
                                    average="macro", zero_division=0) for s in s4.SEEDS]
                    accs = [accuracy_score(y, self.tree.preds[(recipe, model, s)][split]) for s in s4.SEEDS]
                    cell = self.stats["scores"][model][recipe][split]
                    for seed, expected in zip(s4.SEEDS, f1s):
                        self.assertAlmostEqual(cell["macro_f1_by_seed"][str(seed)], expected, places=12)
                    self.assertAlmostEqual(cell["macro_f1"]["mean"], float(np.mean(f1s)), places=12)
                    self.assertAlmostEqual(cell["macro_f1"]["sd"], float(np.std(f1s, ddof=1)), places=12)
                    self.assertEqual(cell["macro_f1"]["n"], mean_sd(f1s)["n"])
                    self.assertAlmostEqual(cell["accuracy"]["mean"], float(np.mean(accs)), places=12)
        self.assertLess(self.stats["max_abs_diff_vs_result_json_macro_f1"], 1e-12)

    def test_comparisons_match_library_and_seed_means(self):
        for split in s4.SPLITS:
            y = self.tree.labels[split]
            self.assertEqual(set(self.stats["comparisons"][split]), {name for name, _a, _b in s4.COMPARISONS})
            for name, (ra, ma), (rb, mb) in s4.COMPARISONS:
                entry = self.stats["comparisons"][split][name]
                mean_a = self.stats["scores"][ma][ra][split]["macro_f1"]["mean"]
                mean_b = self.stats["scores"][mb][rb][split]["macro_f1"]["mean"]
                self.assertAlmostEqual(entry["delta_macro_f1"], mean_a - mean_b, places=12)
                for seed in s4.SEEDS:
                    by_seed_a = self.stats["scores"][ma][ra][split]["macro_f1_by_seed"][str(seed)]
                    by_seed_b = self.stats["scores"][mb][rb][split]["macro_f1_by_seed"][str(seed)]
                    self.assertAlmostEqual(entry["per_seed_delta"][str(seed)], by_seed_a - by_seed_b, places=12)
                boot = seed_sentence_bootstrap(
                    y, [self.tree.preds[(ra, ma, s)][split] for s in s4.SEEDS],
                    [self.tree.preds[(rb, mb, s)][split] for s in s4.SEEDS], metric="macro_f1", B=B, seed=5)
                self.assertEqual(entry["ci_low"], boot["ci_low"])
                self.assertEqual(entry["ci_high"], boot["ci_high"])
                self.assertEqual(entry["p_bootstrap"], boot["p_value"])
                self.assertLessEqual(entry["ci_low"], entry["ci_high"])
                self.assertEqual(entry["interval_contains_zero"], entry["ci_low"] <= 0.0 <= entry["ci_high"])

    def test_mcnemar_holm_over_eight_tests_per_seed(self):
        from src.evaluation.multiseed import holm_bonferroni
        from src.evaluation.stats import mcnemar_test

        for seed in s4.SEEDS:
            raw = {}
            for split in s4.SPLITS:
                for name, (ra, ma), (rb, mb) in s4.COMPARISONS:
                    raw[(split, name)] = mcnemar_test(
                        self.tree.labels[split], self.tree.preds[(ra, ma, seed)][split],
                        self.tree.preds[(rb, mb, seed)][split])["p_value"]
            self.assertEqual(len(raw), 8)
            adjusted = holm_bonferroni({f"{k[0]}/{k[1]}": v for k, v in raw.items()})
            for (split, name), p in raw.items():
                entry = self.stats["comparisons"][split][name]
                self.assertAlmostEqual(entry["mcnemar_by_seed"][str(seed)]["p_value"], p, places=12)
                self.assertAlmostEqual(entry["p_holm_by_seed"][str(seed)], adjusted[f"{split}/{name}"], places=12)
        for split in s4.SPLITS:
            for entry in self.stats["comparisons"][split].values():
                raw_p = [v["p_value"] for v in entry["mcnemar_by_seed"].values()]
                self.assertEqual(entry["p_min"], min(raw_p))
                self.assertEqual(entry["p_max"], max(raw_p))
                self.assertEqual(entry["n_seeds_significant"],
                                 sum(1 for v in entry["p_holm_by_seed"].values() if v < 0.05))

    def _sets(self, split):
        return [[self.tree.preds[(recipe, model, s)][split] for s in s4.SEEDS]
                for recipe in ("tuned", "fixed") for model in s4.MODELS]

    def test_gap_change_comes_from_one_four_set_bootstrap(self):
        for split in s4.SPLITS:
            comps = self.stats["comparisons"][split]
            gap = self.stats["gap_change"][split]
            self.assertAlmostEqual(
                gap["delta_macro_f1"],
                comps["tuned_bert_vs_tuned_finbert"]["delta_macro_f1"]
                - comps["fixed_bert_vs_fixed_finbert"]["delta_macro_f1"], places=12)
            draws = s4.gap_change_draws(self.tree.labels[split], self._sets(split), B, 5)
            self.assertEqual(draws.shape, (B,))
            low, high = np.percentile(draws, [2.5, 97.5])
            self.assertAlmostEqual(gap["ci_low"], float(low), places=12)
            self.assertAlmostEqual(gap["ci_high"], float(high), places=12)
            tail = min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
            self.assertAlmostEqual(gap["p_bootstrap"], min(1.0, 2.0 * tail), places=12)
            self.assertEqual(gap["interval_contains_zero"], gap["ci_low"] <= 0.0 <= gap["ci_high"])

    def test_gap_draws_are_deterministic_and_use_their_own_stream(self):
        y, sets = self.tree.labels["test"], self._sets("test")
        np.testing.assert_array_equal(s4.gap_change_draws(y, sets, B, 5), s4.gap_change_draws(y, sets, B, 5))
        self.assertFalse(np.array_equal(s4.gap_change_draws(y, sets, B, 5), s4.gap_change_draws(y, sets, B, 6)))

    def test_report_and_environment(self):
        self.assertTrue(self.stats["environment"]["identical_across_all_runs"])
        self.assertEqual(self.stats["environment"]["fixed"], [{"device": "mps", "versions": VERSIONS}])
        with open(os.path.join(self.tree.out, s4.REPORT_NAME), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("## Protocol", report)
        self.assertIn("torch 2.8.0", report)
        self.assertIn("device `mps`", report)
        self.assertIn("learning rate 3e-05, batch size 32", report)
        self.assertIn("inside one software environment", report)
        self.assertIn("seeds of the four run sets are resampled independently", report)
        self.assertIn("they are not paired", report)
        self.assertIn("default_rng([seed, 2])", self.stats["protocol"]["bootstrap"])
        self.assertEqual(report.count("- On PhraseBank test,"), 5)   # 4 comparisons + gap change
        self.assertEqual(report.count("- On SEntFiN (external),"), 5)
        self.assertNotIn("no effect", report.lower())
        self.assertTrue("contains zero" in report or "does not contain zero" in report)

    def test_check_passes_after_write_and_writes_nothing(self):
        json_path = os.path.join(self.tree.out, s4.JSON_NAME)
        before = (os.path.getmtime(json_path), sorted(os.listdir(self.tree.out)))
        code, out, _err = self.tree.main("--check")
        self.assertEqual(code, 0)
        self.assertIn("check ok", out)
        self.assertEqual(before, (os.path.getmtime(json_path), sorted(os.listdir(self.tree.out))))

    def test_check_fails_on_a_changed_number(self):
        json_path = os.path.join(self.tree.out, s4.JSON_NAME)
        stats = self.tree.stats()
        stats["comparisons"]["test"]["tuned_vs_fixed_bert"]["delta_macro_f1"] += 1e-6
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle)
        code, _out, err = self.tree.main("--check")
        self.assertEqual(code, 1)
        self.assertIn("differs", err)

    def test_environment_mismatch_is_reported(self):
        run_dir = self.tree.run_dir("tuned", "bert_base", 42)
        path = os.path.join(run_dir, "result.json")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["versions"] = dict(VERSIONS, torch="2.5.1+cu121")
        result["device"] = "cuda"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        code, _out, err = self.tree.main()
        self.assertEqual(code, 0, err)
        self.assertFalse(self.tree.stats()["environment"]["identical_across_all_runs"])
        with open(os.path.join(self.tree.out, s4.REPORT_NAME), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("do NOT all record the same", report)
        self.assertIn("2.5.1+cu121", report)


class IdenticalModelsTests(TempTreeCase):
    identical = True

    def test_delta_zero_and_mcnemar_one(self):
        code, _out, err = self.tree.main()
        self.assertEqual(code, 0, err)
        stats = self.tree.stats()
        for split in s4.SPLITS:
            for entry in stats["comparisons"][split].values():
                self.assertEqual(entry["delta_macro_f1"], 0.0)
                self.assertEqual(set(entry["per_seed_delta"].values()), {0.0})
                self.assertEqual((entry["ci_low"], entry["ci_high"]), (0.0, 0.0))
                self.assertTrue(entry["interval_contains_zero"])
                self.assertEqual(entry["p_bootstrap"], 1.0)
                self.assertEqual((entry["p_min"], entry["p_max"]), (1.0, 1.0))
                self.assertEqual(entry["n_seeds_significant"], 0)
                for res in entry["mcnemar_by_seed"].values():
                    self.assertEqual((res["n10"], res["n01"], res["p_value"]), (0, 0, 1.0))
            self.assertEqual(stats["gap_change"][split]["delta_macro_f1"], 0.0)
            self.assertTrue(stats["gap_change"][split]["interval_contains_zero"])


class GapSeedIndependenceTests(unittest.TestCase):
    """Tuned and fixed runs with the same seed number are different trainings: no seed pairing."""

    def setUp(self):
        n = 60
        self.y = _labels(n)
        strong = []
        for wrong_rows in (0, 15, 30):          # seeds differ strongly: 100%, 75%, 50% correct
            pred = self.y.copy()
            pred[:wrong_rows] = (pred[:wrong_rows] + 1) % 3
            strong.append(pred)
        steady = [self.y.copy() for _ in range(3)]
        # tuned BERT == fixed BERT run by run, FinBERT sets constant: the observed gap change is 0
        self.sets = [strong, steady, [p.copy() for p in strong], [p.copy() for p in steady]]

    def test_four_seed_draws_are_independent(self):
        big_b = 4000
        independent = s4.gap_change_draws(self.y, self.sets, big_b, 11)
        paired = s4.gap_change_draws(self.y, self.sets, big_b, 11, _share_recipe_seeds=True)
        # pairing tuned seed k with fixed seed k cancels the seed spread completely here
        self.assertEqual(float(np.ptp(paired)), 0.0)
        width = float(np.diff(np.percentile(independent, [2.5, 97.5]))[0])
        paired_width = float(np.diff(np.percentile(paired, [2.5, 97.5]))[0])
        self.assertGreaterEqual(width, paired_width)
        self.assertGreater(width, 0.2)
        self.assertLess(abs(float(independent.mean())), 0.02)
        low, high = np.percentile(independent, [2.5, 97.5])
        self.assertTrue(low < 0.0 < high)

    def test_tuned_and_fixed_picks_are_uncorrelated(self):
        # one sentence resample cannot explain the spread: with all-identical sets the draws are exactly 0
        same = [self.sets[1]] * 4
        self.assertEqual(float(np.ptp(s4.gap_change_draws(self.y, same, 500, 11))), 0.0)
        # tuned-BERT and fixed-BERT seed means move independently, so their difference has spread
        only_bert = s4.gap_change_draws(self.y, self.sets, 2000, 3)
        self.assertGreater(float(np.std(only_bert)), 0.05)

    def test_rejects_wrong_number_of_sets(self):
        with self.assertRaises(ValueError):
            s4.gap_change_draws(self.y, self.sets[:3], 10, 1)


class RefusalTests(TempTreeCase):
    def _refused(self, needle: str) -> None:
        code, _out, err = self.tree.main()
        self.assertEqual(code, 2)
        self.assertIn("refused", err)
        self.assertIn(needle, err)
        self.assertFalse(os.path.exists(self.tree.out), "a refused run must write nothing")

    def test_missing_tuned_seed(self):
        shutil.rmtree(self.tree.run_dir("tuned", "finbert_mlm", 100))
        self._refused("finbert_mlm seed 100")

    def test_missing_fixed_seed(self):
        shutil.rmtree(self.tree.run_dir("fixed", "bert_base", 2026))
        self._refused("bert_base seed 2026")

    def test_uncompleted_run_counts_as_missing(self):
        path = os.path.join(self.tree.run_dir("tuned", "bert_base", 42), "result.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"status": "running"}, handle)
        self._refused("not completed")

    def test_missing_selection_json(self):
        os.remove(os.path.join(self.tree.tuning, "selection.json"))
        self._refused("selection.json")

    def test_id_mismatch(self):
        ids = self.tree.ids["test"].copy()
        ids[3] = "pb_other"
        self.tree.rewrite_npz("tuned", "bert_base", 100, test_ids=ids)
        self._refused("test ids differ")

    def test_id_order_mismatch(self):
        self.tree.rewrite_npz("fixed", "finbert_mlm", 42, external_ids=self.tree.ids["external"][::-1].copy())
        self._refused("external ids differ")

    def test_label_mismatch(self):
        y = self.tree.labels["external"].copy()
        y[0] = (y[0] + 1) % 3
        self.tree.rewrite_npz("tuned", "finbert_mlm", 2026, external_y=y)
        self._refused("external labels (y) differ")

    def test_run_of_another_recipe_is_refused(self):
        path = os.path.join(self.tree.run_dir("tuned", "bert_base", 42), "result.json")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["hyperparams"]["batch_size"] = 16
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        self._refused("selection.json selected")

    def test_check_also_refuses(self):
        shutil.rmtree(self.tree.run_dir("tuned", "bert_base", 42))
        code, _out, err = self.tree.main("--check")
        self.assertEqual(code, 2)
        self.assertIn("refused", err)


if __name__ == "__main__":
    unittest.main()
