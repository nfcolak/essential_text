"""Offline tests for the learning-rate / batch-size study.

No pretrained checkpoints are loaded. Transformer runs are stubbed, and the
selection tests use synthetic result.json files.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts import run_step2, run_tuning
from src.data.download import generate_synthetic_dataset
from src.data.io import file_sha256, load_manifest, split_arrays
from src.data.splits import build_records, create_stratified_splits
from src.transformer.finetune import FinetuneConfig

NAME_TO_KEY = {name: key for key, name in run_step2.FINETUNE_KEYS.items()}

# (validation correct-rate, test/external correct-rate). The validation winner is
# not the test winner: selection must follow the first number.
RATES = {
    ("bert_base", "2e-05", 16): (0.0, 1.0),
    ("bert_base", "2e-05", 32): (0.0, 0.0),
    ("bert_base", "3e-05", 16): (0.0, 0.0),
    ("bert_base", "3e-05", 32): (1.0, 0.0),
    ("finbert_mlm", "2e-05", 16): (0.0, 1.0),
    ("finbert_mlm", "2e-05", 32): (1.0, 0.0),
    ("finbert_mlm", "3e-05", 16): (0.0, 0.0),
    ("finbert_mlm", "3e-05", 32): (0.0, 0.0),
}
WINNERS = {"bert_base": (3e-5, 32), "finbert_mlm": (2e-5, 32)}


def probs_for_rate(labels, correct_rate: float) -> np.ndarray:
    y = np.asarray(list(labels), dtype=int)
    n = len(y)
    n_correct = int(np.floor(float(correct_rate) * n + 1e-9))
    prob = np.full((n, 3), 0.05, dtype=np.float64)
    for i, gold in enumerate(y.tolist()):
        pred = gold if i < n_correct else (gold + 1) % 3
        prob[i, pred] = 0.90
    prob /= prob.sum(axis=1, keepdims=True)
    return prob.astype(np.float32)


class StubCalls:
    def __init__(self):
        self.finetune = 0
        self.seen_eval_sets: list[list[str]] = []

    def finetune_fn(self, model_name, train_texts, train_labels, val_texts, val_labels, eval_sets, config,
                    revision=None, model_loader=None):
        self.finetune += 1
        self.seen_eval_sets.append(sorted(eval_sets))
        return {
            "best_epoch": 2,
            "history": [{"epoch": 1, "train_loss": 1.0, "val_macro_f1": 0.3, "seconds": 0.1}],
            "val_prob": probs_for_rate(val_labels, 1.0),
            "eval_probs": {key: probs_for_rate([0, 1, 2] * ((len(texts) // 3) + 1), 0.0)[:len(texts)]
                           for key, texts in eval_sets.items()},
            "wall_seconds": 0.1,
            "device": "cpu",
            "config": config.to_dict(),
        }

    def zero_shot_fn(self, model_name, texts, config, revision=None, model_loader=None):
        return probs_for_rate([0, 1, 2], 1.0)[:0]


class RankingStub(StubCalls):
    """Validation accuracy follows ``RATES``. Holdout texts are scored only when the runner asks."""

    def __init__(self, pb_map, sf_map):
        super().__init__()
        self.pb_map = pb_map
        self.sf_map = sf_map

    def finetune_fn(self, model_name, train_texts, train_labels, val_texts, val_labels, eval_sets, config,
                    revision=None, model_loader=None):
        res = super().finetune_fn(
            model_name, train_texts, train_labels, val_texts, val_labels, eval_sets, config,
            revision=revision, model_loader=model_loader,
        )
        key = (
            NAME_TO_KEY[model_name],
            run_step2.format_learning_rate(config.learning_rate),
            int(config.batch_size),
        )
        val_rate, heldout_rate = RATES[key]
        res["val_prob"] = probs_for_rate(val_labels, val_rate)
        if not eval_sets:
            res["eval_probs"] = {}
            return res
        res["eval_probs"] = {
            "test": probs_for_rate([self.pb_map[text] for text in eval_sets["test"]], heldout_rate),
            "external": probs_for_rate([self.sf_map[text] for text in eval_sets["external"]], heldout_rate),
        }
        return res


def run_step2_main(argv, stub=None):
    stub = stub or StubCalls()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_step2.main(
            argv, finetune_fn=stub.finetune_fn, zero_shot_fn=stub.zero_shot_fn,
            baseline_factory=lambda *args, **kwargs: None,
        )
    return code, out.getvalue(), err.getvalue(), stub


def run_tuning_main(argv, runner=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_tuning.main(argv, runner=runner)
    return code, out.getvalue(), err.getvalue()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flag_values(argv, flag):
    values = []
    tokens = list(argv)
    index = tokens.index(flag) + 1
    while index < len(tokens) and not str(tokens[index]).startswith("--"):
        values.append(tokens[index])
        index += 1
    return values


def metric_block(score: float) -> dict:
    per_class = {
        name: {"precision": score, "recall": score, "f1": score, "support": 10}
        for name in ("negative", "neutral", "positive")
    }
    return {
        "accuracy": score,
        "macro_f1": score,
        "per_class": per_class,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "neutral_to_negative_rate": 0.0,
        "ece": 0.0,
    }


def write_result(runs_dir, model, learning_rate, batch_size, seed, val_f1, test_f1=None, external_f1=0.4,
                 best_epoch=2, holdout=True):
    run_dir = run_tuning.run_directory(runs_dir, model, learning_rate, batch_size, seed)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "status": "completed",
        "stage": "finetune",
        "config": "sentences_allagree",
        "model_key": model,
        "seed": seed,
        "hyperparams": {"learning_rate": learning_rate, "batch_size": batch_size},
        "best_epoch": best_epoch,
        "metrics": {
            "val": metric_block(val_f1),
            "test": metric_block(test_f1) if holdout else None,
            "external": metric_block(external_f1) if holdout else None,
        },
        "n": {
            "train": 8,
            "val": 4,
            "test": 4 if holdout else None,
            "external": 4 if holdout else None,
        },
        "external_sha256": "ab" * 32 if holdout else None,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    if model == "finbert_mlm":
        from src.transformer.finetune import TOKENIZER_CHECK_NAME, injected_finbert_tokenizer_check

        with open(os.path.join(run_dir, TOKENIZER_CHECK_NAME), "w", encoding="utf-8") as handle:
            json.dump(injected_finbert_tokenizer_check(), handle)
    return run_dir


# Validation score, then the test score used only by the final-stage files.
FAKE_GRID = {
    ("bert_base", 2e-5, 16): (0.10, 0.99),
    ("bert_base", 2e-5, 32): (0.40, 0.50),
    ("bert_base", 3e-5, 16): (0.70, 0.40),
    ("bert_base", 3e-5, 32): (0.95, 0.05),
    ("finbert_mlm", 2e-5, 16): (0.20, 0.97),
    ("finbert_mlm", 2e-5, 32): (0.93, 0.07),
    ("finbert_mlm", 3e-5, 16): (0.55, 0.60),
    ("finbert_mlm", 3e-5, 32): (0.30, 0.90),
}


def write_fake_grid(out_dir, include_final_seeds=False):
    selection_dir = os.path.join(out_dir, run_tuning.SELECTION_DIRNAME)
    for (model, learning_rate, batch_size), (val_f1, test_f1) in FAKE_GRID.items():
        write_result(
            selection_dir, model, learning_rate, batch_size, 42, val_f1, test_f1, holdout=False,
        )
    if include_final_seeds:
        final_dir = os.path.join(out_dir, run_tuning.FINAL_DIRNAME)
        write_result(final_dir, "bert_base", 3e-5, 32, 42, 0.95, 0.05, external_f1=0.40)
        write_result(final_dir, "bert_base", 3e-5, 32, 100, 0.91, 0.11, external_f1=0.41)
        write_result(final_dir, "bert_base", 3e-5, 32, 2026, 0.92, 0.13, external_f1=0.42)
        write_result(final_dir, "finbert_mlm", 2e-5, 32, 42, 0.93, 0.07, external_f1=0.40)
        write_result(final_dir, "finbert_mlm", 2e-5, 32, 100, 0.88, 0.08, external_f1=0.43)
        write_result(final_dir, "finbert_mlm", 2e-5, 32, 2026, 0.89, 0.09, external_f1=0.44)
    return selection_dir


def label_map(pairs):
    mapping = {}
    for text, label in pairs:
        label = int(label)
        if text in mapping and mapping[text] != label:
            raise AssertionError(f"text maps to two labels: {text!r}")
        mapping[text] = label
    return mapping


class TagAndPlanTests(unittest.TestCase):
    def test_search_space_matches_protocol(self):
        self.assertEqual(run_tuning.TUNING_MODELS, ("bert_base", "finbert_mlm"))
        self.assertEqual(run_tuning.LEARNING_RATES, (2e-5, 3e-5))
        self.assertEqual(run_tuning.BATCH_SIZES, (16, 32))
        self.assertEqual(len(run_tuning.iter_grid()), 8)
        self.assertEqual(run_tuning.SELECTION_SEED, 42)
        self.assertEqual(run_tuning.FINAL_SEEDS, (42, 100, 2026))
        self.assertEqual(run_tuning.PHRASEBANK_CONFIG, "sentences_allagree")
        self.assertEqual(run_tuning.SELECTION_METRIC, "val_macro_f1")
        self.assertEqual(run_tuning.DEFAULT_OUT_DIR, "results/tuning")
        self.assertTrue(all("test" not in name and "external" not in name for name in run_tuning.TUNING_FIELDS))

    def test_defaults_match_finetune_config_and_historical_tags(self):
        config = FinetuneConfig()
        self.assertEqual(run_step2.DEFAULT_LEARNING_RATE, config.learning_rate)
        self.assertEqual(run_step2.DEFAULT_BATCH_SIZE, config.batch_size)
        self.assertEqual(run_step2.format_learning_rate(2e-5), "2e-05")
        self.assertEqual(run_step2.format_learning_rate(3e-5), "3e-05")
        self.assertNotEqual(run_step2.format_learning_rate(2.1e-5), "2e-05")
        self.assertTrue(run_step2.is_default_recipe(float("2e-05"), 16))
        self.assertFalse(run_step2.is_default_recipe(float("3e-05"), 16))
        self.assertFalse(run_step2.is_default_recipe(2e-5, 32))
        self.assertEqual(run_step2.finetune_tag(42), "seed42")
        self.assertEqual(run_step2.finetune_tag(100, 2e-5, 16), "seed100")
        tags = [
            run_step2.finetune_tag(seed, lr, bs)
            for seed in (42, 100, 2026)
            for lr in (2e-5, 3e-5)
            for bs in (16, 32)
        ]
        self.assertEqual(len(tags), len(set(tags)))
        self.assertIn("lr3e-05_bs32_seed2026", tags)
        self.assertNotIn("lr2e-05_bs16_seed42", tags)

    def test_cv_tags_change_only_for_tuned_transformers(self):
        self.assertEqual(run_step2.cv_tag(0, "bert_base"), "cv_fold0")
        self.assertEqual(run_step2.cv_tag(0, "vader", 3e-5, 32), "cv_fold0")
        self.assertEqual(run_step2.cv_tag(1, "finbert_mlm", 3e-5, 16), "lr3e-05_bs16_cv_fold1")

    def test_build_plan_keeps_default_paths_and_isolates_other_recipes(self):
        runs = run_step2.build_plan(
            "finetune", ["sentences_allagree"], ["bert_base", "finbert_mlm"], [42, 100, 2026],
        )
        self.assertEqual(
            [run.tag for run in runs],
            ["seed42", "seed100", "seed2026", "seed42", "seed100", "seed2026"],
        )
        custom = run_step2.build_plan(
            "all", ["sentences_allagree"], None, [42], learning_rate=3e-5, batch_size=32,
        )
        tags = {(run.stage, run.model_key, run.fold): run.tag for run in custom}
        self.assertEqual(tags[("baselines", "vader", None)], "seed42")
        self.assertEqual(tags[("finetune", "bert_base", None)], "lr3e-05_bs32_seed42")
        self.assertEqual(tags[("finetune", "finbert_mlm", None)], "lr3e-05_bs32_seed42")
        self.assertEqual(tags[("cv", "bert_base", 0)], "lr3e-05_bs32_cv_fold0")
        self.assertEqual(tags[("cv", "tfidf_svm", 0)], "cv_fold0")
        self.assertEqual(tags[("zeroshot", "prosus_finbert", None)], "zeroshot")
        held_in = run_step2.build_plan(
            "finetune", ["sentences_allagree"], ["bert_base"], [42], evaluate_holdout=False,
        )
        self.assertFalse(held_in[0].evaluate_holdout)
        self.assertTrue(run_step2.build_plan(
            "finetune", ["sentences_allagree"], ["bert_base"], [42],
        )[0].evaluate_holdout)

    def test_val_only_refuses_the_default_step2_directory(self):
        code, _stdout, err, _stub = run_step2_main([
            "--val-only", "--stage", "finetune", "--configs", "sentences_allagree",
            "--models", "bert_base", "--seeds", "42",
        ])
        self.assertEqual(code, 2, err)
        self.assertIn("val-only", err)

    def test_invalid_learning_rate_and_batch_size_exit_before_touching_files(self):
        out = os.path.join(tempfile.mkdtemp(prefix="tuning_invalid_"), "out")
        self.addCleanup(shutil.rmtree, os.path.dirname(out), True)
        for argv, needle in (
            (["--learning-rate", "-1"], "learning-rate"),
            (["--learning-rate", "nan"], "learning-rate"),
            (["--batch-size", "0"], "batch-size"),
        ):
            code, _stdout, err, _stub = run_step2_main(
                ["--splits-dir", "does-not-exist", "--out-dir", out, *argv]
            )
            self.assertEqual(code, 2, err)
            self.assertIn(needle, err)
            self.assertNotIn("missing input file", err)
            self.assertFalse(os.path.exists(out))


class SelectionTests(unittest.TestCase):
    def test_select_best_uses_validation_f1_not_test_or_accuracy(self):
        rows = [
            {"model_key": "bert_base", "learning_rate": 2e-5, "batch_size": 16,
             "val_macro_f1": 0.40, "val_accuracy": 0.99, "test_macro_f1": 0.99},
            {"model_key": "bert_base", "learning_rate": 3e-5, "batch_size": 32,
             "val_macro_f1": 0.90, "val_accuracy": 0.10, "test_macro_f1": 0.01},
            {"model_key": "finbert_mlm", "learning_rate": 3e-5, "batch_size": 16,
             "val_macro_f1": 0.50, "val_accuracy": 0.99, "test_macro_f1": 0.98},
            {"model_key": "finbert_mlm", "learning_rate": 2e-5, "batch_size": 32,
             "val_macro_f1": 0.85, "val_accuracy": 0.20, "test_macro_f1": 0.02},
        ]
        chosen = run_tuning.select_best(rows)
        self.assertEqual(chosen["bert_base"]["batch_size"], 32)
        self.assertAlmostEqual(chosen["bert_base"]["learning_rate"], 3e-5)
        self.assertEqual(chosen["finbert_mlm"]["batch_size"], 32)
        self.assertAlmostEqual(chosen["finbert_mlm"]["learning_rate"], 2e-5)

    def test_ties_break_toward_lower_learning_rate_then_smaller_batch(self):
        rows = [
            {"model_key": "bert_base", "learning_rate": 3e-5, "batch_size": 16,
             "val_macro_f1": 0.8, "test_macro_f1": 0.99},
            {"model_key": "bert_base", "learning_rate": 2e-5, "batch_size": 32,
             "val_macro_f1": 0.8, "test_macro_f1": 0.10},
            {"model_key": "bert_base", "learning_rate": 2e-5, "batch_size": 16,
             "val_macro_f1": 0.8, "test_macro_f1": 0.50},
        ]
        chosen = run_tuning.select_best(rows)
        self.assertAlmostEqual(chosen["bert_base"]["learning_rate"], 2e-5)
        self.assertEqual(chosen["bert_base"]["batch_size"], 16)


class TuningTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tuning_tables_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_write_tables_ignores_higher_test_score(self):
        out = os.path.join(self.tmp, "tables")
        write_fake_grid(out, include_final_seeds=True)

        def runner(_argv):
            raise AssertionError("rebuilding tables must not train")

        code, _stdout, err = run_tuning_main(
            ["--out-dir", out, "--phase", "all", "--write-tables"], runner=runner,
        )
        self.assertEqual(code, 0, err)
        tuning = read_csv(os.path.join(out, "tuning_results.csv"))
        self.assertEqual(list(tuning[0]), list(run_tuning.TUNING_FIELDS))
        self.assertEqual(len(tuning), 8)
        selected = {}
        for row in tuning:
            self.assertNotIn("test", row["run_dir"])
            if row["selected"] == "1":
                selected[row["model_key"]] = row
        self.assertEqual(set(selected), {"bert_base", "finbert_mlm"})
        self.assertEqual(selected["bert_base"]["learning_rate"], "3e-05")
        self.assertEqual(selected["bert_base"]["batch_size"], "32")
        self.assertEqual(selected["bert_base"]["val_macro_f1"], "0.950000")
        self.assertEqual(selected["finbert_mlm"]["learning_rate"], "2e-05")
        self.assertEqual(selected["finbert_mlm"]["batch_size"], "32")
        decoy = [row for row in tuning if row["model_key"] == "bert_base" and row["batch_size"] == "16"
                 and row["learning_rate"] == "2e-05"]
        self.assertEqual(decoy[0]["selected"], "0")

        with open(os.path.join(out, "selection.json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["selection_metric"], "val_macro_f1")
        self.assertNotIn("test_macro_f1", json.dumps(payload))
        self.assertAlmostEqual(payload["selected"]["bert_base"]["val_macro_f1"], 0.95)
        self.assertEqual(set(payload["candidates"][0]), set(run_tuning.TUNING_FIELDS))

        final = read_csv(os.path.join(out, "final_results.csv"))
        self.assertEqual(list(final[0]), list(run_tuning.FINAL_FIELDS))
        self.assertEqual(len(final), 6)
        bert = [row for row in final if row["model_key"] == "bert_base"]
        self.assertEqual([row["seed"] for row in bert], ["42", "100", "2026"])
        self.assertTrue(all(row["learning_rate"] == "3e-05" and row["batch_size"] == "32" for row in bert))
        self.assertEqual([row["test_macro_f1"] for row in bert], ["0.050000", "0.110000", "0.130000"])
        finbert = [row for row in final if row["model_key"] == "finbert_mlm"]
        self.assertTrue(all(row["learning_rate"] == "2e-05" and row["batch_size"] == "32" for row in finbert))
        self.assertEqual(finbert[0]["test_macro_f1"], "0.070000")
        self.assertEqual(finbert[0]["external_macro_f1"], "0.400000")

    def test_incomplete_grid_does_not_select(self):
        out = os.path.join(self.tmp, "partial")
        write_fake_grid(out)
        os.remove(run_tuning.run_directory(
            os.path.join(out, run_tuning.SELECTION_DIRNAME), "finbert_mlm", 3e-5, 32, 42,
        ) + os.sep + "result.json")
        code, _stdout, err = run_tuning_main(["--out-dir", out, "--write-tables", "--phase", "select"])
        self.assertEqual(code, 2, err)
        self.assertIn("missing completed run", err)
        self.assertFalse(os.path.exists(os.path.join(out, "tuning_results.csv")))

    def test_hyperparams_in_the_file_must_match_the_directory(self):
        out = os.path.join(self.tmp, "mismatch")
        runs_dir = write_fake_grid(out)
        slipped = run_tuning.run_directory(runs_dir, "bert_base", 3e-5, 32, 42)
        with open(os.path.join(slipped, "result.json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["hyperparams"]["learning_rate"] = 2e-5
        with open(os.path.join(slipped, "result.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        code, _stdout, err = run_tuning_main(["--out-dir", out, "--write-tables", "--phase", "select"])
        self.assertEqual(code, 2, err)
        self.assertIn("learning_rate", err)
        self.assertFalse(os.path.exists(os.path.join(out, "tuning_results.csv")))

    def test_final_execute_launches_validation_winner_only(self):
        out = os.path.join(self.tmp, "launch")
        write_fake_grid(out)
        calls = []

        def runner(argv):
            calls.append(list(argv))
            return 1

        code, _stdout, err = run_tuning_main(
            ["--out-dir", out, "--phase", "final", "--execute"], runner=runner,
        )
        self.assertEqual(code, 1, err)
        self.assertEqual(len(calls), 1)
        argv = calls[0]
        self.assertEqual(flag_values(argv, "--models"), ["bert_base"])
        self.assertEqual(flag_values(argv, "--seeds"), ["42", "100", "2026"])
        self.assertEqual(flag_values(argv, "--learning-rate"), ["3e-05"])
        self.assertEqual(flag_values(argv, "--batch-size"), ["32"])
        self.assertNotIn("--val-only", argv)
        self.assertEqual(flag_values(argv, "--out-dir"), [os.path.join(out, run_tuning.FINAL_DIRNAME)])
        self.assertTrue(os.path.isfile(os.path.join(out, "tuning_results.csv")))
        self.assertFalse(os.path.exists(os.path.join(out, "final_results.csv")))

    def test_selection_rejects_saved_holdout_metrics(self):
        out = os.path.join(self.tmp, "contaminated")
        selection_dir = os.path.join(out, run_tuning.SELECTION_DIRNAME)
        for (model, learning_rate, batch_size), (val_f1, test_f1) in FAKE_GRID.items():
            write_result(selection_dir, model, learning_rate, batch_size, 42, val_f1, test_f1, holdout=True)
        code, _stdout, err = run_tuning_main(["--out-dir", out, "--write-tables", "--phase", "select"])
        self.assertEqual(code, 2, err)
        self.assertIn("validation-only", err)
        self.assertFalse(os.path.exists(os.path.join(out, "tuning_results.csv")))


class TuningCommandTests(unittest.TestCase):
    def test_plan_only_does_not_train_or_write_tables(self):
        calls = []

        def runner(argv):
            calls.append(list(argv))
            return 0

        out = os.path.join(tempfile.mkdtemp(prefix="tuning_plan_"), "out")
        self.addCleanup(shutil.rmtree, os.path.dirname(out), True)
        code, stdout, err = run_tuning_main(["--out-dir", out, "--splits-dir", "unused"], runner=runner)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("--execute" not in call for call in calls))
        self.assertEqual(
            [tuple(flag_values(call, "--learning-rate") + flag_values(call, "--batch-size")) for call in calls],
            [("2e-05", "16"), ("2e-05", "32"), ("3e-05", "16"), ("3e-05", "32")],
        )
        for call in calls:
            self.assertEqual(flag_values(call, "--models"), ["bert_base", "finbert_mlm"])
            self.assertEqual(flag_values(call, "--seeds"), ["42"])
            self.assertEqual(flag_values(call, "--configs"), ["sentences_allagree"])
            self.assertEqual(flag_values(call, "--out-dir"), [os.path.join(out, run_tuning.SELECTION_DIRNAME)])
            self.assertIn("--val-only", call)
            self.assertNotIn("--limit-train", call)
            self.assertNotIn("--max-epochs", call)
        self.assertIn("not complete", stdout)
        self.assertFalse(os.path.exists(os.path.join(out, "tuning_results.csv")))

    def test_training_failure_does_not_write_a_winner(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            return 1

        out = os.path.join(tempfile.mkdtemp(prefix="tuning_fail_"), "out")
        self.addCleanup(shutil.rmtree, os.path.dirname(out), True)
        code, _stdout, _err = run_tuning_main(
            ["--out-dir", out, "--phase", "select", "--execute"], runner=runner,
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(os.path.exists(os.path.join(out, "tuning_results.csv")))

    def test_refuses_to_write_inside_step2_results(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            return 0

        for out in (
            os.path.join(run_tuning.REPO_ROOT, "results", "step2"),
            os.path.join(run_tuning.REPO_ROOT, "results", "step2", "runs"),
        ):
            code, _stdout, err = run_tuning_main(["--out-dir", out, "--execute"], runner=runner)
            self.assertEqual(code, 2, err)
            self.assertIn("results/step2", err)
        self.assertEqual(calls, [])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tuning_pipeline_")
        cls.splits = os.path.join(cls.tmp, "splits")
        os.makedirs(cls.splits)
        synth = generate_synthetic_dataset(n_per_class=40, seed=42)
        phrasebank = synth["phrasebank_synthetic"]
        records = build_records(phrasebank["texts"], phrasebank["labels"], prefix="sentences_allagree")
        manifest = create_stratified_splits(records, seed=42, n_folds=5)
        manifest_path = os.path.join(cls.splits, "sentences_allagree_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        external_path = os.path.join(cls.splits, "sentfin_dedup.json")
        with open(external_path, "w", encoding="utf-8") as handle:
            json.dump({"stats": {}, "records": synth["sentfin_synthetic"]["records"]}, handle)
        lines = []
        for path in (manifest_path, external_path):
            lines.append(f"{file_sha256(path)}  {os.path.basename(path)}")
        with open(os.path.join(cls.splits, "SHA256SUMS"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        loaded = load_manifest(manifest_path)
        cls.pb_map = label_map((rec["text"], rec["label"]) for rec in loaded["records_by_id"].values())
        cls.sf_map = label_map(
            (rec["text"], rec["label"]) for rec in synth["sentfin_synthetic"]["records"]
        )
        _ids, _texts, val_y = split_arrays(loaded, "val")
        _ids, test_texts, test_y = split_arrays(loaded, "test")
        if len(set(val_y)) < 2 or len(set(test_y)) < 2:
            raise AssertionError("synthetic val/test split is missing a class")
        if any(text not in cls.pb_map for text in test_texts):
            raise AssertionError("test text missing from the label map")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_explicit_default_and_other_recipes_do_not_overwrite(self):
        out = os.path.join(self.tmp, "recipes")
        stub = StubCalls()
        recipes = ((2e-5, 16), (2e-5, 32), (3e-5, 16), (3e-5, 32))
        for learning_rate, batch_size in recipes:
            code, _stdout, err, _stub = run_step2_main(
                ["--stage", "finetune", "--configs", "sentences_allagree", "--models", "bert_base",
                 "--seeds", "42", "--learning-rate", run_step2.format_learning_rate(learning_rate),
                 "--batch-size", str(batch_size), "--splits-dir", self.splits, "--out-dir", out, "--execute"],
                stub,
            )
            self.assertEqual(code, 0, err)
        self.assertEqual(stub.finetune, 4)
        code, _stdout, err, _stub = run_step2_main(
            ["--stage", "finetune", "--configs", "sentences_allagree", "--models", "bert_base",
             "--seeds", "42", "--learning-rate", "2e-5", "--batch-size", "16",
             "--splits-dir", self.splits, "--out-dir", out, "--execute"],
            stub,
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(stub.finetune, 4)
        base = os.path.join(out, "sentences_allagree", "bert_base")
        self.assertTrue(os.path.isfile(os.path.join(base, "seed42", "result.json")))
        self.assertFalse(os.path.exists(os.path.join(base, "lr2e-05_bs16_seed42")))
        with open(os.path.join(base, "seed42", "result.json"), encoding="utf-8") as handle:
            default_result = json.load(handle)
        with open(os.path.join(base, "lr3e-05_bs32_seed42", "result.json"), encoding="utf-8") as handle:
            tuned_result = json.load(handle)
        self.assertAlmostEqual(default_result["hyperparams"]["learning_rate"], 2e-5)
        self.assertEqual(default_result["hyperparams"]["batch_size"], 16)
        self.assertAlmostEqual(tuned_result["hyperparams"]["learning_rate"], 3e-5)
        self.assertEqual(tuned_result["hyperparams"]["batch_size"], 32)
        for result in (default_result, tuned_result):
            self.assertEqual(result["hyperparams"]["max_epochs"], 5)
            self.assertEqual(result["hyperparams"]["eval_batch_size"], 64)
            self.assertEqual(result["hyperparams"]["weight_decay"], 0.01)
            self.assertEqual(result["hyperparams"]["warmup_ratio"], 0.1)
            self.assertEqual(result["hyperparams"]["patience"], 2)
            self.assertEqual(result["hyperparams"]["max_length"], 128)
            self.assertEqual(result["hyperparams"]["grad_clip"], 1.0)
            self.assertFalse(result["probe"])
            self.assertIsNotNone(result["metrics"]["test"])
            self.assertIsNotNone(result["metrics"]["external"])

    def test_plan_only_lists_eight_distinct_runs_and_writes_nothing(self):
        out = os.path.join(self.tmp, "plan")
        code, stdout, err = run_tuning_main(["--splits-dir", self.splits, "--out-dir", out])
        self.assertEqual(code, 0, err)
        self.assertEqual(stdout.count("[finetune]"), 8)
        self.assertIn("lr3e-05_bs32_seed42", stdout)
        self.assertIn(os.path.join("bert_base", "seed42"), stdout)
        self.assertIn("not complete", stdout)
        self.assertFalse(os.path.exists(out))

    def test_execute_selects_on_validation_and_retrains_seed_42(self):
        out = os.path.join(self.tmp, "study")
        step2_custom = os.path.join(
            run_tuning.REPO_ROOT, "results", "step2", "runs", "sentences_allagree", "bert_base",
            "lr3e-05_bs32_seed42",
        )
        tuning_root = os.path.join(run_tuning.REPO_ROOT, "results", "tuning")
        tuning_existed = os.path.exists(tuning_root)
        self.assertFalse(os.path.exists(step2_custom))
        stub = RankingStub(self.pb_map, self.sf_map)
        calls = []

        def runner(argv):
            calls.append(list(argv))
            return run_step2.main(
                argv, finetune_fn=stub.finetune_fn, zero_shot_fn=stub.zero_shot_fn,
                baseline_factory=lambda *args, **kwargs: None,
            )

        code, _stdout, err = run_tuning_main(
            ["--splits-dir", self.splits, "--out-dir", out, "--phase", "all", "--execute"],
            runner=runner,
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(len(calls), 6)
        self.assertEqual(stub.finetune, 14)
        self.assertEqual(stub.seen_eval_sets[:8], [[]] * 8)
        self.assertEqual(stub.seen_eval_sets[8:], [["external", "test"]] * 6)
        self.assertEqual(
            [flag_values(call, "--seeds") for call in calls[:4]],
            [["42"], ["42"], ["42"], ["42"]],
        )
        self.assertTrue(all("--val-only" in call for call in calls[:4]))
        self.assertTrue(all(
            flag_values(call, "--out-dir") == [os.path.join(out, run_tuning.SELECTION_DIRNAME)]
            for call in calls[:4]
        ))
        self.assertEqual(flag_values(calls[4], "--models"), ["bert_base"])
        self.assertEqual(flag_values(calls[4], "--learning-rate"), ["3e-05"])
        self.assertEqual(flag_values(calls[4], "--batch-size"), ["32"])
        self.assertEqual(flag_values(calls[4], "--seeds"), ["42", "100", "2026"])
        self.assertEqual(flag_values(calls[5], "--models"), ["finbert_mlm"])
        self.assertEqual(flag_values(calls[5], "--learning-rate"), ["2e-05"])
        self.assertEqual(flag_values(calls[5], "--batch-size"), ["32"])
        for call in calls[4:]:
            self.assertNotIn("--val-only", call)
            self.assertEqual(flag_values(call, "--out-dir"), [os.path.join(out, run_tuning.FINAL_DIRNAME)])
        for call in calls:
            self.assertEqual(flag_values(call, "--configs"), ["sentences_allagree"])
            self.assertNotIn("--limit-train", call)
            self.assertNotIn("--max-epochs", call)

        def read_result(root, model, learning_rate, batch_size, seed=42):
            path = os.path.join(
                run_tuning.run_directory(os.path.join(out, root), model, learning_rate, batch_size, seed),
                "result.json",
            )
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)

        for model, (learning_rate, batch_size) in WINNERS.items():
            selected = read_result(run_tuning.SELECTION_DIRNAME, model, learning_rate, batch_size)
            decoy = read_result(run_tuning.SELECTION_DIRNAME, model, 2e-5, 16)
            final = read_result(run_tuning.FINAL_DIRNAME, model, learning_rate, batch_size)
            self.assertGreater(selected["metrics"]["val"]["macro_f1"], decoy["metrics"]["val"]["macro_f1"])
            self.assertIsNone(selected["metrics"]["test"])
            self.assertIsNone(selected["metrics"]["external"])
            self.assertIsNone(selected["n"]["test"])
            self.assertIsNone(selected["n"]["external"])
            self.assertIsNone(selected["external_sha256"])
            self.assertIsNone(decoy["metrics"]["test"])
            self.assertIsNotNone(final["metrics"]["test"])
            self.assertIsNotNone(final["metrics"]["external"])
            selection_dir = run_tuning.run_directory(
                os.path.join(out, run_tuning.SELECTION_DIRNAME), model, learning_rate, batch_size, 42,
            )
            with np.load(os.path.join(selection_dir, "predictions.npz")) as npz:
                self.assertEqual(set(npz.files), {"val_ids", "val_y", "val_prob"})
            for seed in (42, 100, 2026):
                final_dir = run_tuning.run_directory(
                    os.path.join(out, run_tuning.FINAL_DIRNAME), model, learning_rate, batch_size, seed,
                )
                self.assertTrue(os.path.isfile(os.path.join(final_dir, "result.json")))
                with np.load(os.path.join(final_dir, "predictions.npz")) as npz:
                    self.assertTrue({"test_prob", "external_prob", "val_prob"} <= set(npz.files))
        self.assertFalse(os.path.exists(os.path.join(
            out, run_tuning.FINAL_DIRNAME, "sentences_allagree", "bert_base", "seed100",
        )))

        tuning = read_csv(os.path.join(out, "tuning_results.csv"))
        selected = {row["model_key"]: row for row in tuning if row["selected"] == "1"}
        self.assertEqual(selected["bert_base"]["learning_rate"], "3e-05")
        self.assertEqual(selected["bert_base"]["batch_size"], "32")
        self.assertEqual(selected["finbert_mlm"]["learning_rate"], "2e-05")
        self.assertEqual(selected["finbert_mlm"]["batch_size"], "32")
        final = read_csv(os.path.join(out, "final_results.csv"))
        self.assertEqual(len(final), 6)
        self.assertEqual(sorted({row["seed"] for row in final}), ["100", "2026", "42"])
        self.assertFalse(os.path.exists(step2_custom))
        self.assertEqual(os.path.exists(tuning_root), tuning_existed)

    def test_val_only_run_does_not_load_the_holdout_files(self):
        splits = os.path.join(self.tmp, "manifest_only")
        os.makedirs(splits)
        shutil.copy(
            os.path.join(self.splits, "sentences_allagree_manifest.json"),
            os.path.join(splits, "sentences_allagree_manifest.json"),
        )
        out = os.path.join(self.tmp, "val_only_run")
        stub = StubCalls()
        code, _stdout, err, _stub = run_step2_main(
            ["--val-only", "--execute", "--stage", "finetune", "--configs", "sentences_allagree",
             "--models", "bert_base", "finbert_mlm", "--seeds", "42",
             "--splits-dir", splits, "--out-dir", out],
            stub,
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(stub.seen_eval_sets, [[], []])
        for model in ("bert_base", "finbert_mlm"):
            run_dir = os.path.join(out, "sentences_allagree", model, "seed42")
            with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertIsNone(result["metrics"]["test"])
            self.assertIsNone(result["metrics"]["external"])
            self.assertIsNone(result["n"]["test"])
            self.assertIsNone(result["external_sha256"])
            with np.load(os.path.join(run_dir, "predictions.npz")) as npz:
                self.assertEqual(set(npz.files), {"val_ids", "val_y", "val_prob"})
        missing, _stdout, missing_err, _stub = run_step2_main(
            ["--execute", "--stage", "finetune", "--configs", "sentences_allagree",
             "--models", "bert_base", "--seeds", "42", "--splits-dir", splits,
             "--out-dir", os.path.join(self.tmp, "needs_holdout")],
        )
        self.assertEqual(missing, 2)
        self.assertIn("sentfin_dedup.json", missing_err)


class FinbertTokenizerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "essential_text_tuning")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_run(self, phase, model, tag, *, receipt=None, status="completed"):
        run_dir = os.path.join(self.out, phase, "sentences_allagree", model, tag)
        os.makedirs(run_dir)
        payload = {
            "status": status,
            "model_key": model,
            "seed": 42,
            "config": "sentences_allagree",
            "hyperparams": {"learning_rate": 2e-5, "batch_size": 32},
        }
        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        if receipt is not None:
            from src.transformer.finetune import TOKENIZER_CHECK_NAME

            with open(os.path.join(run_dir, TOKENIZER_CHECK_NAME), "w", encoding="utf-8") as handle:
                json.dump(receipt, handle)
        return run_dir

    def _write_stale_records(self):
        with open(os.path.join(self.out, "selection.json"), "w", encoding="utf-8") as handle:
            json.dump({"chosen": "broken-finbert"}, handle)
        for name in ("tuning_results.csv", "final_results.csv"):
            with open(os.path.join(self.out, name), "w", encoding="utf-8") as handle:
                handle.write("stale\n")
        for phase in ("selection", "final"):
            os.makedirs(os.path.join(self.out, phase), exist_ok=True)
            with open(os.path.join(self.out, phase, "journal.jsonl"), "w", encoding="utf-8") as handle:
                handle.write('{"event":"end","run_dir":"finbert_mlm"}\n')

    def test_archive_keeps_bert_and_a_valid_finbert_receipt(self):
        from src.transformer.finetune import injected_finbert_tokenizer_check

        bert = self._write_run("selection", "bert_base", "lr3e-05_bs32_seed42")
        bert_final = self._write_run("final", "bert_base", "lr3e-05_bs32_seed42")
        broken = self._write_run("selection", "finbert_mlm", "lr2e-05_bs32_seed42")
        broken_final = self._write_run("final", "finbert_mlm", "lr2e-05_bs32_seed42")
        valid = self._write_run(
            "selection", "finbert_mlm", "lr3e-05_bs16_seed42",
            receipt=injected_finbert_tokenizer_check(),
        )
        all_unk = {
            "vocab_size": 30873,
            "valid": True,
            "unk_token_id": 1,
            "terms": {"profit": [1], "ebitda": [1], "depreciation": [1]},
        }
        unmarked = self._write_run("final", "finbert_mlm", "lr2e-05_bs16_seed100", receipt=all_unk)
        pending = self._write_run(
            "selection", "finbert_mlm", "lr3e-05_bs32_seed42", status="failed",
        )
        with open(os.path.join(bert, "result.json"), "rb") as handle:
            bert_bytes = handle.read()
        self._write_stale_records()

        def runner(_argv):
            raise AssertionError("archiving must not train")

        code, stdout, err = run_tuning_main(
            ["--archive-invalid-finbert", "--out-dir", self.out], runner=runner,
        )
        self.assertEqual(code, 0, err)
        self.assertIn("archive: moved 3 invalid FinBERT", stdout)
        self.assertFalse(os.path.exists(broken))
        self.assertFalse(os.path.exists(broken_final))
        self.assertFalse(os.path.exists(unmarked))
        self.assertFalse(run_step2._is_completed(broken))
        self.assertTrue(os.path.isdir(bert))
        self.assertTrue(os.path.isdir(bert_final))
        with open(os.path.join(bert, "result.json"), "rb") as handle:
            self.assertEqual(handle.read(), bert_bytes)
        self.assertTrue(run_tuning.finbert_receipt_is_valid(valid))
        self.assertTrue(os.path.isdir(pending))
        self.assertFalse(os.path.exists(os.path.join(self.out, "selection.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "tuning_results.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "final_results.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "selection", "journal.jsonl")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "final", "journal.jsonl")))
        archive_root = os.path.join(self.out, "archive", "invalid_finbert_tokenizer")
        stamped = os.listdir(archive_root)
        self.assertEqual(len(stamped), 1)
        archived = os.path.join(archive_root, stamped[0])
        self.assertTrue(os.path.isfile(os.path.join(
            archived, "selection", "sentences_allagree", "finbert_mlm", "lr2e-05_bs32_seed42", "result.json",
        )))
        self.assertTrue(os.path.isfile(os.path.join(archived, "selection.json")))
        self.assertTrue(os.path.isfile(os.path.join(archived, "selection", "journal.jsonl")))
        with open(os.path.join(self.out, "selection.json"), "w", encoding="utf-8") as handle:
            json.dump({"chosen": "after-recovery"}, handle)
        again, again_out, again_err = run_tuning_main(
            ["--archive-invalid-finbert", "--out-dir", self.out], runner=runner,
        )
        self.assertEqual(again, 0, again_err)
        self.assertIn("no invalid FinBERT", again_out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "selection.json")))
        self.assertEqual(os.listdir(archive_root), stamped)
        self.assertTrue(os.path.isdir(bert))
        self.assertTrue(os.path.isdir(valid))

    def test_tables_are_not_rebuilt_from_a_broken_finbert_run(self):
        out = os.path.join(self.tmp.name, "tables")
        runs_dir = write_fake_grid(out)
        broken = run_tuning.run_directory(runs_dir, "finbert_mlm", 2e-5, 32, 42)
        os.remove(os.path.join(broken, "tokenizer_check.json"))
        with open(os.path.join(out, "tuning_results.csv"), "w", encoding="utf-8") as handle:
            handle.write("keep-me\n")
        code, _stdout, err = run_tuning_main(
            ["--out-dir", out, "--write-tables", "--phase", "select"],
        )
        self.assertEqual(code, 2, err)
        self.assertIn("tokenizer", err)
        with open(os.path.join(out, "tuning_results.csv"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "keep-me\n")

    def test_execute_archives_broken_finbert_before_resume(self):
        broken = self._write_run("selection", "finbert_mlm", "lr2e-05_bs32_seed42")
        bert = self._write_run("selection", "bert_base", "seed42")
        seen = []

        def runner(_argv):
            seen.append(os.path.exists(broken))
            seen.append(run_step2._is_completed(bert))
            return 0

        code, stdout, err = run_tuning_main(
            ["--execute", "--phase", "select", "--out-dir", self.out, "--splits-dir", self.tmp.name],
            runner=runner,
        )
        self.assertEqual(seen[0], False, err)
        self.assertEqual(seen[1::2], [True, True, True, True])
        self.assertIn("archive: moved 1 invalid FinBERT", stdout)
        self.assertEqual(code, 2)
        self.assertFalse(run_step2._is_completed(broken))
        self.assertTrue(run_step2._is_completed(bert))


if __name__ == "__main__":
    unittest.main()
