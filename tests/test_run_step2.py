"""Tests for src/data/io.py and scripts/run_step2.py on synthetic data (no network, no models)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts import run_step2
from src.baselines.lexicon_official import LMOfficialBaseline, LMOfficialTunedBaseline
from src.baselines.registry import make_baseline
from src.data.download import generate_synthetic_dataset
from src.data.io import CONFIGS, cv_fold_arrays, file_sha256, load_external, load_manifest, split_arrays
from src.data.splits import build_records, create_stratified_splits

BASELINES = (
    "vader_official", "vader_official_tuned", "lm_official", "lm_official_tuned", "vader", "loughran_mcdonald",
    "tfidf_logreg", "tfidf_svm",
)
SEEDS = (42, 100, 2026)
RESULT_KEYS = {
    "status", "stage", "config", "model_key", "model_name", "seed", "fold", "hyperparams", "best_epoch",
    "history", "metrics", "n", "device", "versions", "manifest_sha256", "external_sha256", "probe",
    "wall_seconds", "timestamp",
}


class StubCalls:
    """Records calls and returns deterministic random probabilities in the Step 2 hook format."""

    def __init__(self, fail_models=()):
        self.finetune = 0
        self.zero_shot = 0
        self.fail_models = set(fail_models)
        self.train_sizes = []

    @staticmethod
    def _prob(n, seed):
        raw = np.random.default_rng(seed).random((n, 3)) + 0.05
        return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    def finetune_fn(self, model_name, train_texts, train_labels, val_texts, val_labels, eval_sets, config,
                    revision=None, model_loader=None):
        self.finetune += 1
        self.train_sizes.append(len(train_texts))
        if model_name in self.fail_models:
            raise RuntimeError("boom " + model_name)
        return {
            "best_epoch": 2,
            "history": [{"epoch": 1, "train_loss": 1.0, "val_macro_f1": 0.3, "seconds": 0.1}],
            "val_prob": self._prob(len(val_texts), config.seed),
            "eval_probs": {k: self._prob(len(v), config.seed + 1) for k, v in eval_sets.items()},
            "wall_seconds": 0.1,
            "device": "cpu",
            "config": config.to_dict(),
        }

    def zero_shot_fn(self, model_name, texts, config, revision=None, model_loader=None):
        self.zero_shot += 1
        return self._prob(len(texts), 7)


def baseline_factory(model_key, seed=42):
    """Registry factory, but the official LM dictionary is replaced by tiny injected lists (no data/raw)."""
    if model_key == "lm_official":
        return LMOfficialBaseline(words=({"profit", "gain"}, {"loss", "decline"}))
    if model_key == "lm_official_tuned":
        return LMOfficialTunedBaseline(words=({"profit", "gain"}, {"loss", "decline"}))
    return make_baseline(model_key, seed=seed)


def run_main(argv, stub=None, factory=baseline_factory):
    stub = stub or StubCalls()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_step2.main(
            argv, finetune_fn=stub.finetune_fn, zero_shot_fn=stub.zero_shot_fn,
            baseline_factory=factory,
        )
    return code, out.getvalue(), err.getvalue()


def tree(path):
    return sorted(
        os.path.relpath(os.path.join(root, f), path) for root, _, files in os.walk(path) for f in files
    )


class RunStep2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="run_step2_test_")
        cls.splits = os.path.join(cls.tmp, "splits")
        os.makedirs(cls.splits)
        synth = generate_synthetic_dataset(n_per_class=40, seed=42)
        pb = synth["phrasebank_synthetic"]
        for cfg in CONFIGS:
            records = build_records(pb["texts"], pb["labels"], prefix=cfg)
            manifest = create_stratified_splits(records, seed=42, n_folds=5)
            with open(os.path.join(cls.splits, f"{cfg}_manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh)
        with open(os.path.join(cls.splits, "sentfin_dedup.json"), "w", encoding="utf-8") as fh:
            json.dump({"stats": {}, "records": synth["sentfin_synthetic"]["records"]}, fh)

        cls.full_out = os.path.join(cls.tmp, "full")
        cls.full_stub = StubCalls()
        cls.full_code, cls.full_stdout, cls.full_stderr = run_main(
            ["--splits-dir", cls.splits, "--out-dir", cls.full_out, "--execute"], cls.full_stub
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def argv(self, out_dir, *extra):
        return ["--splits-dir", self.splits, "--out-dir", out_dir, *extra]

    def fresh_out(self, name):
        path = os.path.join(self.tmp, name)
        self.addCleanup(shutil.rmtree, path, True)
        return path

    # ---- io helpers -------------------------------------------------------------------------

    def test_io_helpers(self):
        path = os.path.join(self.splits, "sentences_allagree_manifest.json")
        manifest = load_manifest(path)
        self.assertEqual(len(file_sha256(path)), 64)
        ids, texts, labels = split_arrays(manifest, "test")
        self.assertEqual(len(ids), len(manifest["splits"]["test"]))
        self.assertEqual(len(texts), len(labels))
        f_ids, _, _ = cv_fold_arrays(manifest, 2, "val")
        self.assertEqual(f_ids, manifest["cv_folds"][2]["val"])
        with self.assertRaises(ValueError):
            cv_fold_arrays(manifest, 0, "test")
        e_ids, e_texts, e_labels = load_external(os.path.join(self.splits, "sentfin_dedup.json"))
        self.assertTrue(len(e_ids) == len(e_texts) == len(e_labels) > 0)

    # ---- plan only --------------------------------------------------------------------------

    def test_plan_only_writes_nothing(self):
        out = self.fresh_out("plan_only")
        code, stdout, _ = run_main(self.argv(out))
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(out))
        lines = [ln for ln in stdout.splitlines() if ln.startswith("[")]
        self.assertEqual(len(lines), 24 + 18 + 50 + 1)
        self.assertTrue(all(ln.endswith(" pending") for ln in lines))

    # ---- full matrix ------------------------------------------------------------------------

    def test_full_matrix_runs_clean(self):
        self.assertEqual(self.full_code, 0, self.full_stderr)

    def test_matrix_layout(self):
        expected = set()
        for cfg in CONFIGS:
            for key in BASELINES:
                expected.add(f"{cfg}/{key}/seed42")
            for key in ("bert_base", "finbert_mlm"):
                for seed in SEEDS:
                    expected.add(f"{cfg}/{key}/seed{seed}")
        for key in BASELINES + ("bert_base", "finbert_mlm"):
            for fold in range(5):
                expected.add(f"sentences_allagree/{key}/cv_fold{fold}")
        expected.add("external_only/prosus_finbert/zeroshot")
        found = set()
        for root, _, files in os.walk(self.full_out):
            if "result.json" in files:
                found.add(os.path.relpath(root, self.full_out))
                self.assertIn("predictions.npz", files)
        self.assertEqual(found, expected)
        self.assertFalse([d for d, _, _ in os.walk(self.full_out) if d.endswith(".tmp")])

    def test_official_baselines_have_fixed_rules(self):
        vader = self.read_result("sentences_allagree/vader_official/seed42")
        self.assertEqual(vader["model_key"], "vader_official")
        self.assertFalse(vader["hyperparams"]["tuned"])
        self.assertEqual(vader["hyperparams"]["thresholds"], [-0.05, 0.05])
        lm = self.read_result("sentences_75agree/lm_official/seed42")
        self.assertFalse(lm["hyperparams"]["tuned"])
        self.assertEqual(lm["stage"], "baselines")
        cv = self.read_result("sentences_allagree/lm_official/cv_fold2")
        self.assertEqual(cv["stage"], "cv")

    def test_tuned_official_baselines_are_in_the_plan_and_tuned(self):
        self.assertIn("vader_official_tuned", run_step2.BASELINE_KEYS)
        self.assertIn("lm_official_tuned", run_step2.BASELINE_KEYS)
        for cfg in ("sentences_allagree", "sentences_75agree"):
            for key in ("vader_official_tuned", "lm_official_tuned"):
                res = self.read_result(f"{cfg}/{key}/seed42")
                self.assertEqual(res["model_key"], key)
                self.assertEqual(res["stage"], "baselines")
                self.assertTrue(res["hyperparams"]["tuned"])
        cv = self.read_result("sentences_allagree/lm_official_tuned/cv_fold1")
        self.assertEqual(cv["stage"], "cv")
        self.assertTrue(cv["hyperparams"]["negation"])

    def test_prosus_never_scores_phrasebank(self):
        for cfg in CONFIGS:
            self.assertFalse(os.path.exists(os.path.join(self.full_out, cfg, "prosus_finbert")))
        self.assertEqual(self.full_stub.zero_shot, 1)

    def read_result(self, rel):
        with open(os.path.join(self.full_out, rel, "result.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_result_json_schema(self):
        cases = {
            "sentences_allagree/tfidf_svm/seed42": ("baselines", None, ("val", "test", "external")),
            "sentences_75agree/bert_base/seed100": ("finetune", None, ("val", "test", "external")),
            "sentences_allagree/finbert_mlm/cv_fold3": ("cv", 3, ("val",)),
            "external_only/prosus_finbert/zeroshot": ("zeroshot", None, ("external",)),
        }
        for rel, (stage, fold, splits) in cases.items():
            res = self.read_result(rel)
            self.assertEqual(set(res), RESULT_KEYS, rel)
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["stage"], stage)
            self.assertEqual(res["fold"], fold)
            self.assertFalse(res["probe"])
            for s in ("val", "test", "external"):
                if s in splits:
                    self.assertIn("macro_f1", res["metrics"][s], (rel, s))
                else:
                    self.assertIsNone(res["metrics"][s], (rel, s))
            self.assertEqual(set(res["n"]), {"train", "val", "test", "external"})
            self.assertEqual(set(res["versions"]), {"python", "numpy", "sklearn", "torch", "transformers"})
        zs = self.read_result("external_only/prosus_finbert/zeroshot")
        self.assertIsNone(zs["manifest_sha256"])
        self.assertEqual(len(zs["external_sha256"]), 64)
        self.assertEqual(set(zs["hyperparams"]), {"max_length", "eval_batch_size", "device"})
        ft = self.read_result("sentences_75agree/bert_base/seed100")
        self.assertEqual(ft["seed"], 100)
        self.assertEqual(ft["best_epoch"], 2)
        self.assertEqual(ft["hyperparams"]["seed"], 100)
        self.assertEqual(ft["model_name"], "google-bert/bert-base-uncased")
        self.assertEqual(len(ft["manifest_sha256"]), 64)
        base = self.read_result("sentences_allagree/tfidf_svm/seed42")
        self.assertIsNone(base["best_epoch"])
        self.assertEqual(base["history"], [])

    def test_predictions_npz_arrays(self):
        rel = "sentences_allagree/bert_base/seed42"
        res = self.read_result(rel)
        with np.load(os.path.join(self.full_out, rel, "predictions.npz")) as npz:
            self.assertEqual(
                set(npz.files),
                {f"{s}_{k}" for s in ("val", "test", "external") for k in ("ids", "y", "prob")},
            )
            for s in ("val", "test", "external"):
                n = res["n"][s]
                self.assertEqual(npz[f"{s}_ids"].shape, (n,))
                self.assertEqual(npz[f"{s}_ids"].dtype.kind, "U")
                self.assertEqual(npz[f"{s}_y"].dtype, np.int64)
                self.assertEqual(npz[f"{s}_prob"].dtype, np.float32)
                self.assertEqual(npz[f"{s}_prob"].shape, (n, 3))
        rel = "external_only/prosus_finbert/zeroshot"
        with np.load(os.path.join(self.full_out, rel, "predictions.npz")) as npz:
            self.assertEqual(set(npz.files), {"external_ids", "external_y", "external_prob"})

    def test_cv_uses_fold_val_only(self):
        with open(os.path.join(self.splits, "sentences_allagree_manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        rel = "sentences_allagree/vader/cv_fold1"
        with np.load(os.path.join(self.full_out, rel, "predictions.npz")) as npz:
            self.assertEqual(list(npz["val_ids"]), manifest["cv_folds"][1]["val"])

    def test_journal_has_start_and_end_for_every_run(self):
        with open(os.path.join(self.full_out, "journal.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(ln) for ln in fh]
        self.assertEqual(sum(e["event"] == "start" for e in events), 93)
        self.assertEqual(sum(e["event"] == "end" for e in events), 93)
        self.assertFalse(any(e["event"] == "error" for e in events))
        self.assertEqual(set(events[0]), {"event", "run_dir", "wall_seconds", "timestamp", "error"})

    # ---- resume, stale tmp, errors ----------------------------------------------------------

    def test_resume_skips_completed_runs(self):
        stub = StubCalls()
        code, stdout, _ = run_main(self.argv(self.full_out, "--execute"), stub)
        self.assertEqual(code, 0)
        self.assertEqual((stub.finetune, stub.zero_shot), (0, 0))
        self.assertIn("0 pending", stdout)

    def test_resume_runs_only_the_missing_run(self):
        out = self.fresh_out("resume")
        args = self.argv(out, "--stage", "finetune", "--configs", "sentences_allagree",
                         "--models", "bert_base", "--seeds", "42", "100")
        stub = StubCalls()
        self.assertEqual(run_main(args + ["--execute"], stub)[0], 0)
        self.assertEqual(stub.finetune, 2)
        shutil.rmtree(os.path.join(out, "sentences_allagree", "bert_base", "seed100"))
        self.assertEqual(run_main(args + ["--execute"], stub)[0], 0)
        self.assertEqual(stub.finetune, 3)

    def test_stale_tmp_dir_is_cleaned(self):
        out = self.fresh_out("stale")
        run_dir = os.path.join(out, "sentences_allagree", "bert_base", "seed42")
        os.makedirs(run_dir + ".tmp")
        with open(os.path.join(run_dir + ".tmp", "junk.txt"), "w") as fh:
            fh.write("half written")
        args = self.argv(out, "--stage", "finetune", "--configs", "sentences_allagree",
                         "--models", "bert_base", "--seeds", "42", "--execute")
        self.assertEqual(run_main(args)[0], 0)
        self.assertFalse(os.path.exists(run_dir + ".tmp"))
        self.assertEqual(sorted(os.listdir(run_dir)), ["predictions.npz", "result.json"])

    def test_error_in_one_run_is_journalled_and_others_finish(self):
        out = self.fresh_out("errors")
        stub = StubCalls(fail_models={"yiyanghkust/finbert-pretrain"})
        args = self.argv(out, "--stage", "finetune", "--configs", "sentences_allagree",
                         "--seeds", "42", "100", "--execute")
        code, _, stderr = run_main(args, stub)
        self.assertEqual(code, 1)
        self.assertIn("boom", stderr)
        base = os.path.join(out, "sentences_allagree")
        for seed in (42, 100):
            self.assertTrue(os.path.isfile(os.path.join(base, "bert_base", f"seed{seed}", "result.json")))
            self.assertFalse(os.path.exists(os.path.join(base, "finbert_mlm", f"seed{seed}")))
        self.assertFalse([d for d, _, _ in os.walk(out) if d.endswith(".tmp")])
        with open(os.path.join(out, "journal.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(ln) for ln in fh]
        errors = [e for e in events if e["event"] == "error"]
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("boom" in e["error"] for e in errors))
        self.assertEqual(sum(e["event"] == "end" for e in events), 2)

    # ---- guards -----------------------------------------------------------------------------

    def test_probe_flags_with_default_out_dir_exit_2(self):
        default = os.path.abspath(run_step2.DEFAULT_OUT_DIR)
        existed = os.path.exists(default)
        for flag in (["--limit-train", "20"], ["--max-epochs", "1"]):
            code, _, stderr = run_main(["--splits-dir", self.splits, "--execute", *flag])
            self.assertEqual(code, 2)
            self.assertIn("probe", stderr)
        self.assertEqual(os.path.exists(default), existed)

    def test_probe_run_in_custom_out_dir_is_flagged(self):
        out = self.fresh_out("probe")
        stub = StubCalls()
        args = self.argv(out, "--stage", "finetune", "--configs", "sentences_allagree",
                         "--models", "bert_base", "--seeds", "42", "--limit-train", "20",
                         "--max-epochs", "1", "--execute")
        self.assertEqual(run_main(args, stub)[0], 0)
        self.assertEqual(stub.train_sizes, [20])
        with open(os.path.join(out, "sentences_allagree", "bert_base", "seed42", "result.json")) as fh:
            res = json.load(fh)
        self.assertTrue(res["probe"])
        self.assertEqual(res["n"]["train"], 20)
        self.assertEqual(res["hyperparams"]["max_epochs"], 1)

    def _copy_splits(self, name):
        path = self.fresh_out(name)
        shutil.copytree(self.splits, path)
        return path

    @staticmethod
    def _write_sums(splits_dir, overrides=None):
        lines = []
        for fname in sorted(os.listdir(splits_dir)):
            if fname == "SHA256SUMS":
                continue
            digest = (overrides or {}).get(fname) or file_sha256(os.path.join(splits_dir, fname))
            lines.append(f"{digest}  {fname}")
        with open(os.path.join(splits_dir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_sha256sums_match_runs(self):
        splits = self._copy_splits("sums_ok")
        self._write_sums(splits)
        out = self.fresh_out("sums_ok_out")
        code, stdout, _ = run_main(["--splits-dir", splits, "--out-dir", out, "--stage", "baselines",
                                    "--configs", "sentences_allagree", "--models", "vader_official"])
        self.assertEqual(code, 0)
        self.assertIn("plan: 1 runs", stdout)

    def test_sha256sums_mismatch_exits_2_naming_file(self):
        splits = self._copy_splits("sums_bad")
        self._write_sums(splits, {"sentences_allagree_manifest.json": "0" * 64})
        out = self.fresh_out("sums_bad_out")
        code, _, stderr = run_main(["--splits-dir", splits, "--out-dir", out, "--execute",
                                    "--stage", "baselines", "--configs", "sentences_allagree"])
        self.assertEqual(code, 2)
        self.assertIn("sentences_allagree_manifest.json", stderr)
        self.assertFalse(os.path.exists(out))

    def test_sha256sums_mismatch_of_external_file_exits_2(self):
        splits = self._copy_splits("sums_bad_ext")
        self._write_sums(splits, {"sentfin_dedup.json": "f" * 64})
        code, _, stderr = run_main(["--splits-dir", splits, "--out-dir", self.fresh_out("sums_bad_ext_out"),
                                    "--stage", "zeroshot"])
        self.assertEqual(code, 2)
        self.assertIn("sentfin_dedup.json", stderr)

    def test_missing_sha256sums_is_only_a_warning(self):
        code, _, stderr = run_main(self.argv(self.fresh_out("no_sums"), "--stage", "zeroshot"))
        self.assertEqual(code, 0)
        self.assertIn("SHA256SUMS", stderr)

    def test_nonfinite_probabilities_fail_the_run(self):
        out = self.fresh_out("nan")

        class NanBaseline:
            def fit(self, *args, **kwargs):
                return self

            def predict_proba(self, texts):
                prob = np.full((len(texts), 3), 1 / 3)
                prob[0, 0] = np.nan
                return prob

        code, _, stderr = run_main(
            self.argv(out, "--stage", "baselines", "--configs", "sentences_allagree",
                      "--models", "vader_official", "--execute"),
            factory=lambda key, seed=42: NanBaseline(),
        )
        self.assertEqual(code, 1)
        self.assertIn("non-finite", stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "sentences_allagree", "vader_official", "seed42")))

    def test_degenerate_transformer_output_fails_the_run(self):
        out = self.fresh_out("degenerate")

        class CollapsedStub(StubCalls):
            def finetune_fn(self, model_name, train_texts, train_labels, val_texts, val_labels, eval_sets,
                            config, revision=None, model_loader=None):
                res = super().finetune_fn(model_name, train_texts, train_labels, val_texts, val_labels,
                                          eval_sets, config, revision, model_loader)
                res["val_prob"] = np.tile(np.array([0.1, 0.8, 0.1], dtype=np.float32), (len(val_texts), 1))
                return res

        code, _, stderr = run_main(
            self.argv(out, "--stage", "finetune", "--configs", "sentences_allagree", "--models", "bert_base",
                      "--seeds", "42", "--execute"),
            CollapsedStub(),
        )
        self.assertEqual(code, 1)
        self.assertIn("degenerate", stderr)
        with open(os.path.join(out, "journal.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(ln) for ln in fh]
        self.assertTrue(any(e["event"] == "error" and "degenerate" in e["error"] for e in events))

    def test_baseline_factory_receives_model_key(self):
        seen = []

        def factory(model_key, seed=42):
            seen.append((model_key, seed))
            return baseline_factory(model_key, seed=seed)

        out = self.fresh_out("factory")
        code, _, _ = run_main(
            self.argv(out, "--stage", "baselines", "--configs", "sentences_allagree",
                      "--models", "lm_official", "tfidf_svm", "--execute"),
            factory=factory,
        )
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("lm_official", 42), ("tfidf_svm", 42)])

    def test_missing_inputs_exit_2(self):
        empty = os.path.join(self.tmp, "empty_splits")
        os.makedirs(empty, exist_ok=True)
        out = self.fresh_out("missing")
        code, _, stderr = run_main(["--splits-dir", empty, "--out-dir", out, "--execute"])
        self.assertEqual(code, 2)
        self.assertIn("sentences_allagree_manifest.json", stderr)
        self.assertFalse(os.path.exists(out))

    def test_unknown_config_and_model_exit_2(self):
        out = self.fresh_out("unknown")
        self.assertEqual(run_main(self.argv(out, "--configs", "sentences_nope"))[0], 2)
        self.assertEqual(run_main(self.argv(out, "--models", "gpt"))[0], 2)


if __name__ == "__main__":
    unittest.main()
