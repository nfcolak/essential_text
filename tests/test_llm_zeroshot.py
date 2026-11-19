"""Offline coverage for the Ollama runner and fixed-recipe paired comparisons."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_llm_zeroshot as runner, summarize_llm_zeroshot as summary


class TestZeroShot(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "out"
        self.splits = self.root / "splits"
        self.splits.mkdir()
        self.phrase = runner.DATASETS[0]
        self.examples = {}
        extra = {}
        for name in runner.DATASETS:
            records = [{"id": f"{name}_{i}", "text": f"Sentence {name} {i}", "label": i} for i in range(3)]
            self.examples[name] = [r["id"] for r in records]
            if name in runner.PHRASEBANK:
                manifest = {"splits": {"test": self.examples[name]},
                            "records_by_id": {r["id"]: r for r in records}}
                self.write(self.splits / (runner.PHRASEBANK[name] + "_manifest.json"), manifest)
            elif name == "sentfin":
                self.write(self.splits / "sentfin_dedup.json", {"records": records})
            else:
                extra[name] = {"records": records}
        self.write(self.splits / "extra_external.json", {"datasets": extra})
        self.replies = []
        self.requests = []
        self.digest = "mock-digest"
        # Includes the HTTP boundary: no DNS or local model needed.
        mock = patch.object(runner.request, "urlopen", side_effect=self.http)
        self.urlopen = mock.start()
        self.addCleanup(mock.stop)
        self.stderr = io.StringIO()
        for cm in (redirect_stdout(io.StringIO()), redirect_stderr(self.stderr)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8")

    def http(self, req, timeout):
        if req.full_url == runner.URL + "/api/tags":
            data = {"models": [{"name": runner.DEFAULT_MODEL, "digest": self.digest}]}
        else:
            self.assertEqual(req.full_url, runner.URL + "/api/generate")
            payload = json.loads(req.data)
            self.requests.append(payload)
            index = int(payload["prompt"].split()[-1])
            raw = self.replies.pop(0) if self.replies else json.dumps({"label": runner.CLASS_NAMES[index]})
            data = {"response": raw}
        return io.BytesIO(json.dumps(data).encode())

    def run_args(self, *args):
        return runner.main(["--out", str(self.out), "--splits-dir", str(self.splits),
                            "--workers", "1", *args])

    def test_plan_and_exact_schema(self):
        self.assertEqual(self.run_args(), 0)
        self.urlopen.assert_not_called()
        self.assertFalse(self.out.exists())
        for label in runner.CLASS_NAMES:
            self.assertEqual(runner.parse_label(json.dumps({"label": label})), label)
        for raw in ('"positive"', 'positive', '{"label": "Negative"}', '{"label":[]}',
                    '{"label":"positive","explanation":"good"}', None):
            self.assertIsNone(runner.parse_label(raw))

    def test_retry_fallback_resume_and_order(self):
        args = ["--datasets", self.phrase, "--execute"]
        self.replies = ["bad", '{"label":"positive"}', "invalid", "{}"]
        self.assertEqual(self.run_args(*args, "--limit", "2"), 0)
        directory = self.out / self.phrase
        raw = directory / "raw.jsonl"
        rows = {r["id"]: r for r in map(json.loads, raw.read_text().splitlines())}
        first, second, _ = self.examples[self.phrase]
        self.assertEqual((rows[first]["label"], rows[first]["attempts"], rows[first]["valid"]),
                         ("positive", 2, True))
        self.assertEqual((rows[second]["label"], rows[second]["attempts"], rows[second]["valid"]),
                         ("neutral", 2, False))
        self.assertEqual(len(self.requests), 4)
        with raw.open("ab") as handle:
            handle.write(b'{"id":"torn')
        self.assertEqual(self.run_args(*args), 0)
        self.assertEqual(self.run_args(*args), 0)
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(len(raw.read_text().splitlines()), 3)
        with np.load(directory / "predictions.npz") as archive:
            self.assertEqual(set(archive.files), {"test_ids", "test_y", "test_prob"})
            np.testing.assert_array_equal(archive["test_ids"], self.examples[self.phrase])
            np.testing.assert_array_equal(archive["test_y"], [0, 1, 2])
            np.testing.assert_array_equal(archive["test_prob"], np.eye(3)[[2, 1, 2]])
            self.assertTrue(np.issubdtype(archive["test_prob"].dtype, np.floating))
        result = json.loads((directory / "result.json").read_text())
        self.assertEqual((result["n"], result["n_invalid"], result["digest"]), (3, 1, self.digest))
        self.assertEqual(result["contamination_note"], runner.NOTE)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 2 / 3)
        call = self.requests[0]
        self.assertEqual(call["system"], result["prompt"]["system"])
        self.assertEqual(call["prompt"], result["prompt"]["user"].format(sentence=f"Sentence {self.phrase} 0"))
        self.assertEqual(call["options"], {"seed": 42, "temperature": 0, "num_predict": 16})
        self.assertEqual(call["format"], runner.SCHEMA)
        self.digest = "different-model-weights"
        self.assertEqual(self.run_args(*args), 2)
        self.assertEqual(len(self.requests), 5)

    def make_reference_runs(self):
        for name, ids in self.examples.items():
            extension = name in ("fiqa", "twitter")
            split = name if extension else ("test" if name in runner.PHRASEBANK else "external")
            base = self.root / ("extensions" if extension else "step2")
            config = runner.PHRASEBANK.get(name, "sentences_allagree")
            for model in summary.MODELS:
                for seed in (42, 100):
                    directory = base / config / model / f"seed{seed}"
                    directory.mkdir(parents=True, exist_ok=True)
                    # Step 2 stores test and external together, and extension runs contain both extra sets.
                    path = directory / "predictions.npz"
                    arrays = {}
                    if path.exists():
                        with np.load(path) as archive:
                            arrays.update(archive)
                    pred = [0, 1, 2] if seed == 42 else [1, 2, 0]
                    arrays.update({split + "_ids": ids, split + "_y": [0, 1, 2],
                                   split + "_prob": np.eye(3)[pred]})
                    np.savez(path, **arrays)

    def summarize(self, *args):
        return summary.main(["--out", str(self.out), "--bootstrap", "40",
                             "--step2-dir", str(self.root / "step2"),
                             "--extensions-dir", str(self.root / "extensions"), *args])

    def test_paired_summary_and_check_across_five_datasets(self):
        self.assertEqual(self.run_args("--execute"), 0)
        self.make_reference_runs()
        self.assertEqual(self.summarize(), 0)
        path = self.out / "summary.json"
        result = json.loads(path.read_text())
        self.assertEqual(set(result["datasets"]), set(runner.DATASETS))
        for entry in result["datasets"].values():
            self.assertEqual(entry["metrics"]["macro_f1"], 1.0)
            for comp in entry["comparisons"].values():
                self.assertEqual(comp["bootstrap_macro_f1"]["delta"], 0.5)
                self.assertEqual(comp["bootstrap_macro_f1"]["method"], "seed+sentence")
                self.assertEqual(comp["mcnemar_by_seed"]["100"]["p_value"], 0.25)
                self.assertEqual(comp["mcnemar_by_seed"]["100"]["p_holm"], 0.5)
        self.assertIn(runner.NOTE, (self.out / "REPORT.md").read_text())
        mtime = path.stat().st_mtime_ns
        self.assertEqual(self.summarize("--check"), 0)
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        path.write_text("{}")
        self.assertEqual(self.summarize("--check"), 1)
        self.assertEqual(path.read_text(), "{}")

    def test_refuse_id_order_or_gold_mismatch_before_writing(self):
        self.assertEqual(self.run_args("--datasets", self.phrase, "--execute"), 0)
        self.make_reference_runs()
        path = self.root / "step2/sentences_allagree/bert_base/seed42/predictions.npz"
        ids = self.examples[self.phrase]
        for bad_ids, gold, message in ((ids[::-1], [0, 1, 2], "id/order mismatch"),
                                       (ids, [1, 0, 2], "gold label mismatch")):
            np.savez(path, test_ids=bad_ids, test_y=gold, test_prob=np.eye(3))
            self.assertEqual(self.summarize("--datasets", self.phrase), 2)
            self.assertIn(message, self.stderr.getvalue())
            self.assertFalse((self.out / "summary.json").exists())
            self.assertFalse((self.out / "REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
