"""Offline tests for the full-split diagnostic builder and analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import analyze_diagnostic_extension as analysis
from scripts import build_diagnostic_extension as builder
from scripts.run_step3_analysis import MODELS
from src.analysis.phenomena import PHENOMENA


class DiagnosticExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.diagnostic = self.root / "diagnostic"
        self.diagnostic.mkdir()
        self.extension = self.diagnostic / "extension"
        self.out = self.root / "output"
        self.ids = ["r0", "r1", "r2", "r3"]
        self.labels = [0, 0, 1, 2]
        texts = ["Costs did not fall.", "Debt doubled.", "No change.", 'Sales, "very strong".']
        manifest = {
            "splits": {"train": ["train"], "test": self.ids},
            "records_by_id": {
                **{record_id: {"text": text, "label": label}
                   for record_id, text, label in zip(self.ids, texts, self.labels)},
                "train": {"text": "A training record.", "label": 1},
            },
        }
        path = self.root / f"{builder.CONFIG}_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.records, self.manifest_hash = builder.load_test(self.root)
        self.original_key = {
            "config": builder.CONFIG,
            "seed": 42,
            "n": 2,
            "source_split": "test",
            "manifest_sha256": self.manifest_hash,
            "items": [{"diag_id": "d001", "record_id": "r0", "label": 0},
                      {"diag_id": "d002", "record_id": "r1", "label": 0}],
        }
        (self.diagnostic / "diagnostic_key.json").write_text(json.dumps(self.original_key),
                                                             encoding="utf-8")
        self.args = ["--diagnostic-dir", str(self.diagnostic), "--splits-dir", str(self.root),
                     "--out-dir", str(self.out), "--runs-dir", str(self.root / "runs"),
                     "--llm-dir", str(self.root / "llm"), "--bootstrap-samples", "16"]

    def _files(self) -> dict[str, bytes]:
        return builder.build_files(self.records, self.manifest_hash, self.original_key,
                                   expected_original=2, expected_extension=2)

    def _analysis_main(self, *args: str) -> int:
        with patch.multiple(analysis, N_ORIGINAL=2, N_EXTENSION=2):
            return analysis.main(self.args + list(args))

    def test_selection_is_deterministic_and_blind(self) -> None:
        files = self._files()
        reversed_records = dict(reversed(list(self.records.items())))
        self.assertEqual(files, builder.build_files(
            reversed_records, self.manifest_hash, self.original_key,
            expected_original=2, expected_extension=2))
        key = json.loads(files["key.json"])
        self.assertEqual(key["seed"], 2026)
        self.assertEqual(key["manifest_sha256"], self.manifest_hash)
        self.assertEqual({item["record_id"] for item in key["items"]}, {"r2", "r3"})
        rows = list(csv.reader(io.StringIO(files["items.csv"].decode())))
        self.assertEqual(rows[0], ["ext_id", "text"])
        self.assertEqual([row[0] for row in rows[1:]], ["e001", "e002"])
        for (ext_id, text), item in zip(rows[1:], key["items"]):
            self.assertEqual(ext_id, item["ext_id"])
            self.assertEqual((text, item["label"]), self.records[item["record_id"]])
            self.assertEqual(item["sha256"], hashlib.sha256(text.encode()).hexdigest())
        rules = list(csv.reader(io.StringIO(files["rule_tags.csv"].decode())))
        self.assertEqual(rules[0], ["diag_id", *PHENOMENA])
        tags = {row[0]: row[1:] for row in rules[1:]}
        for ext_id, text in rows[1:]:
            self.assertEqual(tags[ext_id][PHENOMENA.index("negation")], str(int(text == "No change.")))

    def test_wrong_remaining_count_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 3 remaining test records, found 2"):
            builder.build_files(self.records, self.manifest_hash, self.original_key,
                                expected_original=2, expected_extension=3)

    def _write_annotations(self) -> None:
        self.extension.mkdir()
        for name, content in self._files().items():
            (self.extension / name).write_bytes(content)
        ext_key = json.loads((self.extension / "key.json").read_bytes())
        original_ids = [item["diag_id"] for item in self.original_key["items"]]
        extension_ids = {item["record_id"]: item["ext_id"] for item in ext_key["items"]}
        for directory, ids in ((self.diagnostic, original_ids),
                               (self.extension, [extension_ids["r2"], extension_ids["r3"]])):
            for name in analysis.ANNOTATION_FILES:
                if name == "rule_tags.csv" and directory == self.extension:
                    continue
                second = int(directory == self.diagnostic or name == "annotations_a.csv")
                with (directory / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["diag_id", *PHENOMENA])
                    writer.writerow([ids[0], 1, 0, 0, 0, 0])
                    writer.writerow([ids[1], second, 0, 0, 0, 0])

    def _write_predictions(self, path: Path, predictions: list[int]) -> None:
        path.mkdir(parents=True)
        np.savez(path / "predictions.npz", test_ids=np.array(self.ids), test_y=np.array(self.labels),
                 test_prob=np.eye(3)[predictions])

    def test_analysis_uses_union_consensus_and_step3_runs(self) -> None:
        self._write_annotations()
        for model in MODELS:
            path = self.root / "runs" / builder.CONFIG / model / "seed42"
            predictions = {"bert_base": [1, 1, 1, 2], "tfidf_svm": [1, 1, 0, 2]}
            self._write_predictions(path, predictions.get(model, self.labels))
            score = 0.7 if model == "tfidf_svm" else 0.6
            (path / "result.json").write_text(json.dumps({
                "status": "completed", "metrics": {"val": {"macro_f1": score}},
            }), encoding="utf-8")
        self._write_predictions(self.root / "llm", self.labels)

        with redirect_stdout(io.StringIO()):
            self.assertEqual(self._analysis_main(), 0)
            self.assertEqual(self._analysis_main("--check"), 0)
        payload = json.loads((self.out / "phenomenon_analysis.json").read_bytes())
        self.assertEqual(payload["n"], 4)
        self.assertEqual(payload["tfidf_best"], "tfidf_svm")
        term = payload["error_rates"]["financial_terminology"]
        self.assertEqual(term["n_tag1"], 3)
        self.assertEqual(term["gold_mix_tag1"], {"n_neg": 2, "n_neu": 1, "n_pos": 0})
        self.assertAlmostEqual(term["models"]["bert_base"]["tag1"]["error_rate"], 2 / 3)
        self.assertEqual(term["models"]["bert_base"]["tag1"]["balanced_error"], 0.5)
        self.assertEqual(term["models"]["llm_zeroshot"]["tag1"]["error_rate"], 0)
        deltas = payload["attribution"]["financial_terminology"]
        self.assertAlmostEqual(deltas["delta_arch"]["delta"], 1 / 3)
        self.assertAlmostEqual(deltas["delta_domain"]["delta"], 2 / 3)
        self.assertIsNone(deltas["delta_domain"]["p_value"])
        self.assertIsNone(deltas["delta_domain"]["p_holm"])
        ext_agreement = payload["agreement"]["extension"]["financial_terminology"]
        union_agreement = payload["agreement"]["union"]["financial_terminology"]
        self.assertEqual((ext_agreement["kappa_ab"], ext_agreement["raw_agreement_ab"]), (0, 0.5))
        self.assertEqual((union_agreement["kappa_ab"], union_agreement["raw_agreement_ab"]), (0, 0.75))
        report = self.out / "REPORT.md"
        self.assertIn("Zero-shot LLM", report.read_text(encoding="utf-8"))
        report.write_text("changed", encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(self._analysis_main("--check"), 1)
        self.assertEqual(report.read_text(encoding="utf-8"), "changed")

    def test_missing_annotations_exit_two_even_in_check_mode(self) -> None:
        self._write_annotations()
        (self.extension / "annotations_b.csv").unlink()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(self._analysis_main("--check"), 2)
        self.assertIn(str(self.extension / "annotations_b.csv"), stderr.getvalue())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
