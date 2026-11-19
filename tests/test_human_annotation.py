"""Tests for the human-vs-LLM annotator agreement script (scripts/analyze_human_annotation.py)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest

from scripts import analyze_human_annotation as aha
from src.analysis.attribution import PHENOMENA, read_annotations

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED_JSON = os.path.join(REPO_ROOT, "results", "step3", "human_annotation_agreement.json")
HEADER = "diag_id," + ",".join(PHENOMENA)


def _tags(rows: dict[str, tuple[int, ...]]) -> dict[str, dict[str, int]]:
    return {diag_id: dict(zip(PHENOMENA, values)) for diag_id, values in rows.items()}


def _varied() -> dict[str, dict[str, int]]:
    """Six items; every phenomenon column has both values."""
    return _tags({
        "d1": (1, 0, 1, 0, 1), "d2": (0, 1, 0, 1, 0), "d3": (1, 1, 0, 0, 1),
        "d4": (0, 0, 1, 1, 0), "d5": (1, 0, 0, 1, 1), "d6": (0, 1, 1, 0, 0),
    })


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = aha.main(argv)
    return code, out.getvalue(), err.getvalue()


def _write_csv(path: str, tags: dict, crlf: bool = False, final_newline: bool = True) -> None:
    eol = "\r\n" if crlf else "\n"
    lines = [HEADER + ",note"] + [
        diag_id + "," + ",".join(str(row[p]) for p in PHENOMENA) + "," for diag_id, row in tags.items()]
    text = eol.join(lines) + (eol if final_newline else "")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _write_dir(folder: str, a: dict, b: dict, h: dict, rule: dict) -> None:
    _write_csv(os.path.join(folder, "annotations_a.csv"), a)
    _write_csv(os.path.join(folder, "annotations_b.csv"), b)
    _write_csv(os.path.join(folder, aha.HUMAN_FILE), h, crlf=True, final_newline=False)
    _write_csv(os.path.join(folder, "rule_tags.csv"), rule, crlf=True)
    with open(os.path.join(folder, "annotators.json"), "w", encoding="utf-8") as handle:
        json.dump({"a": "model a", "b": "model b", "c": "human"}, handle)


class ComputeTest(unittest.TestCase):
    def test_perfect_agreement_gives_kappa_one(self) -> None:
        tags = _varied()
        result = aha.compute(tags, tags, tags, tags)
        self.assertEqual(result["n_items"], 6)
        for name in PHENOMENA:
            row = result["agreement"][name]
            self.assertEqual(row["n"], 6)
            for pair in aha.PAIRS:
                self.assertAlmostEqual(row[f"kappa_{pair}"], 1.0)
                self.assertEqual(row[f"raw_agreement_{pair}"], 1.0)
            self.assertAlmostEqual(row["kappa_ab"], 1.0)
            self.assertEqual(row["disagreements_h_consensus"], [])
            self.assertEqual(row["prevalence_h"], 0.5)
        self.assertEqual(set(result["mean_kappa"]), set(aha.PAIRS) | {"ab"})
        for value in result["mean_kappa"].values():
            self.assertAlmostEqual(value, 1.0)

    def test_constant_column_gives_none_and_is_skipped_in_mean(self) -> None:
        tags = _varied()
        for row in tags.values():
            row["negation"] = 0
        result = aha.compute(tags, tags, tags, tags)
        row = result["agreement"]["negation"]
        for pair in aha.PAIRS:
            self.assertIsNone(row[f"kappa_{pair}"])
            self.assertEqual(row[f"raw_agreement_{pair}"], 1.0)
        self.assertIsNone(row["kappa_ab"])
        self.assertEqual(row["prevalence_h"], 0.0)
        # the mean runs over the four defined phenomena only
        self.assertAlmostEqual(result["mean_kappa"]["h_a"], 1.0)
        self.assertAlmostEqual(result["mean_kappa"]["ab"], 1.0)

    def test_all_columns_constant_gives_none_mean(self) -> None:
        tags = _tags({"d1": (0, 0, 0, 0, 0), "d2": (0, 0, 0, 0, 0)})
        result = aha.compute(tags, tags, tags, tags)
        self.assertTrue(all(v is None for v in result["mean_kappa"].values()))

    def test_one_side_constant_is_zero_not_none(self) -> None:
        human = _varied()
        other = _varied()
        for row in other.values():
            row["hedging"] = 0
        result = aha.compute(other, other, human, other)
        self.assertAlmostEqual(result["agreement"]["hedging"]["kappa_h_a"], 0.0)

    def test_id_mismatch_raises(self) -> None:
        tags = _varied()
        short = {k: v for k, v in tags.items() if k != "d6"}
        extra = dict(tags, d7=dict(zip(PHENOMENA, (0, 0, 0, 0, 0))))
        for bad_position in range(4):
            for bad in (short, extra):
                args = [tags, tags, tags, tags]
                args[bad_position] = bad
                with self.assertRaises(ValueError):
                    aha.compute(*args)
        with self.assertRaises(ValueError):
            aha.compute({}, {}, {}, {})

    def test_consensus_needs_both_annotators(self) -> None:
        # column 0 (financial_terminology): a=1,1,0,0  b=1,0,1,0  -> consensus 1,0,0,0
        a = _tags({"d1": (1, 1, 0, 1, 0), "d2": (1, 0, 1, 0, 1), "d3": (0, 1, 0, 1, 0), "d4": (0, 0, 1, 0, 1)})
        b = _tags({"d1": (1, 1, 0, 1, 0), "d2": (0, 0, 1, 0, 1), "d3": (1, 1, 0, 1, 0), "d4": (0, 0, 1, 0, 1)})
        h = _tags({"d1": (1, 1, 0, 1, 0), "d2": (1, 0, 1, 0, 1), "d3": (0, 1, 0, 1, 0), "d4": (0, 0, 1, 0, 1)})
        row = aha.compute(a, b, h, h)["agreement"]["financial_terminology"]
        self.assertEqual(row["prevalence_consensus"], 0.25)
        self.assertEqual(row["prevalence_a"], 0.5)
        self.assertEqual(row["prevalence_b"], 0.5)
        # h = 1,1,0,0 differs from the consensus 1,0,0,0 only on d2
        self.assertEqual(row["disagreements_h_consensus"], ["d2"])
        self.assertEqual(row["raw_agreement_h_consensus"], 0.75)
        self.assertAlmostEqual(row["kappa_h_a"], 1.0)
        self.assertAlmostEqual(row["kappa_h_b"], 0.0)
        self.assertAlmostEqual(row["kappa_h_consensus"], 0.5)

    def test_kappa_matches_hand_value(self) -> None:
        # 2x2 table: both 1 = 2, h only = 1, a only = 1, both 0 = 4 -> po = 0.75, pe = 0.53125
        h_col = [1, 1, 1, 0, 0, 0, 0, 0]
        a_col = [1, 1, 0, 1, 0, 0, 0, 0]
        ids = [f"d{i}" for i in range(8)]
        h = _tags({i: (x, x, x, x, x) for i, x in zip(ids, h_col)})
        a = _tags({i: (x, x, x, x, x) for i, x in zip(ids, a_col)})
        row = aha.compute(a, a, h, a)["agreement"]["negation"]
        self.assertAlmostEqual(row["kappa_h_a"], (0.75 - 0.53125) / (1 - 0.53125))
        self.assertEqual(row["disagreements_h_consensus"], ["d2", "d3"])

    def test_second_human_agreement_and_consensus(self) -> None:
        h1 = _varied()
        h2 = _varied()
        h2["d1"]["negation"] = 1
        h2["d2"]["negation"] = 0
        result = aha.compute(h1, h2, h1, h1, h2)
        row = result["agreement"]["negation"]
        self.assertAlmostEqual(row["kappa_h1_h2"], 1 / 3)
        self.assertAlmostEqual(row["kappa_h2_a"], 1 / 3)
        self.assertAlmostEqual(row["kappa_h2_b"], 1.0)
        self.assertAlmostEqual(row["kappa_h2_consensus"], 2 / 3)
        self.assertAlmostEqual(row["kappa_human_consensus_vs_llm_consensus"], 1.0)
        self.assertEqual(row["human_consensus_ids"], ["d3", "d6"])
        self.assertAlmostEqual(row["prevalence_human_consensus"], 2 / 6)
        self.assertAlmostEqual(result["mean_kappa"]["h1_h2"], (4 + 1 / 3) / 5)
        self.assertEqual(set(result["mean_kappa"]), set(aha.PAIRS + aha.HUMAN2_PAIRS) | {"ab"})


class FileTest(unittest.TestCase):
    def test_crlf_without_trailing_newline_is_parsed(self) -> None:
        tags = _varied()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, aha.HUMAN_FILE)
            _write_csv(path, tags, crlf=True, final_newline=False)
            with open(path, "rb") as handle:
                raw = handle.read()
            self.assertIn(b"\r\n", raw)
            self.assertFalse(raw.endswith(b"\n"))
            self.assertEqual(read_annotations(path), tags)

    def test_cli_write_then_check(self) -> None:
        tags = _varied()
        other = _varied()
        other["d1"]["hedging"] = 0
        with tempfile.TemporaryDirectory() as folder:
            _write_dir(folder, other, tags, tags, other)
            out = os.path.join(folder, "out", "agreement.json")
            argv = ["--diagnostic-dir", folder, "--out", out]
            code, stdout, _ = _run(argv)
            self.assertEqual(code, 0)
            self.assertIn("mean kappa", stdout)
            with open(out, encoding="utf-8") as handle:
                text = handle.read()
            self.assertTrue(text.endswith("}\n"))
            payload = json.loads(text)
            self.assertEqual(text, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            self.assertEqual(set(payload), {"annotators", "n_items", "agreement", "mean_kappa", "note"})
            self.assertEqual(payload["annotators"], {"a": "model a", "b": "model b", "c": "human"})
            self.assertEqual(payload["n_items"], 6)
            self.assertEqual(payload["note"], aha.NOTE)
            self.assertEqual(payload["agreement"]["hedging"]["disagreements_h_consensus"], ["d1"])

            self.assertEqual(_run(argv + ["--check"])[0], 0)
            before = os.path.getmtime(out)

            # a changed input makes the check fail, and the check never rewrites the file
            other["d2"]["negation"] = 0
            _write_csv(os.path.join(folder, "annotations_a.csv"), other)
            code, _, stderr = _run(argv + ["--check"])
            self.assertEqual(code, 1)
            self.assertIn("CHECK FAILED", stderr)
            self.assertEqual(os.path.getmtime(out), before)
            with open(out, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), text)

    def test_check_tolerance_and_missing_file(self) -> None:
        tags = _varied()
        with tempfile.TemporaryDirectory() as folder:
            _write_dir(folder, tags, tags, tags, tags)
            out = os.path.join(folder, "agreement.json")
            argv = ["--diagnostic-dir", folder, "--out", out]
            self.assertEqual(_run(argv + ["--check"])[0], 1)  # nothing written yet
            self.assertFalse(os.path.exists(out))
            self.assertEqual(_run(argv)[0], 0)
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["mean_kappa"]["h_a"] += 1e-14
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            self.assertEqual(_run(argv + ["--check"])[0], 0)
            payload["mean_kappa"]["h_a"] += 1e-9
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            self.assertEqual(_run(argv + ["--check"])[0], 1)

    def test_id_mismatch_in_files_is_an_error(self) -> None:
        tags = _varied()
        short = {k: v for k, v in tags.items() if k != "d6"}
        with tempfile.TemporaryDirectory() as folder:
            _write_dir(folder, tags, tags, short, tags)
            out = os.path.join(folder, "agreement.json")
            code, _, stderr = _run(["--diagnostic-dir", folder, "--out", out])
            self.assertNotEqual(code, 0)
            self.assertIn("do not match", stderr)
            self.assertFalse(os.path.exists(out))

    def test_second_human_default_and_explicit_override(self) -> None:
        tags = _varied()
        alternate = _varied()
        alternate["d2"]["negation"] = 0
        with tempfile.TemporaryDirectory() as folder:
            _write_dir(folder, tags, tags, tags, tags)
            metadata = os.path.join(folder, "annotators.json")
            with open(metadata, "w", encoding="utf-8") as handle:
                json.dump({"a": "model a", "b": "model b", "c": "human", "d": "second human"}, handle)
            _write_csv(os.path.join(folder, aha.HUMAN2_FILE), tags)
            override = os.path.join(folder, "override.csv")
            _write_csv(override, alternate)
            out = os.path.join(folder, "agreement.json")
            argv = ["--diagnostic-dir", folder, "--out", out]
            code, stdout, stderr = _run(argv)
            self.assertEqual(code, 0, stderr)
            self.assertIn("k h1-h2", stdout)
            self.assertEqual(_run(argv + ["--check"])[0], 0)
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["annotators"]["d"], "second human")
            self.assertAlmostEqual(payload["mean_kappa"]["h1_h2"], 1.0)
            self.assertEqual(_run(argv + ["--human2", override])[0], 0)
            with open(out, encoding="utf-8") as handle:
                self.assertLess(json.load(handle)["mean_kappa"]["h1_h2"], 1.0)

    def test_second_human_with_incomplete_ids_is_refused(self) -> None:
        tags = _varied()
        short = {key: value for key, value in tags.items() if key != "d6"}
        with tempfile.TemporaryDirectory() as folder:
            _write_dir(folder, tags, tags, tags, tags)
            path = os.path.join(folder, aha.HUMAN2_FILE)
            _write_csv(path, short)
            out = os.path.join(folder, "agreement.json")
            code, _, stderr = _run(["--diagnostic-dir", folder, "--out", out])
            self.assertEqual(code, 2)
            self.assertIn("diag_ids of 'h2' do not match", stderr)
            self.assertFalse(os.path.exists(out))


class CommittedFileTest(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(COMMITTED_JSON), "human_annotation_agreement.json not present")
    def test_committed_json_passes_check(self) -> None:
        code, _, stderr = _run(["--check"])
        self.assertEqual(code, 0, stderr)
        with open(COMMITTED_JSON, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["n_items"], 300)
        self.assertIn("c", payload["annotators"])
        self.assertEqual(set(payload["agreement"]), set(PHENOMENA))

    def test_second_human_template_has_all_ids_and_blank_cells(self) -> None:
        folder = os.path.join(REPO_ROOT, "data", "diagnostic")
        with open(os.path.join(folder, "diagnostic_items.csv"), encoding="utf-8", newline="") as handle:
            ids = [row["diag_id"] for row in csv.DictReader(handle)]
        path = os.path.join(folder, "annotations_human2_template.csv")
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            self.assertEqual(next(reader), HEADER.split(",") + ["note"])
            self.assertEqual(len(ids), 300)
            self.assertEqual(list(reader), [[diag_id, "", "", "", "", "", ""] for diag_id in ids])


if __name__ == "__main__":
    unittest.main()
