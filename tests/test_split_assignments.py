from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import export_split_assignments as exp  # noqa: E402
from src.data.assignments import assignments_from_manifest, read_csv, sentfin_rows  # noqa: E402


def _manifest():
    ids = ["s{}".format(i) for i in range(10)]
    recs = {
        i: {"id": i, "text": "secret text " + i, "sha256": "h" + i, "label": n % 3,
            "label_name": "x", "cluster_id": n}
        for n, i in enumerate(ids)
    }
    return {
        "splits": {"train": ids[:6], "val": ids[6:8], "test": ids[8:]},
        "cv_folds": [
            {"fold": 0, "train": ids[3:6], "val": ids[0:3]},
            {"fold": 1, "train": ids[0:3], "val": ids[3:6]},
        ],
        "records_by_id": recs,
    }


def _dedup():
    return {"stats": {}, "records": [
        {"id": "sentfin_2", "text": "t2", "sha256": "b", "label": 2},
        {"id": "sentfin_1", "text": "t1", "sha256": "a", "label": 0},
    ]}


def _write_splits(d, manifest=None, dedup=None, sums=True):
    manifest = manifest or _manifest()
    files = {"sentences_toy_manifest.json": manifest, "sentfin_dedup.json": dedup or _dedup()}
    lines = []
    for name, obj in files.items():
        p = os.path.join(d, name)
        with open(p, "w") as f:
            json.dump(obj, f)
        with open(p, "rb") as f:
            lines.append("{}  {}".format(hashlib.sha256(f.read()).hexdigest(), name))
    if sums:
        with open(os.path.join(d, "SHA256SUMS"), "w") as f:
            f.write("\n".join(lines) + "\n")


class AssignmentFunctions(unittest.TestCase):
    def test_fold_mapping_train_only_and_sorted(self):
        rows = assignments_from_manifest(_manifest())
        self.assertEqual([r["id"] for r in rows], sorted(r["id"] for r in rows))
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["s0"]["cv_fold"], 0)
        self.assertEqual(by_id["s4"]["cv_fold"], 1)
        self.assertEqual(by_id["s6"]["cv_fold"], "")
        self.assertEqual(by_id["s9"]["split"], "test")
        self.assertEqual(by_id["s9"]["cv_fold"], "")

    def test_no_text_column(self):
        rows = assignments_from_manifest(_manifest())
        self.assertTrue(all("text" not in r and "norm_text" not in r for r in rows))
        self.assertEqual(set(sentfin_rows(_dedup())[0]), {"id", "sha256", "label"})

    def test_sentfin_sorted(self):
        self.assertEqual([r["id"] for r in sentfin_rows(_dedup())], ["sentfin_1", "sentfin_2"])


class ExportCli(unittest.TestCase):
    def _run(self, d, *extra):
        return exp.main(["--splits-dir", d, "--out-dir", os.path.join(d, "assignments")] + list(extra))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_splits(d)
            self.assertEqual(self._run(d), 0)
            self.assertFalse(os.path.exists(os.path.join(d, "assignments")))

    def test_write_files_and_no_text(self):
        with tempfile.TemporaryDirectory() as d:
            _write_splits(d)
            self.assertEqual(self._run(d, "--execute"), 0)
            out = os.path.join(d, "assignments")
            self.assertEqual(sorted(os.listdir(out)), ["README.md", "sentfin_holdout.csv", "toy.csv"])
            with open(os.path.join(out, "toy.csv")) as f:
                text = f.read()
            self.assertTrue(text.startswith("id,sha256,label,cluster_id,split,cv_fold\n"))
            self.assertNotIn("secret", text)
            self.assertEqual(len(read_csv(os.path.join(out, "toy.csv"))), 10)
            self.assertEqual(self._run(d, "--check"), 0)

    def test_check_detects_changed_row(self):
        with tempfile.TemporaryDirectory() as d:
            _write_splits(d)
            self._run(d, "--execute")
            path = os.path.join(d, "assignments", "toy.csv")
            with open(path) as f:
                content = f.read()
            with open(path, "w") as f:
                f.write(content.replace(",train,", ",test,", 1))
            self.assertEqual(self._run(d, "--check"), 1)

    def test_check_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_splits(d)
            self.assertEqual(self._run(d, "--check"), 1)

    def test_sha_mismatch_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            _write_splits(d)
            m = _manifest()
            m["records_by_id"]["s0"]["label"] = 2
            with open(os.path.join(d, "sentences_toy_manifest.json"), "w") as f:
                json.dump(m, f)
            for extra in ((), ("--execute",), ("--check",)):
                self.assertEqual(self._run(d, *extra), 2)
            self.assertFalse(os.path.exists(os.path.join(d, "assignments")))


if __name__ == "__main__":
    unittest.main()
