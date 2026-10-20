"""Tests for the phenomenon pre-tagger and the diagnostic subset builder (synthetic data only)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest

from scripts import build_diagnostic_set as bds
from src.analysis.phenomena import PHENOMENA, context_dependent_v1, financial_terms, tag_many, tag_sentence


class TestTagger(unittest.TestCase):
    def test_keys_equal_phenomena(self):
        self.assertEqual(tuple(tag_sentence("Hello world.").keys()), PHENOMENA)
        self.assertEqual(len(PHENOMENA), 5)

    def test_financial_terminology(self):
        self.assertEqual(tag_sentence("Depreciation of the plant was booked in March.")["financial_terminology"], 1)
        self.assertEqual(tag_sentence("The bakery opened a second shop.")["financial_terminology"], 0)

    def test_negation(self):
        self.assertEqual(tag_sentence("The firm did not meet the schedule.")["negation"], 1)
        self.assertEqual(tag_sentence("The firm couldn't deliver.")["negation"], 1)
        self.assertEqual(tag_sentence("The firm met the schedule.")["negation"], 0)

    def test_hedging(self):
        self.assertEqual(tag_sentence("The board expects demand to pick up.")["hedging"], 1)
        self.assertEqual(tag_sentence("Revenue reached about 40 million.")["hedging"], 1)
        self.assertEqual(tag_sentence("The board approved the budget.")["hedging"], 0)

    def test_numerical_comparison(self):
        self.assertEqual(tag_sentence("Sales rose to 52 million from 47 million.")["numerical_comparison"], 1)
        self.assertEqual(tag_sentence("Sales were 52 million.")["numerical_comparison"], 0)
        self.assertEqual(tag_sentence("Sales were higher than before.")["numerical_comparison"], 0)

    def test_context_dependent(self):
        self.assertEqual(tag_sentence("The deficit narrowed during the second half.")["context_dependent"], 1)
        self.assertEqual(tag_sentence("The board approved the budget during the second half.")["context_dependent"], 0)

    def test_context_dependent_v2(self):
        for text in ("Operating profit decreased to EUR 1.7 mn.", "Operating costs fell.", "The company will cut 80 jobs.",
                     "Net sales rose in the second half.", "The company laid off 40 people."):
            self.assertEqual(tag_sentence(text)["context_dependent"], 1, text)
        self.assertEqual(tag_sentence("Costs are listed in the annex.")["context_dependent"], 0)
        self.assertEqual(context_dependent_v1("Operating profit decreased to EUR 1.7 mn."), 0)
        self.assertEqual(context_dependent_v1("Operating costs fell."), 1)

    def test_up_down_needs_number(self):
        self.assertEqual(tag_sentence("Shares were up 5 percent.")["numerical_comparison"], 1)
        self.assertEqual(tag_sentence("The shares were up on the news.")["numerical_comparison"], 0)
        self.assertEqual(tag_sentence("The start-up company has 12 staff.")["numerical_comparison"], 0)
        self.assertEqual(tag_sentence("Please sign up before 5 May.")["numerical_comparison"], 0)

    def test_no_abbreviation_is_not_negation(self):
        self.assertEqual(tag_sentence("It is the No. 1 supplier.")["negation"], 0)
        self.assertEqual(tag_sentence("There is no supplier.")["negation"], 1)

    def test_charge_role_sense_excluded(self):
        self.assertEqual(tag_sentence("Smith takes charge as CEO.")["financial_terminology"], 0)
        self.assertEqual(tag_sentence("She is in charge of sales.")["financial_terminology"], 0)
        self.assertEqual(tag_sentence("The firm booked a one-off charge.")["financial_terminology"], 1)
        self.assertEqual(financial_terms("Taxes and a restructuring charge."), ["taxes", "restructuring", "charge"])

    def test_context_window_is_four_tokens(self):
        near = "Costs a b c fell."          # distance 4 -> tagged
        far = "Costs a b c d fell."          # distance 5 -> not tagged
        self.assertEqual(tag_sentence(near)["context_dependent"], 1)
        self.assertEqual(tag_sentence(far)["context_dependent"], 0)

    def test_multi_tag(self):
        tags = tag_sentence("The company expects losses not to rise by 5 percent from 40 million.")
        self.assertEqual(tags["financial_terminology"], 1)
        self.assertEqual(tags["negation"], 1)
        self.assertEqual(tags["hedging"], 1)
        self.assertEqual(tags["numerical_comparison"], 1)
        self.assertEqual(tags["context_dependent"], 1)

    def test_tag_many(self):
        texts = ["Costs fell.", "Nothing here."]
        out = tag_many(texts)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], tag_sentence(texts[0]))


def _synthetic_manifest(path: str) -> dict:
    records = {}
    splits = {"train": [], "val": [], "test": []}
    n = 0
    for part, per_class in (("train", (20, 30, 10)), ("val", (5, 5, 5)), ("test", (30, 50, 20))):
        for label, count in enumerate(per_class):
            for k in range(count):
                rid = "r{:04d}".format(n)
                n += 1
                records[rid] = {"id": rid, "text": "Sentence {} of class {} in {}.".format(k, label, part),
                                "label": label, "sha256": "sha{}".format(rid)}
                splits[part].append(rid)
    manifest = {"splits": splits, "cv_folds": [], "records_by_id": records}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = bds.main(argv)
    return code


class TestBuildDiagnosticSet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.splits = os.path.join(self.tmp.name, "splits")
        os.makedirs(self.splits)
        self.manifest = _synthetic_manifest(os.path.join(self.splits, "cfg_manifest.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, out, **kw):
        a = ["--config", "cfg", "--n", "50", "--seed", "7", "--splits-dir", self.splits, "--out-dir", out]
        return a + (["--execute"] if kw.get("execute") else [])

    def test_plan_mode_writes_nothing(self):
        out = os.path.join(self.tmp.name, "diag")
        self.assertEqual(_run(self._args(out)), 0)
        self.assertFalse(os.path.exists(out))

    def test_missing_manifest_exit_2(self):
        out = os.path.join(self.tmp.name, "diag")
        code = _run(["--config", "nope", "--splits-dir", self.splits, "--out-dir", out, "--execute"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))

    def test_execute_outputs(self):
        out = os.path.join(self.tmp.name, "diag")
        self.assertEqual(_run(self._args(out, execute=True)), 0)
        with open(os.path.join(out, "diagnostic_key.json"), encoding="utf-8") as f:
            key = json.load(f)
        test_ids = set(self.manifest["splits"]["test"])
        self.assertEqual(key["n"], 50)
        self.assertEqual(key["source_split"], "test")
        self.assertEqual(len(key["items"]), 50)
        self.assertEqual([i["diag_id"] for i in key["items"]], ["d{:03d}".format(i + 1) for i in range(50)])
        self.assertTrue(all(i["record_id"] in test_ids for i in key["items"]))
        self.assertEqual(len({i["record_id"] for i in key["items"]}), 50)
        counts = {}
        for i in key["items"]:
            counts[i["label"]] = counts.get(i["label"], 0) + 1
        self.assertEqual(counts, {0: 15, 1: 25, 2: 10})   # test shares 30/50/20 of 100 -> n=50
        with open(os.path.join(out, "diagnostic_items.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["diag_id", "text"])
        self.assertEqual(len(rows), 51)
        with open(os.path.join(out, "rule_tags.csv"), encoding="utf-8", newline="") as f:
            tag_rows = list(csv.reader(f))
        self.assertEqual(tag_rows[0], ["diag_id"] + list(PHENOMENA))
        self.assertEqual(len(tag_rows), 51)

    def test_rule_tags_only_keeps_key_and_items(self):
        out = os.path.join(self.tmp.name, "diag")
        self.assertEqual(_run(self._args(out, execute=True)), 0)
        key_path = os.path.join(out, "diagnostic_key.json")
        items_path = os.path.join(out, "diagnostic_items.csv")
        tags_path = os.path.join(out, "rule_tags.csv")
        with open(key_path, "rb") as f:
            key_before = f.read()
        with open(items_path, "rb") as f:
            items_before = f.read()
        with open(tags_path, "w", encoding="utf-8") as f:
            f.write("stale\n")
        self.assertEqual(_run(["--rule-tags-only", "--out-dir", out]), 0)       # plan only
        with open(tags_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "stale\n")
        self.assertEqual(_run(["--rule-tags-only", "--execute", "--out-dir", out]), 0)
        with open(tags_path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["diag_id"] + list(PHENOMENA))
        self.assertEqual(len(rows), 51)
        with open(key_path, "rb") as f:
            self.assertEqual(f.read(), key_before)
        with open(items_path, "rb") as f:
            self.assertEqual(f.read(), items_before)

    def test_rule_tags_only_missing_items_exit_2(self):
        out = os.path.join(self.tmp.name, "empty")
        os.makedirs(out)
        self.assertEqual(_run(["--rule-tags-only", "--execute", "--out-dir", out]), 2)

    def test_sha256sums_mismatch_exit_2_and_match_ok(self):
        out = os.path.join(self.tmp.name, "diag")
        manifest_path = os.path.join(self.splits, "cfg_manifest.json")
        sums = os.path.join(self.splits, "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as f:
            f.write("{}  cfg_manifest.json\n".format("0" * 64))
        self.assertEqual(_run(self._args(out, execute=True)), 2)
        self.assertFalse(os.path.exists(out))
        with open(sums, "w", encoding="utf-8") as f:
            f.write("{}  cfg_manifest.json\n".format(bds._file_sha256(manifest_path)))
        self.assertEqual(_run(self._args(out, execute=True)), 0)

    def test_largest_remainder_sums_to_n(self):
        counts = bds.largest_remainder_counts({0: 1, 1: 1, 2: 1}, 10)
        self.assertEqual(sum(counts.values()), 10)

    def test_reproducible_and_seed_sensitive(self):
        a, b, c = (os.path.join(self.tmp.name, x) for x in ("a", "b", "c"))
        self.assertEqual(_run(self._args(a, execute=True)), 0)
        self.assertEqual(_run(self._args(b, execute=True)), 0)
        with open(os.path.join(a, "diagnostic_key.json"), encoding="utf-8") as f:
            ka = f.read()
        with open(os.path.join(b, "diagnostic_key.json"), encoding="utf-8") as f:
            kb = f.read()
        self.assertEqual(ka, kb)
        args = self._args(c, execute=True)
        args[args.index("--seed") + 1] = "8"
        self.assertEqual(_run(args), 0)
        with open(os.path.join(c, "diagnostic_key.json"), encoding="utf-8") as f:
            kc = f.read()
        self.assertNotEqual(ka, kc)


if __name__ == "__main__":
    unittest.main()
