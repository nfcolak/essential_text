"""Prepare two additional external test sets: FiQA 2018 Task 1 and Twitter Financial News.

Both are evaluation-only (no model is trained or tuned on them). Files are downloaded from pinned Hugging Face
revisions and checked against SHA-256 sums; nothing is downloaded without ``--execute``.

- FiQA 2018 Task 1 (``pauri32/fiqa-2018``; train + validation + test, 1,213 aspect rows from financial headlines and
  microblog posts). The 3-class label of the release is used: sentiment score > 0.1 positive, < -0.1 negative,
  otherwise neutral (verified on the rows: label 0 = 0.101..0.975, 1 = -0.1..0.1, 2 = -0.938..-0.105). A sentence
  with several aspect rows is kept once if all its rows share the label and dropped otherwise (as for SEntFiN).
- Twitter Financial News (``zeroshot/twitter-financial-news-sentiment``, MIT; train + valid, 11,931 tweets).
  Bearish -> negative, bullish -> positive, neutral -> neutral. URLs (``https://t.co/...``) are removed, because the
  models were trained on sentences without URLs; the cashtags stay. Bearish/bullish is a market-outlook label, which
  is close to, but not the same as, the PhraseBank sentiment label.

Leakage control: rows whose normalised text duplicates an earlier row are dropped, and every row with an exact
(normalised SHA-256) or near duplicate (token Jaccard >= 0.80) in Financial PhraseBank (all 4,846 sentences, the
training source of every model, incl. ``ProsusAI/finbert``) is removed, using ``deduplicate_external_holdout``.

Writes ``<splits-dir>/extra_external.json`` (git-ignored like the other split files; its SHA-256 is appended to
``SHA256SUMS``) and prints the counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dedup import deduplicate_external_holdout, sentence_sha256  # noqa: E402
from src.data.io import file_sha256  # noqa: E402

LABEL_NAMES = ("negative", "neutral", "positive")
OUT_FILE = "extra_external.json"

FIQA_REPO, FIQA_REV = "pauri32/fiqa-2018", "74b7dd5d4fc61e0fa5d979bef2c53111ea69c58a"
TWITTER_REPO, TWITTER_REV = "zeroshot/twitter-financial-news-sentiment", "ccbe24de388e287beb92dd393a335c376b350ac3"
FILES = {
    # local name: (repo, revision, file, sha256)
    "fiqa_train.csv": (FIQA_REPO, FIQA_REV, "train.csv",
                       "8789e9d4962916aa434e5842866d836e337dd5fd0e879fd25f9f7640fb87d8c2"),
    "fiqa_validation.csv": (FIQA_REPO, FIQA_REV, "validation.csv",
                            "87fe6cf8c0cfacaf3cd82fdf729ef5283284995f8831e7919eb947db233f8f24"),
    "fiqa_test.csv": (FIQA_REPO, FIQA_REV, "test.csv",
                      "751d420cec6baee971675d8b70fe4b99e780e3f6e2e21416b3d676895ab506c5"),
    "twitter_sent_train.csv": (TWITTER_REPO, TWITTER_REV, "sent_train.csv",
                               "71aa99d0d37feced03c999ec23b5b25302c92f39d69b6acb0a37aa076c5420e8"),
    "twitter_sent_valid.csv": (TWITTER_REPO, TWITTER_REV, "sent_valid.csv",
                               "31ea8274c98b52212daa68f2fb962811bf1913c1efe51b99765c8e2311806de6"),
}
# release label -> project label (0 negative, 1 neutral, 2 positive)
FIQA_LABELS = {"0": 2, "1": 1, "2": 0}
TWITTER_LABELS = {"0": 0, "1": 2, "2": 1}
_URL = re.compile(r"https?://\S+")


def _url(repo: str, rev: str, name: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"


def download(raw_dir: str) -> dict[str, dict[str, str]]:
    os.makedirs(raw_dir, exist_ok=True)
    provenance = {}
    for local, (repo, rev, name, expected) in FILES.items():
        path = os.path.join(raw_dir, local)
        if not os.path.isfile(path) or file_sha256(path) != expected:
            urllib.request.urlretrieve(_url(repo, rev, name), path)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {local}: {actual} != {expected}")
        provenance[local] = {"repo": repo, "revision": rev, "file": name, "sha256": actual}
    return provenance


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_fiqa(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One record per sentence; sentences whose aspect rows disagree on the label are dropped."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        text = (row.get("sentence") or "").strip()
        if not text:
            continue
        if text not in groups:
            order.append(text)
        groups[text].append(row)
    records, conflicting = [], 0
    for text in order:
        labels = {FIQA_LABELS[r["label"]] for r in groups[text]}
        if len(labels) != 1:
            conflicting += 1
            continue
        first = groups[text][0]
        records.append({"text": text, "label": labels.pop(), "format": first.get("format") or "",
                        "release_split": first.get("_split", "")})
    return records, {"rows": len(rows), "sentences": len(order), "conflicting_label_dropped": conflicting}


def parse_twitter(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records, empty = [], 0
    for row in rows:
        text = re.sub(r"\s+", " ", _URL.sub(" ", row.get("text") or "")).strip()
        if not text:
            empty += 1
            continue
        records.append({"text": text, "label": TWITTER_LABELS[row["label"]], "format": "tweet",
                        "release_split": row.get("_split", "")})
    return records, {"rows": len(rows), "empty_after_url_removal": empty}


def drop_internal_duplicates(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Keep the first record per normalised text; drop every copy of a text whose copies disagree on the label."""
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_hash[sentence_sha256(r["text"])].append(r)
    kept, dup, conflict = [], 0, 0
    seen = set()
    for r in records:
        h = sentence_sha256(r["text"])
        if h in seen:
            continue
        seen.add(h)
        copies = by_hash[h]
        if len({c["label"] for c in copies}) > 1:
            conflict += len(copies)
            continue
        dup += len(copies) - 1
        kept.append(dict(r, sha256=h))
    return kept, dup, conflict


def build(raw_dir: str, phrasebank_texts: list[str]) -> dict[str, Any]:
    fiqa_rows = []
    for split in ("train", "validation", "test"):
        for row in _read_csv(os.path.join(raw_dir, f"fiqa_{split}.csv")):
            row["_split"] = split
            fiqa_rows.append(row)
    tw_rows = []
    for split in ("train", "valid"):
        for row in _read_csv(os.path.join(raw_dir, f"twitter_sent_{split}.csv")):
            row["_split"] = split
            tw_rows.append(row)

    out: dict[str, Any] = {"datasets": {}, "stats": {}}
    for name, (records, stats) in {"fiqa": parse_fiqa(fiqa_rows), "twitter": parse_twitter(tw_rows)}.items():
        records, dup, conflict = drop_internal_duplicates(records)
        stats["internal_duplicates_dropped"] = dup
        stats["internal_conflicting_label_dropped"] = conflict
        kept, overlap = deduplicate_external_holdout(records, phrasebank_texts, threshold=0.80)
        stats["phrasebank_overlap_removed"] = overlap["removed_overlap"]
        stats["kept"] = len(kept)
        for k, r in enumerate(kept):
            r["id"] = f"{name}_{k:05d}"
            r["label_name"] = LABEL_NAMES[r["label"]]
        stats["label_counts"] = dict(Counter(r["label_name"] for r in kept))
        stats["format_counts"] = dict(Counter(r["format"] for r in kept))
        out["datasets"][name] = {"records": kept}
        out["stats"][name] = stats
    return out


def load_extra_external(path: str, name: str) -> tuple[list[str], list[str], list[int]]:
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)["datasets"][name]["records"]
    return [r["id"] for r in records], [r["text"] for r in records], [int(r["label"]) for r in records]


def append_sha256sum(splits_dir: str, path: str) -> None:
    sums = os.path.join(splits_dir, "SHA256SUMS")
    name = os.path.basename(path)
    lines = []
    if os.path.isfile(sums):
        with open(sums, encoding="utf-8") as handle:
            lines = [ln for ln in handle.read().splitlines() if ln.strip() and not ln.rstrip().endswith(name)]
    lines.append(f"{file_sha256(path)}  {name}")
    with open(sums, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw", "extra_external"))
    parser.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    parser.add_argument("--execute", action="store_true", help="download (if needed) and write the file")
    args = parser.parse_args(argv)
    if not args.execute:
        for local, (repo, rev, name, _) in FILES.items():
            print(f"would fetch {_url(repo, rev, name)} -> {os.path.join(args.raw_dir, local)}")
        print("Plan only; pass --execute.")
        return 0
    provenance = download(args.raw_dir)
    with open(os.path.join(args.splits_dir, "sentences_50agree_manifest.json"), encoding="utf-8") as handle:
        phrasebank = [r["text"] for r in json.load(handle)["records_by_id"].values()]
    if len(phrasebank) != 4846:
        raise RuntimeError(f"expected all 4,846 PhraseBank sentences in the 50agree manifest, got {len(phrasebank)}")
    payload = build(args.raw_dir, phrasebank)
    payload["provenance"] = provenance
    payload["label_map"] = {"fiqa": "release label 0 (score > 0.1) positive, 1 neutral, 2 (score < -0.1) negative",
                            "twitter": "0 bearish -> negative, 1 bullish -> positive, 2 neutral -> neutral; URLs removed"}
    path = os.path.join(args.splits_dir, OUT_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    append_sha256sum(args.splits_dir, path)
    print(json.dumps(payload["stats"], indent=2))
    print(f"Wrote {path} ({hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
