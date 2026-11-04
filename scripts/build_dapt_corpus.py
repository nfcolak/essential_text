"""Build the financial pretraining corpus for the controlled DAPT arm (BERT + continued MLM, BERT vocabulary).

Source: EDGAR-CORPUS (``eloukas/edgar-corpus``, Apache-2.0, pinned revision), annual-report (10-K) sections
Item 1 (Business) and Item 7 (MD&A) of the given years' ``validate.jsonl`` files (the dataset's own splits play no
role here). Leakage filter: a corpus sentence is dropped when its normalised text equals, or shares any 8-token
window with, a sentence of an evaluation set (Financial PhraseBank 4,846, SEntFiN 9,460, FiQA, Twitter). Documents
are rebuilt from the kept sentences and written one per line until the token budget (BERT word pieces) is reached.

Plan-only without ``--execute``. Output: ``<out-dir>/corpus.txt`` and ``<out-dir>/corpus_stats.json`` (counts, source
sha256, output sha256). The output is not committed (data/raw is git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from typing import Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dedup import normalize_text  # noqa: E402
from src.data.io import file_sha256  # noqa: E402

REPO, REVISION = "eloukas/edgar-corpus", "7e90f0f342569b35213445f809cfaf3b91f9964f"
SOURCE_SHA256 = {  # LFS oids of <year>/validate.jsonl at REVISION (newest first; read until the budget is met)
    "2020": "89bba2f87645bf0e6639204987fe71434fcf6d63222cee7d92f85e192a2d9f59",
    "2019": "9b220d29c23341cc9fa528dd81bdd06484aa246ad478231f57bb0097dd03b07e",
    "2018": "f0c62c3cec722a9e7f1e92ceb81e7b213bd6dacbd8b7ab4869207109ac56666e",
    "2017": "c231c1c1113cf2fd15a14b686163790c3afeb84aa98f88a1621446d48ada8285",
}
SECTIONS = ("section_1", "section_7")
NGRAM = 8
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
MIN_WORDS, MAX_WORDS = 6, 80


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    toks = normalize_text(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def eval_index(texts: Iterable[str]) -> tuple[set[str], set[str]]:
    """(normalised sentences, hashed 8-grams) of all evaluation texts."""
    sents, grams = set(), set()
    for t in texts:
        norm = normalize_text(t)
        if norm:
            sents.add(norm)
            grams |= {hashlib.md5(g.encode()).hexdigest()[:16] for g in ngrams(t)}
    return sents, grams


def is_leak(sentence: str, sents: set[str], grams: set[str]) -> bool:
    if normalize_text(sentence) in sents:
        return True
    return any(hashlib.md5(g.encode()).hexdigest()[:16] in grams for g in ngrams(sentence))


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n+", text or ""):
        for s in _SENT.split(para.strip()):
            s = re.sub(r"\s+", " ", s).strip()
            if MIN_WORDS <= len(s.split()) <= MAX_WORDS and re.search(r"[a-z]", s):
                out.append(s)
    return out


def eval_texts(splits_dir: str) -> list[str]:
    def load(name: str) -> dict:
        with open(os.path.join(splits_dir, name), encoding="utf-8") as handle:
            return json.load(handle)

    texts = [r["text"] for r in load("sentences_50agree_manifest.json")["records_by_id"].values()]
    texts += [r["text"] for r in load("sentfin_dedup.json")["records"]]
    extra = load("extra_external.json")["datasets"]
    for name in ("fiqa", "twitter"):
        texts += [r["text"] for r in extra[name]["records"]]
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", nargs="+", default=list(SOURCE_SHA256))
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw", "edgar"))
    parser.add_argument("--out-dir", default=os.path.join("data", "raw", "dapt"))
    parser.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    parser.add_argument("--token-budget", type=int, default=50_000_000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    urls = {y: f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{y}/validate.jsonl" for y in args.years}
    if not args.execute:
        for y, u in urls.items():
            print(f"would fetch {u}")
        print("Plan only; pass --execute.")
        return 0

    from src.transformer.finetune import MODEL_REVISIONS, _build_tokenizer

    tok = _build_tokenizer("google-bert/bert-base-uncased", MODEL_REVISIONS["google-bert/bert-base-uncased"])
    sents, grams = eval_index(eval_texts(args.splits_dir))
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    stats = {"source": {"repo": REPO, "revision": REVISION, "files": {}}, "sections": list(SECTIONS),
             "eval_sentences_indexed": len(sents), "eval_8grams_indexed": len(grams), "documents": 0,
             "sentences_seen": 0, "sentences_dropped_leak": 0, "sentences_kept": 0, "tokens": 0,
             "token_budget": args.token_budget}
    out_path = os.path.join(args.out_dir, "corpus.txt")
    done = False
    with open(out_path, "w", encoding="utf-8") as out:
        for year, url in urls.items():
            path = os.path.join(args.raw_dir, f"{year}_validate.jsonl")
            if not os.path.isfile(path) or file_sha256(path) != SOURCE_SHA256.get(year):
                urllib.request.urlretrieve(url, path)
            sha = file_sha256(path)
            if year in SOURCE_SHA256 and sha != SOURCE_SHA256[year]:
                raise RuntimeError(f"sha256 mismatch for {path}")
            stats["source"]["files"][year] = sha
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    rec = json.loads(line)
                    for section in SECTIONS:
                        kept = []
                        for s in split_sentences(rec.get(section) or ""):
                            stats["sentences_seen"] += 1
                            if is_leak(s, sents, grams):
                                stats["sentences_dropped_leak"] += 1
                            else:
                                kept.append(s)
                        if not kept:
                            continue
                        doc = " ".join(kept)
                        stats["tokens"] += len(tok(doc, add_special_tokens=False)["input_ids"])
                        stats["sentences_kept"] += len(kept)
                        stats["documents"] += 1
                        out.write(doc + "\n")
                        if stats["tokens"] >= args.token_budget:
                            done = True
                            break
                    if done:
                        break
            if done:
                break
    stats["budget_reached"] = done
    stats["output_sha256"] = file_sha256(out_path)
    with open(os.path.join(args.out_dir, "corpus_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
