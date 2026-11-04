"""Tokenizer comparison of the two fine-tuned encoders (vocabulary confound of the Δdomain contrast).

``google-bert/bert-base-uncased`` and ``yiyanghkust/finbert-pretrain`` differ in more than the pretraining corpus:
FinBERT-MLM ships its own uncased WordPiece vocabulary (FinVocab). The Step 2 contrast
``Δdomain = FinBERT-MLM - BERT`` therefore measures corpus, vocabulary and pretraining budget together. This script
quantifies the vocabulary part without training anything:

1. vocabulary sizes and overlap (whole-word entries and ``##`` continuation pieces separately);
2. per corpus (Financial PhraseBank, all 4,846 sentences; SEntFiN 1.0, 9,460 headlines) and tokenizer: word pieces
   per sentence, pieces per word (fertility), the share of words split into several pieces, the [UNK] rate and the
   share of sentences longer than the 128-token training limit;
3. how a list of financial terms is split by each tokenizer;
4. whether the stored Step 2 predictions differ by tokenization: sentences are grouped by whether FinBERT-MLM's
   tokenizer needs fewer, as many or more pieces than BERT's, and ``Δdomain`` accuracy is reported per group
   (seed+sentence bootstrap, as in ``results/step2/attribution.json``).

No model is loaded or refit; only the two tokenizers (pinned revisions from ``src/transformer/finetune.py``) are read
from the local Hugging Face cache. Writes ``<out-dir>/tokenizer_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts import summarize_step2 as summ  # noqa: E402
from src.evaluation.multiseed import load_runs  # noqa: E402

BERT = "google-bert/bert-base-uncased"
FINBERT = "yiyanghkust/finbert-pretrain"
TOKENIZER_KEYS = {"bert_base": BERT, "finbert_mlm": FINBERT}
MAX_LENGTH = 128  # training max_length incl. [CLS] and [SEP]
SPECIAL = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
FINANCE_TERMS = (
    "cost", "costs", "liability", "liabilities", "debt", "depreciation", "amortization", "amortisation",
    "impairment", "impairments", "writedown", "write-off", "provision", "provisions", "restructuring",
    "ebitda", "ebit", "dividend", "dividends", "revenue", "revenues", "profit", "loss", "losses",
    "earnings", "guidance", "shareholders", "quarterly", "year-on-year", "bearish", "bullish", "downgrade",
    "upgrade", "outperform", "underperform", "leverage", "volatility", "mln", "bn", "eur", "nasdaq",
)
# Pretraining corpora as stated by the model cards / papers (not measured here)
PRETRAINING = {
    BERT: {"corpus": "BooksCorpus (800M words) + English Wikipedia (2,500M words)",
           "source": "Devlin et al. 2019, BERT, section 3.1"},
    FINBERT: {"corpus": "10-K/10-Q reports (2.5B tokens), earnings call transcripts (1.3B), analyst reports (1.1B); "
                        "4.9B tokens in total",
              "source": "https://huggingface.co/yiyanghkust/finbert-pretrain (model card)"},
}
GROUPS = ("finbert_fewer", "same", "finbert_more")


def load_tokenizers() -> dict[str, Any]:
    from src.transformer.finetune import MODEL_REVISIONS, _build_tokenizer

    return {key: _build_tokenizer(name, MODEL_REVISIONS[name]) for key, name in TOKENIZER_KEYS.items()}


def vocab_overlap(vocab_a: Sequence[str], vocab_b: Sequence[str]) -> dict[str, Any]:
    # special tokens and BERT's ~1,000 reserved "[unusedN]" slots are not real vocabulary
    a = {t for t in vocab_a if t not in SPECIAL and not t.startswith("[unused")}
    b = {t for t in vocab_b if t not in SPECIAL and not t.startswith("[unused")}

    def part(sa: set, sb: set) -> dict[str, Any]:
        shared = sa & sb
        union = sa | sb
        return {"a": len(sa), "b": len(sb), "shared": len(shared),
                "share_of_a": len(shared) / max(len(sa), 1), "share_of_b": len(shared) / max(len(sb), 1),
                "jaccard": len(shared) / max(len(union), 1)}

    def words(s: set) -> set:
        return {t for t in s if not t.startswith("##")}

    def pieces(s: set) -> set:
        return {t for t in s if t.startswith("##")}

    return {"all": part(a, b), "whole_words": part(words(a), words(b)), "continuation_pieces": part(pieces(a), pieces(b))}


def tokenize_stats(tokenizer: Any, texts: Sequence[str]) -> tuple[dict[str, Any], np.ndarray]:
    """Corpus statistics and the per-sentence word-piece count (without [CLS]/[SEP])."""
    enc = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    unk_id = tokenizer.unk_token_id
    n_pieces = np.array([len(ids) for ids in enc["input_ids"]], dtype=np.int64)
    n_words, n_split, n_unk = 0, 0, 0
    for k, ids in enumerate(enc["input_ids"]):
        word_ids = enc.word_ids(k)
        counts: dict[int, int] = {}
        for w in word_ids:
            if w is not None:
                counts[w] = counts.get(w, 0) + 1
        n_words += len(counts)
        n_split += sum(1 for c in counts.values() if c > 1)
        n_unk += sum(1 for i in ids if i == unk_id)
    total = int(n_pieces.sum())
    stats = {
        "n_sentences": len(texts),
        "pieces_per_sentence": float(n_pieces.mean()) if len(texts) else None,
        "pieces_per_word": total / max(n_words, 1),
        "share_words_split": n_split / max(n_words, 1),
        "unk_rate": n_unk / max(total, 1),
        "share_over_max_length": float(np.mean(n_pieces > MAX_LENGTH - 2)) if len(texts) else None,
    }
    return stats, n_pieces


def term_pieces(tokenizers: dict[str, Any], terms: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    return {t: {key: tok.tokenize(t) for key, tok in tokenizers.items()} for t in terms}


def tokenization_groups(n_bert: np.ndarray, n_fin: np.ndarray) -> dict[str, np.ndarray]:
    return {"finbert_fewer": n_fin < n_bert, "same": n_fin == n_bert, "finbert_more": n_fin > n_bert}


def domain_by_group(table: summ.RunTable, config: str, split: str, texts_by_id: dict[str, str],
                    count_fn: Callable[[str, list[str]], np.ndarray], B: int, seed: int) -> dict[str, Any] | None:
    ref = summ._reference(table, config, split)
    if ref is None:
        return None
    ids, y = ref
    bert = summ._model_preds(table, config, "bert_base", split, ids)
    fin = summ._model_preds(table, config, "finbert_mlm", split, ids)
    if not bert or not fin:
        return None
    texts = [texts_by_id[i] for i in ids.tolist()]
    groups = tokenization_groups(count_fn("bert_base", texts), count_fn("finbert_mlm", texts))
    out: dict[str, Any] = {"n": int(len(y))}
    for name, mask in groups.items():
        entry: dict[str, Any] = {"n": int(mask.sum())}
        if mask.sum() >= 20:
            yy = y[mask]
            entry["bert_accuracy"] = float(np.mean([np.mean(p[mask] == yy) for p in bert.values()]))
            entry["finbert_accuracy"] = float(np.mean([np.mean(p[mask] == yy) for p in fin.values()]))
            cell = summ.attribution_cell(yy, [p[mask] for p in fin.values()], [p[mask] for p in bert.values()],
                                         "accuracy", B, seed)
            entry["delta_domain_accuracy"] = {k: cell[k] for k in ("delta", "ci_low", "ci_high", "p_value", "B")}
        out[name] = entry
    return out


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=os.path.join("results", "step2", "runs"))
    parser.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    parser.add_argument("--out-dir", default=os.path.join("results", "extensions"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    tokenizers = load_tokenizers()
    fpb = _load_json(os.path.join(args.splits_dir, "sentences_50agree_manifest.json"))["records_by_id"]
    sentfin = {str(r["id"]): r["text"] for r in _load_json(os.path.join(args.splits_dir, "sentfin_dedup.json"))["records"]}
    corpora = {"financial_phrasebank_all": [r["text"] for r in fpb.values()], "sentfin": list(sentfin.values())}

    result: dict[str, Any] = {
        "tokenizers": {key: {"name": name, "vocab_size": len(tokenizers[key].get_vocab()),
                             "pretraining": PRETRAINING[name]} for key, name in TOKENIZER_KEYS.items()},
        "vocab_overlap": vocab_overlap(list(tokenizers["bert_base"].get_vocab()),
                                       list(tokenizers["finbert_mlm"].get_vocab())),
        "corpora": {},
        "finance_terms": term_pieces(tokenizers, FINANCE_TERMS),
        "delta_domain_by_tokenization": {},
    }
    for corpus, texts in corpora.items():
        result["corpora"][corpus] = {key: tokenize_stats(tok, texts)[0] for key, tok in tokenizers.items()}

    def count_fn(key: str, texts: list[str]) -> np.ndarray:
        return tokenize_stats(tokenizers[key], texts)[1]

    table = summ.RunTable(load_runs(args.runs_dir))
    for config in summ.CONFIGS:
        manifest = _load_json(os.path.join(args.splits_dir, f"{config}_manifest.json"))["records_by_id"]
        texts_by_id = {i: r["text"] for i, r in manifest.items()}
        cell = {"test": domain_by_group(table, config, "test", texts_by_id, count_fn, args.bootstrap_samples, args.seed),
                "external": domain_by_group(table, config, "external", sentfin, count_fn, args.bootstrap_samples,
                                            args.seed)}
        result["delta_domain_by_tokenization"][config] = cell

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tokenizer_comparison.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    ov = result["vocab_overlap"]["all"]
    print(f"vocab: BERT {ov['a']}, FinBERT-MLM {ov['b']}, shared {ov['shared']} "
          f"({ov['share_of_a']:.1%} of BERT, {ov['share_of_b']:.1%} of FinBERT-MLM)")
    for corpus, stats in result["corpora"].items():
        for key, s in stats.items():
            print(f"{corpus:26s} {key:12s} pieces/word {s['pieces_per_word']:.3f}  split words "
                  f"{s['share_words_split']:.1%}  unk {s['unk_rate']:.4f}  >128 {s['share_over_max_length']:.3%}")
    for config, cell in result["delta_domain_by_tokenization"].items():
        for split, groups in cell.items():
            if not groups:
                continue
            parts = []
            for g in GROUPS:
                e = groups[g]
                d = e.get("delta_domain_accuracy")
                parts.append(f"{g} n={e['n']}" + (f" {d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
                                                  if d else ""))
            print(f"{config:20s} {split:8s} " + " | ".join(parts))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
