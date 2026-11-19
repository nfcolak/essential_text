#!/usr/bin/env python3
"""Score locked financial sentiment splits with local Ollama. Plan-only unless --execute."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
from urllib import request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_extra_external import OUT_FILE, load_extra_external
from scripts.run_step2 import _Data
from src.data.io import split_arrays
from src.evaluation.metrics import CLASS_NAMES, LABEL_TO_ID, compute_classification_metrics

URL = "http://localhost:11434"
DEFAULT_OUT = "results/llm_zeroshot"
DEFAULT_MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 16}
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": CLASS_NAMES}},
          "required": ["label"], "additionalProperties": False}
SYSTEM_PROMPT = ("Classify the financial sentiment expressed by the sentence for the company or market "
                 "as negative, neutral, or positive. Return a JSON object with the single field label.")
USER_PROMPT = "Sentence:\n{sentence}"
NOTE = "the LLM may have seen Financial PhraseBank during pretraining"
PHRASEBANK = {"phrasebank_allagree_test": "sentences_allagree",
              "phrasebank_50agree_test": "sentences_50agree"}
DATASETS = (*PHRASEBANK, "sentfin", "fiqa", "twitter")


def dataset_arrays(data, name):
    if name in PHRASEBANK:
        return split_arrays(data.manifest(PHRASEBANK[name]), "test")
    if name == "sentfin":
        return data.external()
    return load_extra_external(str(Path(data.splits_dir) / OUT_FILE), name)


def api(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = request.Request(URL + path, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=180) as response:
        return json.load(response)


def model_digest(model):
    for tag in api("/api/tags")["models"]:
        if tag["name"] == model and tag.get("digest"):
            return tag["digest"]
    raise ValueError(f"Ollama has no installed model with a digest for {model!r}")


def parse_label(raw):
    if isinstance(raw, str):
        try:
            fields = json.loads(raw)
        except ValueError:
            return None
        if isinstance(fields, dict) and set(fields) == {"label"} and fields["label"] in CLASS_NAMES:
            return fields["label"]
    return None


def classify(model, rid, sentence):
    payload = {"model": model, "system": SYSTEM_PROMPT, "prompt": USER_PROMPT.format(sentence=sentence),
               "stream": False, "options": OPTIONS, "format": SCHEMA}
    responses = []
    for attempt in (1, 2):
        raw = api("/api/generate", payload)["response"]
        responses.append(raw)
        label = parse_label(raw)
        if label is not None:
            break
    return {"id": rid, "raw": raw, "responses": responses, "label": label or "neutral",
            "valid": label is not None, "attempts": attempt}


def read_cache(handle, signature, allowed_ids):
    """An interrupted append may leave an unterminated tail. Only newline-terminated rows commit."""
    handle.seek(0)
    rows = {}
    while True:
        start = handle.tell()
        line = handle.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            handle.truncate(start)
            break
        row = json.loads(line)
        rid = row["id"]
        if row.get("signature") != signature:
            raise ValueError(f"Cache provenance differs at {rid}. Use another --out directory.")
        if rid not in allowed_ids or rid in rows:
            raise ValueError(f"Unexpected or duplicate cached id {rid}")
        label = parse_label(row["raw"])
        if row["valid"] != (label is not None) or row["label"] != (label or "neutral"):
            raise ValueError(f"Inconsistent cached prediction for {rid}")
        rows[rid] = row
    return rows


def run_dataset(args, name, arrays, digest):
    ids, texts, y = arrays
    if not ids or len(set(ids)) != len(ids) or not (len(ids) == len(texts) == len(y)):
        raise ValueError(f"{name} has empty, duplicate or unpaired records")
    if any(label not in LABEL_TO_ID.values() for label in y):
        raise ValueError(f"{name} has an unknown sentiment label")
    metadata = {"dataset": name, "model": args.model, "digest": digest, "options": OPTIONS,
                "prompt": {"system": SYSTEM_PROMPT, "user": USER_PROMPT}, "format": SCHEMA}
    signature = hashlib.sha256(json.dumps([metadata, arrays], sort_keys=True).encode()).hexdigest()
    directory = args.out / name
    directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    with (directory / "raw.jsonl").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = read_cache(handle, signature, set(ids))
        ids, texts, y = ids[:args.limit], texts[:args.limit], y[:args.limit]
        if rows.keys() - set(ids):
            raise ValueError("The cache covers more items than this limit. Use another --out directory.")
        pool = ThreadPoolExecutor(max_workers=args.workers)
        try:
            pending = [pool.submit(classify, args.model, rid, text)
                       for rid, text in zip(ids, texts) if rid not in rows]
            for future in as_completed(pending):
                row = future.result()
                row["signature"] = signature
                handle.write((json.dumps(row) + "\n").encode())
                handle.flush()
                rows[row["id"]] = row
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        split = "test" if name in PHRASEBANK else "external"
        pred = [LABEL_TO_ID[rows[rid]["label"]] for rid in ids]
        prob = np.eye(len(CLASS_NAMES), dtype=np.float32)[pred]
        result = dict(metadata, n=len(ids), n_invalid=sum(not rows[rid]["valid"] for rid in ids),
                      metrics=compute_classification_metrics(y, pred, prob),
                      timestamp=datetime.now(timezone.utc).isoformat(), dataset_size=len(arrays[0]))
        if name in PHRASEBANK:
            result["contamination_note"] = NOTE
        temp = directory / "predictions.tmp.npz"
        np.savez_compressed(temp, **{f"{split}_ids": np.asarray(ids), f"{split}_y": np.asarray(y),
                                     f"{split}_prob": prob})
        temp.replace(directory / "predictions.npz")
        temp = directory / "result.tmp.json"
        temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temp.replace(directory / "result.json")
    elapsed = monotonic() - started
    print(f"{name}: {len(ids)} rows, {result['n_invalid']} invalid, "
          f"{len(pending)} newly scored in {elapsed:.2f} s ({len(pending)/elapsed:.3f} items/s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    try:
        data = _Data(args.splits_dir)
        digest = model_digest(args.model) if args.execute else None
        for name in args.datasets:
            arrays = dataset_arrays(data, name)
            if args.execute:
                run_dataset(args, name, arrays, digest)
            else:
                print(f"{name}: {len(arrays[0][:args.limit])} items with {args.model} -> {args.out / name}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
