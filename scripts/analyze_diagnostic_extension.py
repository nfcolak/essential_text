"""Analyze both-annotator consensus on the full locked 50agree test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_diagnostic_extension import (
    CONFIG,
    DIAGNOSTIC_DIR,
    N_EXTENSION,
    N_ORIGINAL,
    SPLITS_DIR,
    check_files,
    load_test,
)
from scripts.run_step3_analysis import (
    MissingInput,
    _attribution_table,
    _clean,
    _config_preds,
    _fmt,
    _fmt_ci,
    _load_json,
    _name,
    _pick_tfidf_best,
    _read_ann,
    _run_preds,
)
from src.analysis.attribution import (
    PHENOMENA,
    agreement_table,
    consensus_tags,
    phenomenon_attribution,
    phenomenon_error_rates,
)

ANNOTATION_FILES = ("annotations_a.csv", "annotations_b.csv", "rule_tags.csv")
LLM_DIR = REPO_ROOT / "results" / "llm_zeroshot" / "phrasebank_50agree_test"


def _check_ids(expected, actual, context: str) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{context}: ID mismatch (missing {missing[:5]}, extra {extra[:5]})")


def _read_wave(directory: Path, key_name: str, id_field: str, n: int, records: dict,
               manifest_hash: str) -> tuple[dict[str, dict], ...]:
    key_path = directory / key_name
    key = _load_json(str(key_path))
    if (key["config"], key["source_split"], key["manifest_sha256"]) != (
        CONFIG, "test", manifest_hash
    ):
        raise ValueError(f"{key_path} does not match the locked test manifest")
    by_record = {item["record_id"]: item for item in key["items"]}
    by_annotation = {item[id_field]: record_id for record_id, item in by_record.items()}
    if any(size != n for size in (len(by_record), len(by_annotation), len(key["items"]), key["n"])):
        raise ValueError(f"{key_path}: expected {n} items with unique record and annotation IDs")
    for record_id, item in by_record.items():
        if record_id not in records or int(item["label"]) != records[record_id][1]:
            raise ValueError(f"{key_path}: unexpected record or gold label for {record_id}")

    wave = []
    for filename in ANNOTATION_FILES:
        path = directory / filename
        tags = _read_ann(str(path))
        _check_ids(by_annotation.keys(), tags.keys(), str(path))
        wave.append({by_annotation[annotation_id]: row for annotation_id, row in tags.items()})
    return tuple(wave)


def analyze(original: tuple[dict, ...], extension: tuple[dict, ...], gold: dict, predictions: dict,
            tfidf_best: str, bootstrap_samples: int, seed: int) -> dict:
    union = tuple({**left, **right} for left, right in zip(original, extension))
    tags = consensus_tags(union[0], union[1], mode="both")
    return {
        "n": len(tags),
        "n_original": len(original[0]),
        "n_extension": len(extension[0]),
        "error_rates": phenomenon_error_rates(tags, gold, predictions, B=bootstrap_samples, seed=seed),
        "attribution": phenomenon_attribution(
            tags, gold, predictions, tfidf_best, B=bootstrap_samples, seed=seed),
        "agreement": {"extension": agreement_table(*extension), "union": agreement_table(*union)},
    }


def _report(payload: dict) -> str:
    rates = payload["error_rates"]
    models = list(rates[PHENOMENA[0]]["models"])
    lines = [
        "# Full 50agree test-split phenomenon analysis",
        "",
        f"The analysis covers all {payload['n']} locked test records "
        f"({payload['n_original']} original and {payload['n_extension']} extension items). "
        "A phenomenon is present when both blind LLM annotators marked it.",
        "",
        "## Annotator agreement",
        "",
        "| Phenomenon | Cohen’s κ A–B (extension) | Cohen’s κ A–B (union) |",
        "|---|---|---|",
    ]
    for name in PHENOMENA:
        ext = payload["agreement"]["extension"][name]["kappa_ab"]
        union = payload["agreement"]["union"][name]["kappa_ab"]
        lines.append(f"| {name} | {_fmt(ext)} | {_fmt(union)} |")
    lines += [
        "",
        "## Error rates",
        "",
        f"95% intervals use {payload['bootstrap_samples']:,} bootstrap draws over sentences and "
        "model seeds. Label-balanced error averages the error rates of the gold classes present.",
    ]
    for name, rate in rates.items():
        lines += ["", f"### {name}", ""]
        for tag in ("tag1", "tag0"):
            mix = rate[f"gold_mix_{tag}"]
            subset = "Tagged" if tag == "tag1" else "Untagged"
            lines.append(f"- {subset} (n = {rate['n_' + tag]}) has {mix['n_neg']} negative, "
                         f"{mix['n_neu']} neutral, and {mix['n_pos']} positive gold labels.")
        lines += [
            "",
            "| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |",
            "|---|---|---|---|---|",
        ]
        for model in models:
            label = "Zero-shot LLM" if model == "llm_zeroshot" else _name(model)
            entries = rate["models"][model]
            cells = [_fmt_ci(entries[tag], metric) for tag in ("tag1", "tag0")
                     for metric in ("error_rate", "balanced_error")]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Architecture and domain-pretraining attribution",
        "",
        f"Architecture is the accuracy of BERT-base minus the accuracy of `{payload['tfidf_best']}`, "
        "the TF-IDF baseline with the highest validation macro-F1. Domain pretraining is the accuracy "
        "of FinBERT-MLM minus the accuracy of BERT-base. Holm correction covers the tested comparisons "
        "in this table. Rows marked with an asterisk have fewer than 20 tagged sentences and report "
        "intervals only.",
        "",
    ]
    lines += _attribution_table(payload["attribution"])
    if "llm_zeroshot" in models:
        lines += ["The zero-shot LLM may have seen Financial PhraseBank during pretraining.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DIAGNOSTIC_DIR,
                        help="use the Step 3 annotation CSV schema (diag_id) for both waves")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "results" / "step2" / "runs")
    parser.add_argument("--llm-dir", type=Path, default=LLM_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "step3" / "full_split")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="compare existing files without writing")
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    try:
        extension_dir = args.diagnostic_dir / "extension"
        missing = [str(directory / name) for directory in (args.diagnostic_dir, extension_dir)
                   for name in ("annotations_a.csv", "annotations_b.csv")
                   if not (directory / name).is_file()]
        if missing:
            raise MissingInput(f"Missing annotation files: {', '.join(missing)}")
        records, manifest_hash = load_test(args.splits_dir)
        original = _read_wave(args.diagnostic_dir, "diagnostic_key.json", "diag_id",
                              N_ORIGINAL, records, manifest_hash)
        extension = _read_wave(extension_dir, "key.json", "ext_id", N_EXTENSION, records, manifest_hash)
        if original[0].keys() & extension[0].keys():
            raise ValueError("The original and extension keys overlap")
        _check_ids(records.keys(), original[0].keys() | extension[0].keys(), "Full test split")

        runs = _config_preds(str(args.runs_dir), CONFIG, "test")
        if (args.llm_dir / "predictions.npz").is_file():
            runs["llm_zeroshot"] = [_run_preds(str(args.llm_dir), "test")]
        for model, seed_preds in runs.items():
            for run, predictions in enumerate(seed_preds, start=1):
                _check_ids(records.keys(), predictions.keys(), f"{model} test predictions (run {run})")
        best, scores = _pick_tfidf_best(str(args.runs_dir), CONFIG)
        gold = {record_id: label for record_id, (_, label) in records.items()}
        payload = _clean({
            "config": CONFIG,
            "manifest_sha256": manifest_hash,
            "consensus_mode": "both",
            "tfidf_best": best,
            "tfidf_val_macro_f1": scores,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            **analyze(original, extension, gold, runs, best, args.bootstrap_samples, args.seed),
        })
        files = {
            "phenomenon_analysis.json": (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            "REPORT.md": _report(payload).encode("utf-8"),
        }
        if args.check:
            return check_files(args.out_dir, files)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (args.out_dir / name).write_bytes(content)
    except (MissingInput, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote analysis for {payload['n']} test items to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
