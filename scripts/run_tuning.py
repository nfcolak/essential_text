#!/usr/bin/env python3
"""Small learning-rate and batch-size study for bert_base and finbert_mlm.

Reuses ``scripts/run_step2.py``. The search is the Cartesian product

    models {bert_base, finbert_mlm}
    x learning rates {2e-5, 3e-5}
    x batch sizes {16, 32}

on the locked ``sentences_allagree`` training and validation splits (8 runs).
Selection uses seed 42 and validation Macro-F1 only.

The winners are then trained with seeds 42, 100 and 2026. Selection runs train on the
locked training split and score validation only: the PhraseBank test split and SEntFiN
are not loaded, scored, or saved. The selection run does not keep a checkpoint, so the
final stage retrains seed 42 together with seeds 100 and 2026. That second training is
the first time those holdouts are scored. Ties break toward the lower learning rate,
then the smaller batch size.

Outputs (never under ``results/step2``)::

    results/tuning/tuning_results.csv
    results/tuning/final_results.csv
    results/tuning/selection.json
    results/tuning/selection/<config>/<model>/<tag>/
    results/tuning/final/<config>/<model>/<tag>/

Plan only unless ``--execute`` is passed. ``--write-tables`` rebuilds the CSVs
from completed ``result.json`` files and does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_step2

PHRASEBANK_CONFIG = "sentences_allagree"
TUNING_MODELS = ("bert_base", "finbert_mlm")
LEARNING_RATES = (2e-5, 3e-5)
BATCH_SIZES = (16, 32)
SELECTION_SEED = 42
FINAL_SEEDS = (42, 100, 2026)
SELECTION_METRIC = "val_macro_f1"

DEFAULT_OUT_DIR = "results/tuning"
DEFAULT_SPLITS_DIR = run_step2.DEFAULT_SPLITS_DIR
SELECTION_DIRNAME = "selection"
FINAL_DIRNAME = "final"
TUNING_CSV = "tuning_results.csv"
FINAL_CSV = "final_results.csv"
SELECTION_JSON = "selection.json"

TUNING_FIELDS = (
    "model_key", "learning_rate", "batch_size", "seed",
    "val_macro_f1", "val_accuracy", "selected", "run_dir",
)
FINAL_FIELDS = (
    "model_key", "learning_rate", "batch_size", "seed", "best_epoch",
    "val_macro_f1",
    "test_accuracy", "test_macro_f1", "test_f1_negative", "test_f1_neutral", "test_f1_positive",
    "external_accuracy", "external_macro_f1",
    "external_f1_negative", "external_f1_neutral", "external_f1_positive",
    "run_dir",
)

Runner = Callable[[Sequence[str]], int]


class IncompleteRuns(Exception):
    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__(f"{len(self.missing)} completed tuning run(s) are missing")


def iter_recipes() -> list[tuple[float, int]]:
    return [(lr, bs) for lr in LEARNING_RATES for bs in BATCH_SIZES]


def iter_grid() -> list[tuple[str, float, int]]:
    """Model-major order: all recipes for bert_base, then all recipes for finbert_mlm."""
    return [(model, lr, bs) for model in TUNING_MODELS for lr, bs in iter_recipes()]


def _recipe_key(model_key: str, learning_rate: float, batch_size: int, seed: int) -> tuple:
    return (model_key, run_step2.format_learning_rate(learning_rate), int(batch_size), int(seed))


def _same_recipe(row: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    left = _recipe_key(row["model_key"], row["learning_rate"], row["batch_size"], 0)
    right = _recipe_key(other["model_key"], other["learning_rate"], other["batch_size"], 0)
    return left[:3] == right[:3]


def select_best(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """One winner per model. The sort key reads validation Macro-F1, learning rate and batch size only."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_key"]), []).append(row)
    return {
        model: min(group, key=_selection_key)
        for model, group in grouped.items()
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    # Higher validation Macro-F1 wins. Ties: lower learning rate, then smaller batch size.
    # Test and external scores are not arguments of this key.
    return (-float(row["val_macro_f1"]), float(row["learning_rate"]), int(row["batch_size"]))


def run_directory(runs_dir: str, model_key: str, learning_rate: float, batch_size: int, seed: int) -> str:
    tag = run_step2.finetune_tag(seed, learning_rate, batch_size)
    return os.path.join(runs_dir, PHRASEBANK_CONFIG, model_key, tag)


def _rel_run(run_dir: str, out_dir: str) -> str:
    return os.path.relpath(run_dir, out_dir).replace(os.sep, "/")


def _load_result(path: str) -> dict[str, Any]:
    run_dir = os.path.dirname(path)
    if not run_step2._is_completed(run_dir):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path} is not a result object")
    return result


def _require_slot(result: Mapping[str, Any], path: str, model_key: str, learning_rate: float,
                  batch_size: int, seed: int) -> None:
    if result.get("model_key") != model_key or int(result.get("seed")) != int(seed):
        raise ValueError(f"{path} does not match {model_key} seed {seed}.")
    if result.get("config") != PHRASEBANK_CONFIG:
        raise ValueError(f"{path} config is {result.get('config')!r}, expected {PHRASEBANK_CONFIG}.")
    hyper = result.get("hyperparams") or {}
    try:
        got_lr = run_step2.format_learning_rate(hyper["learning_rate"])
        got_bs = int(hyper["batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} is missing learning_rate or batch_size in hyperparams.") from exc
    if got_lr != run_step2.format_learning_rate(learning_rate) or got_bs != int(batch_size):
        raise ValueError(
            f"{path} records learning_rate={hyper.get('learning_rate')} batch_size={hyper.get('batch_size')}, "
            f"expected {learning_rate} and {batch_size}."
        )


def _val_scores(result: Mapping[str, Any], path: str) -> tuple[float, float]:
    val = (result.get("metrics") or {}).get("val") or {}
    try:
        macro_f1 = float(val["macro_f1"])
        accuracy = float(val["accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has no validation accuracy/macro-F1; selection cannot use it.") from exc
    if not math.isfinite(macro_f1) or not math.isfinite(accuracy):
        raise ValueError(f"{path} validation metrics are not finite.")
    return macro_f1, accuracy


def _assert_validation_only(result: Mapping[str, Any], path: str) -> None:
    """Reject a selection run that loaded or stored a holdout. Metric values are not read."""
    metrics = result.get("metrics") or {}
    if metrics.get("test") is not None or metrics.get("external") is not None:
        raise ValueError(
            f"{path} contains PhraseBank test or SEntFiN metrics. "
            "Selection runs must stay validation-only."
        )
    counts = result.get("n") or {}
    if counts.get("test") is not None or counts.get("external") is not None:
        raise ValueError(
            f"{path} records a test or SEntFiN size. Selection runs must stay validation-only."
        )
    if result.get("external_sha256"):
        raise ValueError(f"{path} records an external-file hash. Selection runs must not load SEntFiN.")
    npz_path = os.path.join(os.path.dirname(path), "predictions.npz")
    if os.path.isfile(npz_path):
        with np.load(npz_path) as npz:
            heldout = [name for name in npz.files if name.startswith(("test_", "external_"))]
        if heldout:
            raise ValueError(
                f"{os.path.dirname(path)} saved holdout predictions {heldout}. "
                "Selection runs must stay validation-only."
            )


def collect_selection_rows(runs_dir: str, out_dir: str) -> list[dict[str, Any]]:
    """Load the seed-42 grid. Rows contain validation scores only."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model_key, learning_rate, batch_size in iter_grid():
        run_dir = run_directory(runs_dir, model_key, learning_rate, batch_size, SELECTION_SEED)
        path = os.path.join(run_dir, "result.json")
        if not run_step2._is_completed(run_dir):
            missing.append(run_dir)
            continue
        result = _load_result(path)
        _require_slot(result, path, model_key, learning_rate, batch_size, SELECTION_SEED)
        _assert_validation_only(result, path)
        macro_f1, accuracy = _val_scores(result, path)
        rows.append({
            "model_key": model_key,
            "learning_rate": float(result["hyperparams"]["learning_rate"]),
            "batch_size": int(result["hyperparams"]["batch_size"]),
            "seed": SELECTION_SEED,
            "val_macro_f1": macro_f1,
            "val_accuracy": accuracy,
            "run_dir": _rel_run(run_dir, out_dir),
        })
    if missing:
        raise IncompleteRuns(missing)
    return rows


def _split_metrics(result: Mapping[str, Any], path: str, split: str) -> dict[str, float]:
    block = (result.get("metrics") or {}).get(split)
    if not block:
        raise ValueError(f"{path} has no {split} metrics.")
    try:
        per_class = block["per_class"]
        values = {
            "accuracy": float(block["accuracy"]),
            "macro_f1": float(block["macro_f1"]),
            "f1_negative": float(per_class["negative"]["f1"]),
            "f1_neutral": float(per_class["neutral"]["f1"]),
            "f1_positive": float(per_class["positive"]["f1"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} {split} metrics are incomplete.") from exc
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"{path} {split} metrics are not finite.")
    return values


def collect_final_rows(runs_dir: str, out_dir: str, chosen: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model_key in TUNING_MODELS:
        if model_key not in chosen:
            raise ValueError(f"no selected configuration for {model_key}.")
        spec = chosen[model_key]
        for seed in FINAL_SEEDS:
            run_dir = run_directory(runs_dir, model_key, spec["learning_rate"], spec["batch_size"], seed)
            path = os.path.join(run_dir, "result.json")
            if not run_step2._is_completed(run_dir):
                missing.append(run_dir)
                continue
            result = _load_result(path)
            _require_slot(result, path, model_key, spec["learning_rate"], spec["batch_size"], seed)
            val_f1, _accuracy = _val_scores(result, path)
            test = _split_metrics(result, path, "test")
            external = _split_metrics(result, path, "external")
            rows.append({
                "model_key": model_key,
                "learning_rate": float(result["hyperparams"]["learning_rate"]),
                "batch_size": int(result["hyperparams"]["batch_size"]),
                "seed": int(seed),
                "best_epoch": result.get("best_epoch"),
                "val_macro_f1": val_f1,
                "test": test,
                "external": external,
                "run_dir": _rel_run(run_dir, out_dir),
            })
    if missing:
        raise IncompleteRuns(missing)
    return rows


def _fmt_metric(value: float) -> str:
    return f"{float(value):.6f}"


def _tuning_records(rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    order = {_recipe_key(model, lr, bs, SELECTION_SEED): index for index, (model, lr, bs) in enumerate(iter_grid())}
    ordered = sorted(rows, key=lambda row: order[_recipe_key(row["model_key"], row["learning_rate"], row["batch_size"], row["seed"])])
    records = []
    for row in ordered:
        winner = chosen[row["model_key"]]
        records.append({
            "model_key": row["model_key"],
            "learning_rate": run_step2.format_learning_rate(row["learning_rate"]),
            "batch_size": str(int(row["batch_size"])),
            "seed": str(int(row["seed"])),
            "val_macro_f1": _fmt_metric(row["val_macro_f1"]),
            "val_accuracy": _fmt_metric(row["val_accuracy"]),
            "selected": "1" if _same_recipe(row, winner) else "0",
            "run_dir": row["run_dir"],
        })
    return records


def _final_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    records = []
    for row in rows:
        record = {
            "model_key": row["model_key"],
            "learning_rate": run_step2.format_learning_rate(row["learning_rate"]),
            "batch_size": str(int(row["batch_size"])),
            "seed": str(int(row["seed"])),
            "best_epoch": "" if row["best_epoch"] is None else str(int(row["best_epoch"])),
            "val_macro_f1": _fmt_metric(row["val_macro_f1"]),
            "run_dir": row["run_dir"],
        }
        for split in ("test", "external"):
            for name, value in row[split].items():
                record[f"{split}_{name}"] = _fmt_metric(value)
        records.append(record)
    return records


def _write_csv(path: str, fieldnames: Sequence[str], records: Sequence[Mapping[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def _write_selection(out_dir: str, rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Mapping[str, Any]]) -> None:
    records = _tuning_records(rows, chosen)
    _write_csv(os.path.join(out_dir, TUNING_CSV), TUNING_FIELDS, records)
    payload = {
        "phrasebank_config": PHRASEBANK_CONFIG,
        "selection_seed": SELECTION_SEED,
        "selection_metric": SELECTION_METRIC,
        "tie_break": ["max val_macro_f1", "min learning_rate", "min batch_size"],
        "selected": {
            model: {
                "learning_rate": float(row["learning_rate"]),
                "batch_size": int(row["batch_size"]),
                "val_macro_f1": float(row["val_macro_f1"]),
            }
            for model, row in chosen.items()
        },
        "candidates": records,
    }
    with open(os.path.join(out_dir, SELECTION_JSON), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {os.path.join(out_dir, TUNING_CSV)}")


def _write_final(out_dir: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = os.path.join(out_dir, FINAL_CSV)
    _write_csv(path, FINAL_FIELDS, _final_records(rows))
    print(f"wrote {path}")


def _announce(chosen: Mapping[str, Mapping[str, Any]]) -> None:
    for model in TUNING_MODELS:
        row = chosen[model]
        print(
            f"selected {model}: learning_rate={run_step2.format_learning_rate(row['learning_rate'])} "
            f"batch_size={int(row['batch_size'])} val_macro_f1={float(row['val_macro_f1']):.4f}"
        )


def _step2_argv(runs_dir: str, splits_dir: str, models: Sequence[str], learning_rate: float,
                batch_size: int, seeds: Sequence[int], execute: bool, val_only: bool) -> list[str]:
    argv = [
        "--stage", "finetune",
        "--configs", PHRASEBANK_CONFIG,
        "--models", *models,
        "--seeds", *[str(int(seed)) for seed in seeds],
        "--learning-rate", run_step2.format_learning_rate(learning_rate),
        "--batch-size", str(int(batch_size)),
        "--splits-dir", splits_dir,
        "--out-dir", runs_dir,
    ]
    if val_only:
        argv.append("--val-only")
    if execute:
        argv.append("--execute")
    return argv


def _launch(runner: Runner, runs_dir: str, splits_dir: str, models: Sequence[str], learning_rate: float,
            batch_size: int, seeds: Sequence[int], execute: bool, val_only: bool) -> int:
    return int(runner(_step2_argv(
        runs_dir, splits_dir, models, learning_rate, batch_size, seeds, execute, val_only,
    )))


def _launch_grid(runner: Runner, runs_dir: str, splits_dir: str, execute: bool) -> int:
    for learning_rate, batch_size in iter_recipes():
        code = _launch(
            runner, runs_dir, splits_dir, TUNING_MODELS, learning_rate, batch_size,
            (SELECTION_SEED,), execute, val_only=True,
        )
        if code != 0:
            return code
    return 0


def _launch_final(runner: Runner, runs_dir: str, splits_dir: str, chosen: Mapping[str, Mapping[str, Any]],
                  execute: bool) -> int:
    for model in TUNING_MODELS:
        row = chosen[model]
        code = _launch(
            runner, runs_dir, splits_dir, (model,),
            row["learning_rate"], row["batch_size"], FINAL_SEEDS, execute, val_only=False,
        )
        if code != 0:
            return code
    return 0


def _print_missing(exc: IncompleteRuns) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for path in exc.missing:
        print(f"error: missing completed run: {path}", file=sys.stderr)


def _is_step2_output(out_dir: str) -> bool:
    target = os.path.realpath(out_dir)
    step2 = os.path.realpath(os.path.join(REPO_ROOT, "results", "step2"))
    try:
        return os.path.commonpath([target, step2]) == step2
    except ValueError:
        return False


def _plan_final_if_ready(runner: Runner, selection_dir: str, final_dir: str, out_dir: str, splits_dir: str) -> int:
    try:
        rows = collect_selection_rows(selection_dir, out_dir)
    except IncompleteRuns:
        print(
            "final phase: retrain seeds 42, 100 and 2026 for the selected recipe of each model, "
            "then score the PhraseBank test split and SEntFiN. "
            "The validation-only grid is not complete, so those directories are not listed yet."
        )
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    chosen = select_best(rows)
    _announce(chosen)
    return _launch_final(runner, final_dir, splits_dir, chosen, execute=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a learning rate and batch size per encoder on validation Macro-F1, then rerun the winners."
    )
    parser.add_argument("--phase", choices=("select", "final", "all"), default="all")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Train pending runs. Without it, the plan is printed and nothing is written.")
    parser.add_argument("--write-tables", action="store_true", default=False,
                        help="Rebuild the CSVs from completed result.json files without training.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, runner: Runner | None = None) -> int:
    args = parse_args(argv)
    if _is_step2_output(args.out_dir):
        print(
            f"error: refusing to write tuning outputs into {args.out_dir}; "
            f"use {DEFAULT_OUT_DIR} so results/step2 is left untouched.",
            file=sys.stderr,
        )
        return 2
    runner = runner or run_step2.main
    out_dir = args.out_dir
    selection_dir = os.path.join(out_dir, SELECTION_DIRNAME)
    final_dir = os.path.join(out_dir, FINAL_DIRNAME)
    train = bool(args.execute)
    write = bool(args.execute or args.write_tables)
    do_select = args.phase in ("select", "all")
    do_final = args.phase in ("final", "all")

    print(
        "tuning: sentences_allagree; models bert_base, finbert_mlm; "
        "learning rates 2e-5, 3e-5; batch sizes 16, 32; "
        "select on seed 42 validation macro-F1 with no test or SEntFiN scoring; "
        "final stage retrains seeds 42, 100, 2026 and then scores both holdouts."
    )

    if do_select and (train or not write):
        code = _launch_grid(runner, selection_dir, args.splits_dir, execute=train)
        if code != 0:
            return code

    if not write:
        if do_final:
            return _plan_final_if_ready(runner, selection_dir, final_dir, out_dir, args.splits_dir)
        print(
            "Plan only; pass --execute to train. "
            "Selection scores validation only. Test and SEntFiN are scored in the final stage."
        )
        return 0

    try:
        rows = collect_selection_rows(selection_dir, out_dir)
    except IncompleteRuns as exc:
        _print_missing(exc)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    chosen = select_best(rows)
    _write_selection(out_dir, rows, chosen)
    _announce(chosen)
    if not do_final:
        return 0

    if train:
        code = _launch_final(runner, final_dir, args.splits_dir, chosen, execute=True)
        if code != 0:
            return code
    try:
        final_rows = collect_final_rows(final_dir, out_dir, chosen)
    except IncompleteRuns as exc:
        _print_missing(exc)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _write_final(out_dir, final_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
