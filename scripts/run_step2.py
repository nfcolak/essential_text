#!/usr/bin/env python3
"""Step 2 run matrix: baselines, controlled fine-tuning, 5-fold CV and zero-shot scoring.

Plan-only by default; `--execute` runs the matrix. Every run is written atomically to
`<out-dir>/<config>/<model_key>/<tag>/` (`result.json` + `predictions.npz`), completed runs
are skipped on resume, and an error in one run is journalled without stopping the others.
This script never downloads anything: models come from the Hugging Face cache and the
datasets from `--splits-dir`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.io import (
    CONFIGS,
    cv_fold_arrays,
    file_sha256,
    load_external,
    load_manifest,
    split_arrays,
)
from src.evaluation.metrics import compute_classification_metrics

DEFAULT_OUT_DIR = "results/step2/runs"
DEFAULT_SPLITS_DIR = "data/splits"
DEFAULT_SEEDS = (42, 100, 2026)
BASELINE_SEED = 42
EXTERNAL_DIR = "external_only"
EXTERNAL_FILE = "sentfin_dedup.json"
SHA256SUMS_FILE = "SHA256SUMS"

STAGES = ("baselines", "finetune", "cv", "zeroshot", "all")

# model key -> descriptive model name (the factory is src.baselines.registry.make_baseline(key, seed))
BASELINE_KEYS = {
    "vader_official": "VADER (official vaderSentiment lexicon)",
    "lm_official": "Loughran-McDonald (official Master Dictionary)",
    "vader": "VADER",
    "loughran_mcdonald": "Loughran-McDonald",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
}
# Mirrors design.md section 1 (src/transformer/finetune.py MODEL_REGISTRY and friends)
FINETUNE_KEYS = {
    "bert_base": "google-bert/bert-base-uncased",
    "finbert_mlm": "yiyanghkust/finbert-pretrain",
}
ZERO_SHOT_KEYS = {"prosus_finbert": "ProsusAI/finbert"}
MODEL_REVISIONS = {
    "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "yiyanghkust/finbert-pretrain": "88ab954a39ea6d3ce2b62cff086dd5ad1172c664",
    "ProsusAI/finbert": "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
}
CV_KEYS = tuple(BASELINE_KEYS) + tuple(FINETUNE_KEYS)
ALL_KEYS = tuple(BASELINE_KEYS) + tuple(FINETUNE_KEYS) + tuple(ZERO_SHOT_KEYS)
CV_CONFIG = "sentences_allagree"
CV_FOLDS = (0, 1, 2, 3, 4)


@dataclass
class _FallbackConfig:
    """Same fields as `FinetuneConfig`; used only if src.transformer.finetune cannot be imported."""

    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 5
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 42
    device: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_config(seed: int, max_epochs: int | None) -> Any:
    try:
        from src.transformer.finetune import FinetuneConfig as config_cls
    except ImportError:
        config_cls = _FallbackConfig
    config = config_cls(seed=seed)
    if max_epochs is not None:
        config.max_epochs = int(max_epochs)
    return config


@dataclass
class Run:
    stage: str
    config: str
    model_key: str
    tag: str
    seed: int
    fold: int | None = None

    def run_dir(self, out_dir: str) -> str:
        return os.path.join(out_dir, self.config, self.model_key, self.tag)


def build_plan(
    stage: str,
    configs: Sequence[str],
    models: Sequence[str] | None,
    seeds: Sequence[int],
) -> list[Run]:
    """Return the ordered run matrix (baselines, finetune, cv, zeroshot)."""

    def keys_for(valid: Sequence[str]) -> list[str]:
        return [k for k in valid if models is None or k in models]

    stages = ("baselines", "finetune", "cv", "zeroshot") if stage == "all" else (stage,)
    runs: list[Run] = []
    for st in stages:
        if st == "baselines":
            for cfg in configs:
                for key in keys_for(tuple(BASELINE_KEYS)):
                    runs.append(Run("baselines", cfg, key, f"seed{BASELINE_SEED}", BASELINE_SEED))
        elif st == "finetune":
            for cfg in configs:
                for key in keys_for(tuple(FINETUNE_KEYS)):
                    for seed in seeds:
                        runs.append(Run("finetune", cfg, key, f"seed{seed}", int(seed)))
        elif st == "cv":
            if CV_CONFIG in configs:
                for key in keys_for(CV_KEYS):
                    for fold in CV_FOLDS:
                        runs.append(Run("cv", CV_CONFIG, key, f"cv_fold{fold}", BASELINE_SEED, fold))
        elif st == "zeroshot":
            for key in keys_for(tuple(ZERO_SHOT_KEYS)):
                runs.append(Run("zeroshot", EXTERNAL_DIR, key, "zeroshot", BASELINE_SEED))
    return runs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version(), "numpy": np.__version__}
    for label, module in (("sklearn", "sklearn"), ("torch", "torch"), ("transformers", "transformers")):
        try:
            out[label] = __import__(module).__version__
        except Exception:
            out[label] = None
    return out


def _is_completed(run_dir: str) -> bool:
    path = os.path.join(run_dir, "result.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _journal(out_dir: str, event: str, run_dir: str, wall_seconds: float | None, error: str | None) -> None:
    line = {
        "event": event,
        "run_dir": run_dir,
        "wall_seconds": wall_seconds,
        "timestamp": _now(),
        "error": error,
    }
    with open(os.path.join(out_dir, "journal.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def _limit_train(
    texts: list[str], labels: list[int], limit: int | None, seed: int
) -> tuple[list[str], list[int]]:
    if limit is None or limit >= len(texts):
        return texts, labels
    order = np.random.default_rng(seed).permutation(len(texts))[: int(limit)]
    return [texts[i] for i in order], [labels[i] for i in order]


def _check_prob(prob: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float32)
    if arr.shape != (n, 3):
        raise ValueError(f"{name} probabilities have shape {arr.shape}, expected ({n}, 3).")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} probabilities contain non-finite values (NaN or inf).")
    return arr


def _check_not_degenerate(prob: np.ndarray, name: str) -> None:
    """A trained transformer that predicts one class for every validation sentence has collapsed."""
    if len(np.unique(np.argmax(prob, axis=1))) < 2:
        raise ValueError(f"{name} predictions are degenerate: fewer than 2 distinct argmax classes.")


def _read_sha256sums(path: str) -> dict[str, str]:
    """Parse a `shasum -a 256` file into {file name: hex digest}."""
    sums: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(None, 1)
            if len(parts) == 2:
                sums[os.path.basename(parts[1].strip().lstrip("*"))] = parts[0].lower()
    return sums


def verify_split_hashes(paths: Sequence[str], splits_dir: str) -> list[str]:
    """Return the input files whose SHA-256 differs from `<splits_dir>/SHA256SUMS`.

    Without a SHA256SUMS file nothing is checked (a warning is printed); files it does not list are skipped.
    """
    sums_path = os.path.join(splits_dir, SHA256SUMS_FILE)
    if not os.path.isfile(sums_path):
        print(f"warning: {sums_path} not found; split files were not hash-checked.", file=sys.stderr)
        return []
    expected = _read_sha256sums(sums_path)
    return [
        path for path in paths
        if os.path.basename(path) in expected and file_sha256(path) != expected[os.path.basename(path)]
    ]


class _Data:
    """Lazily loaded manifests and external holdout for the runs of one invocation."""

    def __init__(self, splits_dir: str):
        self.splits_dir = splits_dir
        self._manifests: dict[str, dict[str, Any]] = {}
        self._external: tuple[list[str], list[str], list[int]] | None = None

    def manifest_path(self, config: str) -> str:
        return os.path.join(self.splits_dir, f"{config}_manifest.json")

    @property
    def external_path(self) -> str:
        return os.path.join(self.splits_dir, EXTERNAL_FILE)

    def manifest(self, config: str) -> dict[str, Any]:
        if config not in self._manifests:
            self._manifests[config] = load_manifest(self.manifest_path(config))
        return self._manifests[config]

    def external(self) -> tuple[list[str], list[str], list[int]]:
        if self._external is None:
            self._external = load_external(self.external_path)
        return self._external


def _metrics(y: Sequence[int], prob: np.ndarray) -> dict[str, Any]:
    return compute_classification_metrics(list(y), np.argmax(prob, axis=1), prob)


def _execute_run(
    run: Run,
    data: _Data,
    args: argparse.Namespace,
    finetune_fn: Callable[..., dict[str, Any]],
    zero_shot_fn: Callable[..., Any],
    baseline_factory: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one matrix entry; return (result dict, predictions.npz arrays)."""
    started = time.time()
    ids: dict[str, list[str]] = {}
    ys: dict[str, list[int]] = {}
    probs: dict[str, np.ndarray] = {}
    hyperparams: dict[str, Any] = {}
    best_epoch: int | None = None
    history: list[Any] = []
    device: str | None = None
    manifest_sha: str | None = None
    external_sha: str | None = None
    n_train: int | None = None
    n_val: int | None = None
    n_test: int | None = None
    n_ext: int | None = None

    if run.stage == "zeroshot":
        ext_ids, ext_texts, ext_y = data.external()
        external_sha = file_sha256(data.external_path)
        model_name = ZERO_SHOT_KEYS[run.model_key]
        config = _make_config(run.seed, None)
        full_config = config.to_dict()
        prob = zero_shot_fn(model_name, ext_texts, config, revision=MODEL_REVISIONS.get(model_name))
        ids["external"], ys["external"] = ext_ids, ext_y
        probs["external"] = _check_prob(prob, len(ext_texts), "external")
        # inference only: training hyperparameters would be misleading here
        hyperparams = {k: full_config.get(k) for k in ("max_length", "eval_batch_size", "device")}
        n_ext = len(ext_texts)
    else:
        manifest = data.manifest(run.config)
        manifest_sha = file_sha256(data.manifest_path(run.config))
        te_ids: list[str] = []
        te_texts: list[str] = []
        te_y: list[int] = []
        ext_ids, ext_texts, ext_y = [], [], []
        if run.stage == "cv":
            _, tr_texts, tr_y = cv_fold_arrays(manifest, run.fold, "train")
            val_ids, va_texts, va_y = cv_fold_arrays(manifest, run.fold, "val")
        else:
            _, tr_texts, tr_y = split_arrays(manifest, "train")
            val_ids, va_texts, va_y = split_arrays(manifest, "val")
            te_ids, te_texts, te_y = split_arrays(manifest, "test")
            ext_ids, ext_texts, ext_y = data.external()
            external_sha = file_sha256(data.external_path)
            n_test, n_ext = len(te_texts), len(ext_texts)
        tr_texts, tr_y = _limit_train(tr_texts, tr_y, args.limit_train, run.seed)
        n_train, n_val = len(tr_texts), len(va_texts)
        ids["val"], ys["val"] = val_ids, va_y

        if run.model_key in BASELINE_KEYS:
            model = baseline_factory(run.model_key, seed=run.seed)
            model.fit(tr_texts, tr_y, val_texts=va_texts, val_labels=va_y)
            probs["val"] = _check_prob(model.predict_proba(va_texts), n_val, "val")
            hyperparams = dict(getattr(model, "best_params_", {}) or {})
            if run.stage == "baselines":
                probs["test"] = _check_prob(model.predict_proba(te_texts), n_test, "test")
                probs["external"] = _check_prob(model.predict_proba(ext_texts), n_ext, "external")
        else:
            model_name = FINETUNE_KEYS[run.model_key]
            config = _make_config(run.seed, args.max_epochs)
            eval_sets: dict[str, list[str]] = {}
            if run.stage == "finetune":
                eval_sets = {"test": te_texts, "external": ext_texts}
            out = finetune_fn(
                model_name, tr_texts, tr_y, va_texts, va_y, eval_sets, config,
                revision=MODEL_REVISIONS.get(model_name),
            )
            probs["val"] = _check_prob(out["val_prob"], n_val, "val")
            _check_not_degenerate(probs["val"], "val")
            if run.stage == "finetune":
                probs["test"] = _check_prob(out["eval_probs"]["test"], n_test, "test")
                probs["external"] = _check_prob(out["eval_probs"]["external"], n_ext, "external")
            hyperparams = dict(out.get("config") or config.to_dict())
            best_epoch = out.get("best_epoch")
            history = list(out.get("history") or [])
            device = out.get("device")
        if "test" in probs:
            ids["test"], ys["test"] = te_ids, te_y
            ids["external"], ys["external"] = ext_ids, ext_y

    arrays: dict[str, np.ndarray] = {}
    for split in ("val", "test", "external"):
        if split in probs:
            arrays[f"{split}_ids"] = np.array(ids[split], dtype=str)
            arrays[f"{split}_y"] = np.asarray(ys[split], dtype=np.int64)
            arrays[f"{split}_prob"] = probs[split]

    result = {
        "status": "completed",
        "stage": run.stage,
        "config": run.config,
        "model_key": run.model_key,
        "model_name": (
            BASELINE_KEYS.get(run.model_key)
            or FINETUNE_KEYS.get(run.model_key)
            or ZERO_SHOT_KEYS[run.model_key]
        ),
        "seed": run.seed,
        "fold": run.fold,
        "hyperparams": hyperparams,
        "best_epoch": best_epoch,
        "history": history,
        "metrics": {s: (_metrics(ys[s], probs[s]) if s in probs else None) for s in ("val", "test", "external")},
        "n": {"train": n_train, "val": n_val, "test": n_test, "external": n_ext},
        "device": device,
        "versions": _versions(),
        "manifest_sha256": manifest_sha,
        "external_sha256": external_sha,
        "probe": args.limit_train is not None or args.max_epochs is not None,
        "wall_seconds": time.time() - started,
        "timestamp": _now(),
    }
    return result, arrays


def _write_run(run_dir: str, result: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    tmp_dir = run_dir + ".tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    try:
        with open(os.path.join(tmp_dir, "result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=_jsonable)
        np.savez_compressed(os.path.join(tmp_dir, "predictions.npz"), **arrays)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        os.replace(tmp_dir, run_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step 2 baseline / transformer run matrix.")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), metavar="CFG")
    parser.add_argument("--models", nargs="+", default=None, metavar="KEY")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="INT")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Run the matrix. Without it the plan is printed and nothing is written.")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Probe only: first N train rows after a seeded shuffle.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Probe only: override the maximum epochs.")
    return parser.parse_args(argv)


def _default_finetune_fn() -> Callable[..., dict[str, Any]]:
    from src.transformer.finetune import finetune_and_predict

    return finetune_and_predict


def _default_zero_shot_fn() -> Callable[..., Any]:
    from src.transformer.finetune import zero_shot_predict

    return zero_shot_predict


def _default_baseline_factory() -> Callable[..., Any]:
    from src.baselines.registry import make_baseline

    return make_baseline


def _free_memory() -> None:
    """Release Python and accelerator caches between runs (MPS keeps freed blocks cached otherwise)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(
    argv: Sequence[str] | None = None,
    finetune_fn: Callable[..., dict[str, Any]] | None = None,
    zero_shot_fn: Callable[..., Any] | None = None,
    baseline_factory: Callable[..., Any] | None = None,
) -> int:
    args = parse_args(argv)

    unknown_cfgs = [c for c in args.configs if c not in CONFIGS]
    if unknown_cfgs:
        print(f"error: unknown config(s) {unknown_cfgs}; valid: {list(CONFIGS)}", file=sys.stderr)
        return 2
    if args.models is not None:
        unknown = [m for m in args.models if m not in ALL_KEYS]
        if unknown:
            print(f"error: unknown model key(s) {unknown}; valid: {list(ALL_KEYS)}", file=sys.stderr)
            return 2

    probe = args.limit_train is not None or args.max_epochs is not None
    if probe and os.path.abspath(args.out_dir) == os.path.abspath(DEFAULT_OUT_DIR):
        print(
            "error: --limit-train/--max-epochs are probe flags and must not write into the default "
            f"--out-dir ({DEFAULT_OUT_DIR}); pass a different --out-dir.",
            file=sys.stderr,
        )
        return 2

    plan = build_plan(args.stage, args.configs, args.models, args.seeds)

    data = _Data(args.splits_dir)
    needed = [data.manifest_path(cfg) for cfg in sorted({r.config for r in plan if r.config != EXTERNAL_DIR})]
    if any(r.stage != "cv" for r in plan):
        needed.append(data.external_path)
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        for path in missing:
            print(f"error: missing input file: {path}", file=sys.stderr)
        return 2

    mismatched = verify_split_hashes(needed, args.splits_dir)
    if mismatched:
        for path in mismatched:
            print(f"error: SHA-256 of {path} does not match {SHA256SUMS_FILE}.", file=sys.stderr)
        return 2

    pending = 0
    for run in plan:
        run_dir = run.run_dir(args.out_dir)
        status = "completed" if _is_completed(run_dir) else "pending"
        pending += status == "pending"
        print(f"[{run.stage}] {run_dir} {status}")
    print(f"plan: {len(plan)} runs, {pending} pending, {len(plan) - pending} completed")

    if not args.execute:
        print("Plan only; pass --execute to run the pending runs.")
        return 0

    finetune_fn = finetune_fn or _default_finetune_fn()
    zero_shot_fn = zero_shot_fn or _default_zero_shot_fn()
    baseline_factory = baseline_factory or _default_baseline_factory()

    os.makedirs(args.out_dir, exist_ok=True)
    errors = 0
    for run in plan:
        run_dir = run.run_dir(args.out_dir)
        if _is_completed(run_dir):
            continue
        stale = run_dir + ".tmp"
        if os.path.isdir(stale):
            shutil.rmtree(stale)
        _journal(args.out_dir, "start", run_dir, None, None)
        started = time.time()
        try:
            result, arrays = _execute_run(run, data, args, finetune_fn, zero_shot_fn, baseline_factory)
            os.makedirs(os.path.dirname(run_dir), exist_ok=True)
            _write_run(run_dir, result, arrays)
            _journal(args.out_dir, "end", run_dir, time.time() - started, None)
            print(f"done {run_dir} ({time.time() - started:.1f}s)")
        except Exception as exc:  # per-run isolation: log and continue
            errors += 1
            message = f"{type(exc).__name__}: {exc}"
            _journal(args.out_dir, "error", run_dir, time.time() - started, message)
            print(f"ERROR {run_dir}: {message}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        _free_memory()
    if errors:
        print(f"{errors} run(s) failed; see {os.path.join(args.out_dir, 'journal.jsonl')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
