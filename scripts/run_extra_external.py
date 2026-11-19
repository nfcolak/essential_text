#!/usr/bin/env python3
"""Rerun the Step 2 matrix (seed runs and zero-shot, no CV) and score two extra external test sets.

The Step 2 fine-tuned models were not saved, so scoring a new test set means training them again. Every run here
uses exactly the Step 2 code path: the same locked splits (checked against ``SHA256SUMS``), the same baseline
factory, the same ``finetune_and_predict`` recipe and seeds, and the same SEntFiN holdout. The only change is that
two more evaluation sets are scored after training: FiQA 2018 and Twitter Financial News
(``data/splits/extra_external.json``, built by ``scripts/prepare_extra_external.py`` and checked against
``SHA256SUMS.extra``). Adding evaluation sets does not touch training (they are scored after the best epoch is
restored).

Each run also records how closely it reproduces the stored Step 2 run (``results/step2/runs``): the share of test /
SEntFiN sentences with the same predicted label and the largest probability difference.

Plan-only by default. ``--only N`` runs plan entry N (one model per process: the queue script calls this once per
run, which keeps Apple MPS memory flat). Completed runs are skipped. Output:
``<out-dir>/<config>/<model_key>/<tag>/{result.json, predictions.npz}`` with splits
``val, test, external, fiqa, twitter``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts import run_step2 as rs  # noqa: E402
from scripts.prepare_extra_external import EXTRA_SHA256SUMS_FILE  # noqa: E402
from scripts.prepare_extra_external import OUT_FILE as EXTRA_FILE  # noqa: E402
from scripts.prepare_extra_external import load_extra_external  # noqa: E402
from src.data.io import file_sha256, split_arrays  # noqa: E402

EXTRA_SETS = ("fiqa", "twitter")
SPLITS = ("val", "test", "external") + EXTRA_SETS
DEFAULT_OUT_DIR = os.path.join("results", "extensions", "runs")
REFERENCE_DIR = os.path.join("results", "step2", "runs")


DAPT_KEY = "bert_dapt"


def build_plan(dapt_model: str | None = None) -> list[rs.Run]:
    """Step 2 baselines, fine-tune seeds and zero-shot, in that order (CV is not rerun).

    With ``dapt_model`` the 9 fine-tune runs of the controlled DAPT arm (``bert_dapt``) are appended, so the indices
    of the first 43 runs never change.
    """
    plan = rs.build_plan("baselines", rs.CONFIGS, None, rs.DEFAULT_SEEDS)
    plan += rs.build_plan("finetune", rs.CONFIGS, None, rs.DEFAULT_SEEDS)
    plan += rs.build_plan("zeroshot", rs.CONFIGS, None, rs.DEFAULT_SEEDS)
    if dapt_model:
        plan += [rs.Run("finetune", cfg, DAPT_KEY, f"seed{s}", int(s)) for cfg in rs.CONFIGS for s in rs.DEFAULT_SEEDS]
    return plan


def reproduction(run: rs.Run, probs: dict[str, np.ndarray], ids: dict[str, list[str]],
                 reference_dir: str) -> dict[str, Any] | None:
    """Agreement with the stored Step 2 run of the same config / model / tag (None when it does not exist)."""
    path = os.path.join(run.run_dir(reference_dir), "predictions.npz")
    if not os.path.isfile(path):
        return None
    out: dict[str, Any] = {"reference": os.path.dirname(path)}
    with np.load(path, allow_pickle=False) as ref:
        for split in ("val", "test", "external"):
            if split not in probs or f"{split}_prob" not in ref.files:
                continue
            ref_ids = np.asarray(ref[f"{split}_ids"]).astype(str)
            if not np.array_equal(ref_ids, np.asarray(ids[split], dtype=str)):
                out[split] = {"error": "id order differs"}
                continue
            a, b = probs[split], np.asarray(ref[f"{split}_prob"])
            out[split] = {"label_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
                          "max_abs_prob_diff": float(np.max(np.abs(a - b)))}
    return out


def execute(run: rs.Run, data: rs._Data, extra_path: str,
            dapt_model: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.time()
    extra = {name: load_extra_external(extra_path, name) for name in EXTRA_SETS}
    ext_ids, ext_texts, ext_y = data.external()
    ids: dict[str, list[str]] = {"external": ext_ids}
    ys: dict[str, list[int]] = {"external": ext_y}
    texts: dict[str, list[str]] = {"external": ext_texts}
    for name, (i, t, y) in extra.items():
        ids[name], texts[name], ys[name] = i, t, y
    probs: dict[str, np.ndarray] = {}
    hyperparams: dict[str, Any] = {}
    best_epoch, history, device, manifest_sha, n_train = None, [], None, None, None

    if run.stage == "zeroshot":
        from src.transformer.finetune import zero_shot_predict

        model_name = rs.ZERO_SHOT_KEYS[run.model_key]
        config = rs._make_config(run.seed, None)
        for split in ("external",) + EXTRA_SETS:
            prob = zero_shot_predict(model_name, texts[split], config, revision=rs.MODEL_REVISIONS.get(model_name))
            probs[split] = rs._check_prob(prob, len(texts[split]), split)
        full = config.to_dict()
        hyperparams = {k: full.get(k) for k in ("max_length", "eval_batch_size", "device")}
    else:
        manifest = data.manifest(run.config)
        manifest_sha = file_sha256(data.manifest_path(run.config))
        _, tr_texts, tr_y = split_arrays(manifest, "train")
        ids["val"], texts["val"], ys["val"] = split_arrays(manifest, "val")
        ids["test"], texts["test"], ys["test"] = split_arrays(manifest, "test")
        n_train = len(tr_texts)
        eval_names = ("test", "external") + EXTRA_SETS
        if run.model_key in rs.BASELINE_KEYS:
            from src.baselines.registry import make_baseline

            model = make_baseline(run.model_key, seed=run.seed)
            model.fit(tr_texts, tr_y, val_texts=texts["val"], val_labels=ys["val"])
            for split in ("val",) + eval_names:
                probs[split] = rs._check_prob(model.predict_proba(texts[split]), len(texts[split]), split)
            hyperparams = dict(getattr(model, "best_params_", {}) or {})
        else:
            from src.transformer.finetune import finetune_and_predict

            if run.model_key == DAPT_KEY:
                if not dapt_model:
                    raise ValueError("bert_dapt runs need --dapt-model")
                model_name, revision = dapt_model, None
            else:
                model_name = rs.FINETUNE_KEYS[run.model_key]
                revision = rs.MODEL_REVISIONS.get(model_name)
            config = rs._make_config(run.seed, None)
            out = finetune_and_predict(model_name, tr_texts, tr_y, texts["val"], ys["val"],
                                       {name: texts[name] for name in eval_names}, config,
                                       revision=revision)
            probs["val"] = rs._check_prob(out["val_prob"], len(texts["val"]), "val")
            rs._check_not_degenerate(probs["val"], "val")
            for split in eval_names:
                probs[split] = rs._check_prob(out["eval_probs"][split], len(texts[split]), split)
            hyperparams = dict(out.get("config") or config.to_dict())
            best_epoch, history, device = out.get("best_epoch"), list(out.get("history") or []), out.get("device")

    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if split in probs:
            arrays[f"{split}_ids"] = np.array(ids[split], dtype=str)
            arrays[f"{split}_y"] = np.asarray(ys[split], dtype=np.int64)
            arrays[f"{split}_prob"] = probs[split]
    result = {
        "status": "completed",
        "stage": run.stage,
        "config": run.config,
        "model_key": run.model_key,
        "model_name": rs.BASELINE_KEYS.get(run.model_key) or rs.FINETUNE_KEYS.get(run.model_key)
        or rs.ZERO_SHOT_KEYS.get(run.model_key) or "bert-base-uncased + continued MLM on 10-K text (DAPT)",
        "dapt_model_sha256": (file_sha256(os.path.join(dapt_model, "model.safetensors"))
                              if run.model_key == DAPT_KEY and dapt_model else None),
        "seed": run.seed,
        "fold": None,
        "hyperparams": hyperparams,
        "best_epoch": best_epoch,
        "history": history,
        "metrics": {s: (rs._metrics(ys[s], probs[s]) if s in probs else None) for s in SPLITS},
        "n": {"train": n_train, **{s: (len(ys[s]) if s in probs else None) for s in SPLITS}},
        "device": device,
        "versions": rs._versions(),
        "manifest_sha256": manifest_sha,
        "external_sha256": file_sha256(data.external_path),
        "extra_external_sha256": file_sha256(extra_path),
        "reproduction_of_step2": reproduction(run, probs, ids, REFERENCE_DIR),
        "probe": False,
        "wall_seconds": time.time() - started,
        "timestamp": rs._now(),
    }
    return result, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits-dir", default=rs.DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", type=int, default=None, help="run only plan entry N (0-based)")
    parser.add_argument("--dapt-model", default=None,
                        help="local checkpoint dir of the DAPT model; appends the 9 bert_dapt runs (indices 43-51)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(args.dapt_model)
    data = rs._Data(args.splits_dir)
    extra_path = os.path.join(args.splits_dir, EXTRA_FILE)
    step1 = [data.manifest_path(c) for c in rs.CONFIGS] + [data.external_path]
    # The Step 1 inputs are locked in SHA256SUMS; the extension file has its own SHA256SUMS.extra.
    checks = ((rs.SHA256SUMS_FILE, step1), (EXTRA_SHA256SUMS_FILE, [extra_path]))
    missing = [p for _, paths in checks for p in paths if not os.path.isfile(p)]
    if missing:
        print(f"error: missing input files: {missing}", file=sys.stderr)
        return 2
    for sums_name, paths in checks:
        sums_path = os.path.join(args.splits_dir, sums_name)
        if not os.path.isfile(sums_path):
            print(f"error: missing checksum file: {sums_path}", file=sys.stderr)
            return 2
        sums = rs._read_sha256sums(sums_path)
        unlisted = [p for p in paths if os.path.basename(p) not in sums]
        mismatched = [p for p in paths if p not in unlisted and file_sha256(p) != sums[os.path.basename(p)]]
        if unlisted or mismatched:
            print(f"error: not in {sums_name} {unlisted} / SHA-256 mismatch against {sums_name} {mismatched}",
                  file=sys.stderr)
            return 2

    selected = list(enumerate(plan)) if args.only is None else [(args.only, plan[args.only])]
    for k, run in selected:
        status = "completed" if rs._is_completed(run.run_dir(args.out_dir)) else "pending"
        print(f"{k:2d} [{run.stage}] {run.run_dir(args.out_dir)} {status}")
    if not args.execute:
        print(f"plan: {len(plan)} runs; plan only, pass --execute (and --only N for one run).")
        return 0

    errors = 0
    os.makedirs(args.out_dir, exist_ok=True)
    for _, run in selected:
        run_dir = run.run_dir(args.out_dir)
        if rs._is_completed(run_dir):
            continue
        rs._journal(args.out_dir, "start", run_dir, None, None)
        started = time.time()
        try:
            result, arrays = execute(run, data, extra_path, args.dapt_model)
            os.makedirs(os.path.dirname(run_dir), exist_ok=True)
            rs._write_run(run_dir, result, arrays)
            rs._journal(args.out_dir, "end", run_dir, time.time() - started, None)
            print(f"done {run_dir} ({time.time() - started:.1f}s)")
        except Exception as exc:
            errors += 1
            rs._journal(args.out_dir, "error", run_dir, time.time() - started, f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)
        rs._free_memory()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
