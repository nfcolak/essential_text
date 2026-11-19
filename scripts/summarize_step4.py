#!/usr/bin/env python3
"""Paired statistics for Step 4: tuned recipe against the fixed Step 2 recipe.

Reads the selected recipe of ``bert_base`` and ``finbert_mlm`` from ``<tuning>/selection.json``, the
final runs under ``<tuning>/final/sentences_allagree/<model>/`` and the fixed-recipe runs under
``<step2-runs>/sentences_allagree/<model>/seed<N>``. Fixed runs can also come from ``--fixed-extra-dir``.
Only stored predictions are read. The default seeds are 42, 100, and 2026.

For the PhraseBank test split and the external SEntFiN split it reports

* Macro-F1 per seed, seed mean, sample sd and mean accuracy for every model x recipe;
* four paired comparisons: tuned BERT vs tuned FinBERT, fixed BERT vs fixed FinBERT, tuned vs fixed
  BERT, tuned vs fixed FinBERT. Each has the Macro-F1 difference of the seed means, the per-seed
  differences, a 95% interval and a two-sided p from ``seed_sentence_bootstrap`` (seeds and sentences
  resampled), and McNemar per seed pairing with Holm-Bonferroni inside each seed family over the
  8 comparisons (4 pairs x 2 splits);
* the gap change ``(tuned BERT - tuned FinBERT) - (fixed BERT - fixed FinBERT)`` with an interval
  from one bootstrap: sentences shared, seeds of the four run sets resampled independently.

Writes ``step4_statistics.json`` and ``STEP4_REPORT.md`` into ``--out-dir`` (with an optional
``--suffix`` before each extension). ``--check`` recomputes
and compares with the existing JSON without writing. Exit codes: 0 ok, 1 ``--check`` mismatch,
2 refused (a run is missing, or ids / labels differ between runs).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_step2 import finetune_tag, format_learning_rate  # noqa: E402
from src.evaluation.multiseed import holm_bonferroni, mean_sd, seed_sentence_bootstrap  # noqa: E402
from src.evaluation.stats import mcnemar_test  # noqa: E402

CONFIG = "sentences_allagree"
MODELS = ("bert_base", "finbert_mlm")
MODEL_LABELS = {"bert_base": "BERT", "finbert_mlm": "FinBERT"}
SEEDS = (42, 100, 2026)
SPLITS = ("test", "external")
SPLIT_LABELS = {"test": "PhraseBank test", "external": "SEntFiN (external)"}
RECIPES = ("fixed", "tuned")
ALPHA = 0.05
CHECK_TOLERANCE = 1e-12

DEFAULT_TUNING_DIR = "results/tuning"
DEFAULT_STEP2_RUNS = "results/step2/runs"
DEFAULT_FIXED_EXTRA_DIR = "results/tuning/fixed_extra"
DEFAULT_OUT_DIR = "results/step4"
JSON_NAME = "step4_statistics.json"
REPORT_NAME = "STEP4_REPORT.md"

# name -> ((recipe, model) of A, (recipe, model) of B); every difference is A - B
COMPARISONS = (
    ("tuned_bert_vs_tuned_finbert", ("tuned", "bert_base"), ("tuned", "finbert_mlm")),
    ("fixed_bert_vs_fixed_finbert", ("fixed", "bert_base"), ("fixed", "finbert_mlm")),
    ("tuned_vs_fixed_bert", ("tuned", "bert_base"), ("fixed", "bert_base")),
    ("tuned_vs_fixed_finbert", ("tuned", "finbert_mlm"), ("fixed", "finbert_mlm")),
)
GAP_TUNED, GAP_FIXED = "tuned_bert_vs_tuned_finbert", "fixed_bert_vs_fixed_finbert"


class Refuse(Exception):
    """Input that the summary must not be computed from (exit code 2)."""


# --------------------------------------------------------------------------- loading


def _rel(path: str) -> str:
    """Path relative to the repository when it lies inside it, else the absolute path."""
    absolute = os.path.abspath(path)
    relative = os.path.relpath(absolute, REPO_ROOT)
    if relative.startswith(".."):
        return absolute.replace(os.sep, "/")
    return relative.replace(os.sep, "/")


def read_selection(tuning_dir: str) -> dict[str, dict[str, Any]]:
    """Selected ``{model: {learning_rate, batch_size, val_macro_f1}}`` from ``selection.json``."""
    path = os.path.join(tuning_dir, "selection.json")
    if not os.path.isfile(path):
        raise Refuse(f"{path} not found: the tuning selection has not been written yet.")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(selected, dict):
        raise Refuse(f"{path} has no 'selected' object.")
    config = payload.get("phrasebank_config", CONFIG)
    if config != CONFIG:
        raise Refuse(f"{path} is for config {config!r}, expected {CONFIG!r}.")
    recipes: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        spec = selected.get(model)
        if not isinstance(spec, dict) or "learning_rate" not in spec or "batch_size" not in spec:
            raise Refuse(f"{path} has no selected learning_rate/batch_size for {model}.")
        recipes[model] = {
            "learning_rate": float(spec["learning_rate"]),
            "batch_size": int(spec["batch_size"]),
            "val_macro_f1": None if spec.get("val_macro_f1") is None else float(spec["val_macro_f1"]),
        }
    return recipes


def _load_run(run_dir: str, model: str, seed: int) -> dict[str, Any]:
    """One completed run: ``result.json`` plus ids, labels and argmax predictions of both splits."""
    result_path = os.path.join(run_dir, "result.json")
    npz_path = os.path.join(run_dir, "predictions.npz")
    if not os.path.isfile(result_path):
        raise Refuse(f"missing run for {model} seed {seed}: {result_path} not found.")
    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise Refuse(f"run for {model} seed {seed} is not completed: {result_path}.")
    if result.get("model_key", model) != model or int(result.get("seed", seed)) != int(seed):
        raise Refuse(f"{result_path} does not belong to {model} seed {seed}.")
    if not os.path.isfile(npz_path):
        raise Refuse(f"missing predictions for {model} seed {seed}: {npz_path} not found.")
    splits: dict[str, dict[str, np.ndarray]] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        names = set(data.files)
        for split in SPLITS:
            needed = [f"{split}_ids", f"{split}_y", f"{split}_prob"]
            absent = [name for name in needed if name not in names]
            if absent:
                raise Refuse(f"{npz_path} has no {', '.join(absent)}; the {split} split was not scored.")
            ids = np.asarray(data[f"{split}_ids"]).astype(str)
            y = np.asarray(data[f"{split}_y"], dtype=np.int64)
            prob = np.asarray(data[f"{split}_prob"], dtype=np.float64)
            if prob.ndim != 2 or prob.shape[1] != 3 or not (len(ids) == len(y) == len(prob)) or len(y) == 0:
                raise Refuse(f"{npz_path} has malformed {split} arrays.")
            splits[split] = {"ids": ids, "y": y, "pred": np.argmax(prob, axis=1).astype(np.int64)}
    return {"run_dir": run_dir, "result": result, "splits": splits}


def _recorded_recipe(result: Mapping[str, Any], path: str) -> tuple[float, int]:
    hyper = result.get("hyperparams") or {}
    try:
        return float(hyper["learning_rate"]), int(hyper["batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Refuse(f"{path} records no learning_rate/batch_size in hyperparams.") from exc


def load_inputs(tuning_dir: str, step2_runs: str, seeds: Sequence[int] = SEEDS,
                fixed_extra_dir: str = DEFAULT_FIXED_EXTRA_DIR) -> dict[str, Any]:
    """Load each requested seed for both models and recipes and verify comparability."""
    selection = read_selection(tuning_dir)
    runs: dict[str, dict[str, dict[int, dict[str, Any]]]] = {recipe: {} for recipe in RECIPES}
    recipes: dict[str, dict[str, dict[str, Any]]] = {recipe: {} for recipe in RECIPES}
    problems: list[str] = []
    for model in MODELS:
        spec = selection[model]
        runs["tuned"][model], runs["fixed"][model] = {}, {}
        for seed in seeds:
            tuned_dir = os.path.join(
                tuning_dir, "final", CONFIG, model, finetune_tag(seed, spec["learning_rate"], spec["batch_size"])
            )
            fixed_path = os.path.join(CONFIG, model, f"seed{seed}")
            fixed_dir = os.path.join(step2_runs, fixed_path)
            extra_dir = os.path.join(fixed_extra_dir, fixed_path)
            if not os.path.isfile(os.path.join(fixed_dir, "result.json")) and os.path.isdir(extra_dir):
                fixed_dir = extra_dir
            for recipe, run_dir in (("tuned", tuned_dir), ("fixed", fixed_dir)):
                try:
                    runs[recipe][model][seed] = _load_run(run_dir, model, seed)
                except Refuse as exc:
                    problems.append(str(exc))
    if problems:
        raise Refuse("\n".join(problems))

    for model in MODELS:
        spec = selection[model]
        want = (format_learning_rate(spec["learning_rate"]), spec["batch_size"])
        fixed_seen: set[tuple[str, int]] = set()
        for seed in seeds:
            run = runs["tuned"][model][seed]
            lr, bs = _recorded_recipe(run["result"], run["run_dir"])
            if (format_learning_rate(lr), bs) != want:
                raise Refuse(
                    f"{run['run_dir']} records learning_rate={lr} batch_size={bs}, "
                    f"but selection.json selected {spec['learning_rate']} and {spec['batch_size']} for {model}."
                )
            run = runs["fixed"][model][seed]
            lr, bs = _recorded_recipe(run["result"], run["run_dir"])
            fixed_seen.add((format_learning_rate(lr), bs))
        if len(fixed_seen) != 1:
            raise Refuse(f"the fixed-recipe runs of {model} do not share one learning rate and batch size.")
        lr_token, bs = next(iter(fixed_seen))
        recipes["fixed"][model] = {"learning_rate": float(lr_token), "batch_size": int(bs)}
        recipes["tuned"][model] = dict(spec)

    reference: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        first = runs["fixed"][MODELS[0]][seeds[0]]
        ref_ids, ref_y = first["splits"][split]["ids"], first["splits"][split]["y"]
        for recipe in RECIPES:
            for model in MODELS:
                for seed in seeds:
                    run = runs[recipe][model][seed]
                    part = run["splits"][split]
                    if part["ids"].shape != ref_ids.shape or not np.array_equal(part["ids"], ref_ids):
                        raise Refuse(
                            f"{split} ids differ between {first['run_dir']} and {run['run_dir']}; "
                            "the runs were not scored on the same sentences."
                        )
                    if not np.array_equal(part["y"], ref_y):
                        raise Refuse(
                            f"{split} labels (y) differ between {first['run_dir']} and {run['run_dir']}."
                        )
        if ref_y.min() < 0 or ref_y.max() > 2:
            raise Refuse(f"{split} labels are outside 0..2 in {first['run_dir']}.")
        reference[split] = {"ids": ref_ids, "y": ref_y}
    return {"runs": runs, "recipes": recipes, "reference": reference, "seeds": tuple(seeds)}


# --------------------------------------------------------------------------- statistics


def _confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(y * 3 + pred, minlength=9).reshape(3, 3).astype(np.float64)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted mean F1 over the three classes (0 for a class with no gold and no predicted row).

    Same definition as the bootstrap in ``src.evaluation.multiseed``.
    """
    cm = _confusion(y, pred)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return float(f1.mean())


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred))


def gap_change_draws(y: np.ndarray, run_sets: Sequence[Sequence[np.ndarray]], B: int, seed: int,
                     _share_recipe_seeds: bool = False) -> np.ndarray:
    """``B`` bootstrap draws of ``(TB - TF) - (FB - FF)`` from ONE bootstrap over four run sets.

    ``run_sets`` = (tuned BERT, tuned FinBERT, fixed BERT, fixed FinBERT), each a list with one
    prediction vector per seed. Per draw, one sentence resample (``n`` indices with replacement) is
    shared by all four sets, and every set gets its own seed resample (as many picks with replacement
    as it has runs). The score of a set is the mean over its picked seeds of Macro-F1 on the resampled
    sentences, the definition ``seed_sentence_bootstrap`` uses. The four seed resamples are independent:
    tuned and fixed runs with the same seed number are different trainings, so they are not paired.
    The stream is ``default_rng([seed, 2])``, separate from the streams of the pair intervals.

    ``_share_recipe_seeds`` is a diagnostic for the tests only: it reuses the tuned seed picks for the
    fixed set of the same model, which is the pairing this function must NOT apply.
    """
    if len(run_sets) != 4 or any(len(group) == 0 for group in run_sets):
        raise ValueError("run_sets must hold four non-empty lists of prediction vectors")
    n = len(y)
    sizes = [len(group) for group in run_sets]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    vectors = [np.asarray(p, dtype=np.int64) for group in run_sets for p in group]
    onehot = np.zeros((n, 9 * len(vectors)), dtype=np.float64)
    rows = np.arange(n)
    for j, pred in enumerate(vectors):
        onehot[rows, 9 * j + y * 3 + pred] = 1.0

    def scores_of(counts: np.ndarray) -> np.ndarray:
        cm = counts.reshape(counts.shape[0], len(vectors), 3, 3)
        tp = np.einsum("bmii->bmi", cm)
        denom = cm.sum(axis=2) + cm.sum(axis=3)
        return np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0).mean(axis=2)

    rng = np.random.default_rng([seed, 2])
    deltas = np.empty(B, dtype=np.float64)
    chunk = max(1, min(B, 250))
    done = 0
    while done < B:
        size = min(chunk, B - done)
        idx = rng.integers(0, n, size=(size, n))
        flat = (idx + (np.arange(size) * n)[:, None]).ravel()
        weights = np.bincount(flat, minlength=size * n).reshape(size, n).astype(np.float64)
        scores = scores_of(weights @ onehot)
        picks = [rng.integers(0, k, size=(size, k)) for k in sizes]
        if _share_recipe_seeds:
            picks[2], picks[3] = picks[0], picks[1]
        means = [
            np.take_along_axis(scores[:, offsets[g] : offsets[g + 1]], picks[g], axis=1).mean(axis=1)
            for g in range(4)
        ]
        deltas[done : done + size] = (means[0] - means[1]) - (means[2] - means[3])
        done += size
    return deltas


def _interval(deltas: np.ndarray) -> tuple[float, float, float]:
    """95% percentile interval and two-sided p, as in ``seed_sentence_bootstrap``."""
    low, high = np.percentile(deltas, [2.5, 97.5])
    tail = min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return float(low), float(high), float(min(1.0, 2.0 * tail))


def _contains_zero(low: float, high: float) -> bool:
    return bool(low <= 0.0 <= high)


def _environment(runs: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    """Device and library versions recorded by the runs, per recipe, and whether all runs agree."""
    out: dict[str, Any] = {}
    every: list[str] = []
    for recipe in RECIPES:
        seen: list[dict[str, Any]] = []
        for model in MODELS:
            for seed in seeds:
                result = runs[recipe][model][seed]["result"]
                entry = {"device": result.get("device"), "versions": result.get("versions") or {}}
                every.append(json.dumps(entry, sort_keys=True))
                if entry not in seen:
                    seen.append(entry)
        out[recipe] = seen
    out["identical_across_all_runs"] = len(set(every)) == 1
    return out


def build_statistics(inputs: Mapping[str, Any], B: int, seed: int) -> dict[str, Any]:
    runs, reference = inputs["runs"], inputs["reference"]
    seeds = inputs.get("seeds", SEEDS)

    def preds(recipe: str, model: str, split: str) -> list[np.ndarray]:
        return [runs[recipe][model][s]["splits"][split]["pred"] for s in seeds]

    scores: dict[str, Any] = {}
    stored_gap = 0.0
    for model in MODELS:
        scores[model] = {}
        for recipe in RECIPES:
            scores[model][recipe] = {}
            for split in SPLITS:
                y = reference[split]["y"]
                f1_by_seed, acc_by_seed = {}, {}
                for s, pred in zip(seeds, preds(recipe, model, split)):
                    f1_by_seed[str(s)] = macro_f1(y, pred)
                    acc_by_seed[str(s)] = accuracy(y, pred)
                    stored = ((runs[recipe][model][s]["result"].get("metrics") or {}).get(split) or {}).get("macro_f1")
                    if stored is not None:
                        stored_gap = max(stored_gap, abs(float(stored) - f1_by_seed[str(s)]))
                scores[model][recipe][split] = {
                    "macro_f1_by_seed": f1_by_seed,
                    "macro_f1": mean_sd(list(f1_by_seed.values())),
                    "accuracy_by_seed": acc_by_seed,
                    "accuracy": mean_sd(list(acc_by_seed.values())),
                }

    comparisons: dict[str, dict[str, Any]] = {split: {} for split in SPLITS}
    gap_change: dict[str, Any] = {}
    for split in SPLITS:
        y = reference[split]["y"]
        for name, (recipe_a, model_a), (recipe_b, model_b) in COMPARISONS:
            preds_a, preds_b = preds(recipe_a, model_a, split), preds(recipe_b, model_b, split)
            boot = seed_sentence_bootstrap(y, preds_a, preds_b, metric="macro_f1", B=B, seed=seed)
            per_seed = {}
            for s, pa, pb in zip(seeds, preds_a, preds_b):
                res = mcnemar_test(y, pa, pb)
                per_seed[str(s)] = {k: res[k] for k in ("n10", "n01", "chi2", "p_value")}
            comparisons[split][name] = {
                "a": {"recipe": recipe_a, "model": model_a},
                "b": {"recipe": recipe_b, "model": model_b},
                "delta_macro_f1": boot["delta"],
                "per_seed_delta": {str(s): d for s, d in zip(seeds, boot["per_seed_delta"])},
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_bootstrap": boot["p_value"],
                "interval_contains_zero": _contains_zero(boot["ci_low"], boot["ci_high"]),
                "mcnemar_by_seed": per_seed,
            }
        gap_draws = gap_change_draws(
            y,
            [preds("tuned", "bert_base", split), preds("tuned", "finbert_mlm", split),
             preds("fixed", "bert_base", split), preds("fixed", "finbert_mlm", split)],
            B, seed,
        )
        low, high, p_value = _interval(gap_draws)
        tuned, fixed = comparisons[split][GAP_TUNED], comparisons[split][GAP_FIXED]
        gap_change[split] = {
            "definition": "(tuned bert_base - tuned finbert_mlm) - (fixed bert_base - fixed finbert_mlm)",
            "delta_macro_f1": tuned["delta_macro_f1"] - fixed["delta_macro_f1"],
            "per_seed_delta": {
                str(s): tuned["per_seed_delta"][str(s)] - fixed["per_seed_delta"][str(s)] for s in seeds
            },
            "ci_low": low,
            "ci_high": high,
            "p_bootstrap": p_value,
            "interval_contains_zero": _contains_zero(low, high),
        }

    # Holm inside each seed family: the 8 McNemar tests (4 pairs x 2 splits) run with that seed pairing.
    keys = [(split, name) for split in SPLITS for name, _a, _b in COMPARISONS]
    holm = {
        s: holm_bonferroni(
            {f"{split}/{name}": comparisons[split][name]["mcnemar_by_seed"][str(s)]["p_value"] for split, name in keys}
        )
        for s in seeds
    }
    for split, name in keys:
        entry = comparisons[split][name]
        raw = [entry["mcnemar_by_seed"][str(s)]["p_value"] for s in seeds]
        adjusted = {str(s): holm[s][f"{split}/{name}"] for s in seeds}
        entry["p_min"] = float(min(raw))
        entry["p_max"] = float(max(raw))
        entry["p_holm_by_seed"] = adjusted
        entry["n_seeds_significant"] = int(sum(1 for v in adjusted.values() if v < ALPHA))
        entry["n_seeds"] = len(seeds)

    return {
        "protocol": {
            "config": CONFIG,
            "models": list(MODELS),
            "seeds": list(seeds),
            "splits": list(SPLITS),
            "metric": "macro_f1",
            "B": int(B),
            "seed": int(seed),
            "alpha": ALPHA,
            "bootstrap": (
                "seed+sentence two-level bootstrap (src.evaluation.multiseed.seed_sentence_bootstrap): per "
                "resample the sentences and, independently for each side, the seeds are drawn with replacement; "
                "95% percentile interval; two-sided p = 2 * the smaller tail share at or beyond zero. The gap "
                "change has its own bootstrap (stream default_rng([seed, 2])): per resample one sentence draw "
                "shared by the four run sets (tuned BERT, tuned FinBERT, fixed BERT, fixed FinBERT) and an "
                "independent seed draw for each set, because tuned and fixed runs with the same seed number are "
                "different trainings and are not paired."
            ),
            "mcnemar": (
                "exact McNemar per seed pairing (seed s of A against seed s of B); Holm-Bonferroni inside each "
                "seed family over the 8 tests (4 pairs x 2 splits). p_min / p_max are the smallest and largest "
                f"raw p over the {len(seeds)} seeds; n_seeds_significant counts the seeds with Holm-adjusted p < alpha."
            ),
            "difference_sign": "every delta is A - B",
        },
        "recipes": inputs["recipes"],
        "run_dirs": {
            recipe: {model: {str(s): _rel(runs[recipe][model][s]["run_dir"]) for s in seeds} for model in MODELS}
            for recipe in RECIPES
        },
        "environment": _environment(runs, seeds),
        "n": {split: int(len(reference[split]["y"])) for split in SPLITS},
        "max_abs_diff_vs_result_json_macro_f1": float(stored_gap),
        "scores": scores,
        "comparisons": comparisons,
        "gap_change": gap_change,
    }


# --------------------------------------------------------------------------- report


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _side(side: Mapping[str, str]) -> str:
    return f"{side['recipe']} {MODEL_LABELS[side['model']]}"


def _recipe_txt(spec: Mapping[str, Any]) -> str:
    return f"learning rate {format_learning_rate(spec['learning_rate'])}, batch size {spec['batch_size']}"


def _environment_txt(env: Mapping[str, Any], n_runs: int) -> str:
    def describe(entry: Mapping[str, Any]) -> str:
        versions = ", ".join(f"{k} {v}" for k, v in sorted(entry["versions"].items())) or "no versions recorded"
        return f"device `{entry['device']}`; {versions}"

    if env["identical_across_all_runs"]:
        return (
            f"All {n_runs} runs (fixed and tuned) record the same device and library versions in their `result.json`: "
            f"{describe(env['fixed'][0])}. The fixed-vs-tuned comparison is therefore inside one software environment."
        )
    lines = ["The runs do NOT all record the same device and library versions, so the fixed-vs-tuned "
             "comparison mixes environments:"]
    for recipe in RECIPES:
        for entry in env[recipe]:
            lines.append(f"- {recipe}: {describe(entry)}")
    return "\n".join(lines)


def _sentence(split: str, entry: Mapping[str, Any]) -> str:
    a, b = _side(entry["a"]), _side(entry["b"])
    delta = entry["delta_macro_f1"]
    if delta > 0:
        direction = f"{a} scores {abs(delta):.4f} Macro-F1 higher than {b}"
    elif delta < 0:
        direction = f"{a} scores {abs(delta):.4f} Macro-F1 lower than {b}"
    else:
        direction = f"{a} and {b} have the same Macro-F1"
    zero = "contains zero" if entry["interval_contains_zero"] else "does not contain zero"
    return (
        f"On {SPLIT_LABELS[split]}, {direction} (seed means); the 95% interval "
        f"[{_signed(entry['ci_low'])}, {_signed(entry['ci_high'])}] {zero}, and McNemar is significant after Holm "
        f"in {entry['n_seeds_significant']} of {entry['n_seeds']} seed pairings."
    )


def render_report(stats: Mapping[str, Any]) -> str:
    protocol, recipes = stats["protocol"], stats["recipes"]
    seed_ids = protocol["seeds"]
    seeds = [str(s) for s in seed_ids]
    seed_count = "three" if len(seeds) == 3 else str(len(seeds))
    lines = [
        "# Step 4: tuned recipe against the fixed Step 2 recipe",
        "",
        "Written by `scripts/summarize_step4.py` from stored predictions; no model is trained or loaded.",
        "",
        "## Protocol",
        "",
        f"Models `bert_base` (BERT) and `finbert_mlm` (FinBERT) on `{protocol['config']}`, seeds "
        f"{', '.join(seeds)}. Fixed recipe = the Step 2 runs; tuned recipe = the recipe that "
        f"`results/tuning/selection.json` selected on validation Macro-F1 with seed 42, retrained for all {seed_count} seeds.",
        "",
    ]
    added = [str(s) for s in seed_ids if s not in SEEDS]
    if added:
        prespecified = ", ".join(str(s) for s in seed_ids if s in SEEDS) or "none"
        lines.extend([
            f"Pre-specified seeds in this analysis were {prespecified}. Seeds added later were {', '.join(added)}.",
            "",
        ])
    for model in MODELS:
        tuned = recipes["tuned"][model]
        val = "" if tuned.get("val_macro_f1") is None else f" (selection validation Macro-F1 {tuned['val_macro_f1']:.4f})"
        lines.append(
            f"- {MODEL_LABELS[model]}: fixed {_recipe_txt(recipes['fixed'][model])}; "
            f"tuned {_recipe_txt(tuned)}{val}."
        )
    lines += [
        "",
        _environment_txt(stats["environment"], len(seeds) * len(MODELS) * len(RECIPES)),
        "",
        f"Splits: {SPLIT_LABELS['test']} (n = {stats['n']['test']}) and {SPLIT_LABELS['external']} "
        f"(n = {stats['n']['external']}). All runs were checked to share the same sentence ids and labels per split.",
        "",
        f"Intervals: seed+sentence bootstrap (sentences and seeds resampled), B = {protocol['B']}, "
        f"random seed {protocol['seed']}, 95% percentile interval, two-sided p. McNemar: exact test per seed "
        "pairing, Holm-Bonferroni inside each seed family over the 8 tests (4 pairs x 2 splits), "
        f"alpha = {protocol['alpha']}. Every difference is A - B. In every interval, including the gap-change "
        f"interval, the seeds of each run set are resampled independently of the other run sets. With {seed_count} seeds the seed-to-seed spread is "
        "estimated from very few runs, so the intervals are approximate.",
        "",
        "## Scores",
        "",
        "| Split | Model | Recipe | " + " | ".join(f"Macro-F1 seed {s}" for s in seeds)
        + " | Macro-F1 mean | sd | Accuracy mean |",
        "|---|---|---|" + "---:|" * (len(seeds) + 3),
    ]
    for split in SPLITS:
        for model in MODELS:
            for recipe in RECIPES:
                cell = stats["scores"][model][recipe][split]
                per_seed = " | ".join(f"{cell['macro_f1_by_seed'][s]:.4f}" for s in seeds)
                lines.append(
                    f"| {SPLIT_LABELS[split]} | {MODEL_LABELS[model]} | {recipe} | {per_seed} | "
                    f"{cell['macro_f1']['mean']:.4f} | {cell['macro_f1']['sd']:.4f} | {cell['accuracy']['mean']:.4f} |"
                )
    lines += [
        "",
        "## Paired comparisons (Macro-F1, A - B)",
        "",
        "| Split | A | B | Delta | " + " | ".join(f"Delta seed {s}" for s in seeds)
        + " | 95% interval | Bootstrap p | McNemar p min | McNemar p max | Seeds significant (Holm) |",
        "|---|---|---|---:|" + "---:|" * len(seeds) + "---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for name, _a, _b in COMPARISONS:
            entry = stats["comparisons"][split][name]
            per_seed = " | ".join(_signed(entry["per_seed_delta"][s]) for s in seeds)
            lines.append(
                f"| {SPLIT_LABELS[split]} | {_side(entry['a'])} | {_side(entry['b'])} | "
                f"{_signed(entry['delta_macro_f1'])} | {per_seed} | "
                f"[{_signed(entry['ci_low'])}, {_signed(entry['ci_high'])}] | {_p(entry['p_bootstrap'])} | "
                f"{_p(entry['p_min'])} | {_p(entry['p_max'])} | {entry['n_seeds_significant']}/{entry['n_seeds']} |"
            )
    lines += ["", "In plain words:", ""]
    for split in SPLITS:
        for name, _a, _b in COMPARISONS:
            lines.append(f"- {_sentence(split, stats['comparisons'][split][name])}")
    lines += [
        "",
        "## Change of the BERT-FinBERT gap",
        "",
        "Gap change = (tuned BERT - tuned FinBERT) - (fixed BERT - fixed FinBERT). Its interval comes from one "
        "bootstrap in which the sentences are resampled once per draw for all four run sets and the seeds of the "
        "four run sets are resampled independently: tuned and fixed runs with the same seed number are different "
        "trainings, so they are not paired.",
        "",
        "| Split | Gap change | 95% interval | Bootstrap p |",
        "|---|---:|---|---:|",
    ]
    for split in SPLITS:
        gap = stats["gap_change"][split]
        lines.append(
            f"| {SPLIT_LABELS[split]} | {_signed(gap['delta_macro_f1'])} | "
            f"[{_signed(gap['ci_low'])}, {_signed(gap['ci_high'])}] | {_p(gap['p_bootstrap'])} |"
        )
    lines.append("")
    for split in SPLITS:
        gap = stats["gap_change"][split]
        zero = "contains zero" if gap["interval_contains_zero"] else "does not contain zero"
        lines.append(
            f"- On {SPLIT_LABELS[split]}, tuning changes the BERT-minus-FinBERT gap by "
            f"{_signed(gap['delta_macro_f1'])} Macro-F1; the 95% interval "
            f"[{_signed(gap['ci_low'])}, {_signed(gap['ci_high'])}] {zero}."
        )
    lines += ["", "## Run directories", ""]
    for recipe in RECIPES:
        for model in MODELS:
            dirs = ", ".join(f"`{stats['run_dirs'][recipe][model][s]}`" for s in seeds)
            lines.append(f"- {recipe} {MODEL_LABELS[model]}: {dirs}")
    if stats["max_abs_diff_vs_result_json_macro_f1"] > 1e-6:
        lines += [
            "",
            f"Note: Macro-F1 recomputed from the stored predictions differs from a `result.json` value by up to "
            f"{stats['max_abs_diff_vs_result_json_macro_f1']:.2e}.",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli


def _equal(left: Any, right: Any, tolerance: float = CHECK_TOLERANCE) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k], tolerance) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(token) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated seed integers") from exc
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct, nonnegative integers")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tuning-dir", default=DEFAULT_TUNING_DIR)
    parser.add_argument("--step2-runs", default=DEFAULT_STEP2_RUNS)
    parser.add_argument("--fixed-extra-dir", default=DEFAULT_FIXED_EXTRA_DIR,
                        help="fallback directory for fixed-recipe runs missing from --step2-runs")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=_parse_seeds, default=SEEDS,
                        help="comma-separated training seeds (default: 42,100,2026)")
    parser.add_argument("--suffix", default="", help="suffix before the JSON and Markdown extensions")
    parser.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=2026, help="bootstrap random seed")
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare with the existing statistics JSON; write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.B < 1:
        print("refused: --B must be at least 1.", file=sys.stderr)
        return 2
    if os.sep in args.suffix or (os.altsep and os.altsep in args.suffix):
        print("refused: --suffix must not contain path separators.", file=sys.stderr)
        return 2
    try:
        inputs = load_inputs(args.tuning_dir, args.step2_runs, args.seeds, args.fixed_extra_dir)
    except Refuse as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    stats = json.loads(json.dumps(build_statistics(inputs, args.B, args.seed)))
    json_stem, json_ext = os.path.splitext(JSON_NAME)
    json_path = os.path.join(args.out_dir, f"{json_stem}{args.suffix}{json_ext}")

    if args.check:
        if not os.path.isfile(json_path):
            print(f"check failed: {json_path} not found.", file=sys.stderr)
            return 1
        with open(json_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if _equal(stats, existing):
            print(f"check ok: {json_path} matches the recomputed statistics.")
            return 0
        print(f"check failed: {json_path} differs from the recomputed statistics.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report_stem, report_ext = os.path.splitext(REPORT_NAME)
    report_path = os.path.join(args.out_dir, f"{report_stem}{args.suffix}{report_ext}")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(stats))
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    for split in SPLITS:
        for name, _a, _b in COMPARISONS:
            entry = stats["comparisons"][split][name]
            print(
                f"{split:8s} {name:28s} delta={entry['delta_macro_f1']:+.4f} "
                f"ci=[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
                f"mcnemar_sig={entry['n_seeds_significant']}/{entry['n_seeds']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
