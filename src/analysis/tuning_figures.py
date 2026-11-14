"""Figures and a short results note for the tuning study.

Reads the completed tables under a tuning directory. Archived runs are refused.
The original comparison is read from Step 2 ``result.json`` files for the fixed
recipe (learning rate 2e-5, batch size 16, seeds 42, 100, 2026). If those files
do not match, that figure is omitted rather than filled with other numbers.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts import run_step2, run_tuning

MODELS = ("bert_base", "finbert_mlm")
MODEL_LABELS = {"bert_base": "BERT-base", "finbert_mlm": "FinBERT-MLM"}
MODEL_COLORS = {"bert_base": "#0072B2", "finbert_mlm": "#D55E00"}
LEARNING_RATES = (2e-5, 3e-5)
BATCH_SIZES = (16, 32)
SEEDS = (42, 100, 2026)
PHRASEBANK_CONFIG = "sentences_allagree"
EXPECTED_TUNING_ROWS = 8
EXPECTED_FINAL_ROWS = 6
F1_AXIS_TOP = 1.12

TABLE_NAMES = ("tuning_results.csv", "final_results.csv", "selection.json")
FIGURE_STEMS = (
    "hyperparameter_heatmaps",
    "final_model_comparison",
    "per_seed_performance",
)
OPTIONAL_FIGURE_STEM = "original_vs_tuned"
SUMMARY_NAME = "RESULTS_SUMMARY.md"


class TuningResultsError(ValueError):
    """The stored tables are missing, incomplete, or not safe to plot."""


@dataclass(frozen=True)
class TuningRow:
    model_key: str
    learning_rate: float
    batch_size: int
    seed: int
    val_macro_f1: float
    selected: bool
    run_dir: str


@dataclass(frozen=True)
class FinalRow:
    model_key: str
    learning_rate: float
    batch_size: int
    seed: int
    test_macro_f1: float
    external_macro_f1: float
    run_dir: str


@dataclass(frozen=True)
class ScoreSummary:
    mean: float
    sample_sd: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class OriginalScores:
    """Fixed-recipe Step 2 scores, one value per seed, in seed order."""

    test: dict[str, tuple[float, ...]]
    external: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class TuningStudy:
    root: Path
    tuning: tuple[TuningRow, ...]
    final: tuple[FinalRow, ...]
    selected: dict[str, TuningRow]


def mean_and_sample_sd(values: Sequence[float]) -> ScoreSummary:
    """Mean and sample standard deviation (divisor n - 1)."""
    numbers = tuple(float(value) for value in values)
    if len(numbers) < 2:
        raise TuningResultsError("Sample standard deviation needs at least two scores.")
    if not all(math.isfinite(value) for value in numbers):
        raise TuningResultsError("Scores must be finite.")
    return ScoreSummary(
        mean=statistics.fmean(numbers),
        sample_sd=statistics.stdev(numbers),
        values=numbers,
    )


def _assert_live_path(path: Path) -> None:
    if "archive" in path.parts:
        raise TuningResultsError(f"Refusing archived results: {path}")


def _assert_live_run_dir(run_dir: str) -> None:
    parts = run_dir.replace("\\", "/").split("/")
    if "archive" in parts:
        raise TuningResultsError(
            f"Refusing a row that points at an archived run: {run_dir}"
        )


def _matches_rate(value: float, expected: float) -> bool:
    return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-12)


def _require_columns(row: Mapping[str, str], columns: Sequence[str], path: Path) -> None:
    missing = [name for name in columns if name not in row or row[name] == ""]
    if missing:
        raise TuningResultsError(f"{path} is missing {', '.join(missing)}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _assert_live_path(path)
    if not path.is_file():
        raise TuningResultsError(f"Missing results table: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_tuning_row(raw: Mapping[str, str], path: Path) -> TuningRow:
    _require_columns(
        raw,
        ("model_key", "learning_rate", "batch_size", "seed", "val_macro_f1", "selected", "run_dir"),
        path,
    )
    try:
        row = TuningRow(
            model_key=raw["model_key"],
            learning_rate=float(raw["learning_rate"]),
            batch_size=int(raw["batch_size"]),
            seed=int(raw["seed"]),
            val_macro_f1=float(raw["val_macro_f1"]),
            selected=raw["selected"] == "1",
            run_dir=raw["run_dir"],
        )
    except (TypeError, ValueError) as exc:
        raise TuningResultsError(f"{path} has a row that cannot be parsed.") from exc
    if not math.isfinite(row.val_macro_f1):
        raise TuningResultsError(f"{path} has a non-finite validation Macro-F1.")
    _assert_live_run_dir(row.run_dir)
    return row


def _parse_final_row(raw: Mapping[str, str], path: Path) -> FinalRow:
    _require_columns(
        raw,
        ("model_key", "learning_rate", "batch_size", "seed", "test_macro_f1", "external_macro_f1", "run_dir"),
        path,
    )
    try:
        row = FinalRow(
            model_key=raw["model_key"],
            learning_rate=float(raw["learning_rate"]),
            batch_size=int(raw["batch_size"]),
            seed=int(raw["seed"]),
            test_macro_f1=float(raw["test_macro_f1"]),
            external_macro_f1=float(raw["external_macro_f1"]),
            run_dir=raw["run_dir"],
        )
    except (TypeError, ValueError) as exc:
        raise TuningResultsError(f"{path} has a row that cannot be parsed.") from exc
    if not math.isfinite(row.test_macro_f1) or not math.isfinite(row.external_macro_f1):
        raise TuningResultsError(f"{path} has a non-finite Macro-F1.")
    _assert_live_run_dir(row.run_dir)
    return row


def _recipe(model: str, learning_rate: float, batch_size: int) -> tuple[str, str, int]:
    return (model, run_step2.format_learning_rate(learning_rate), int(batch_size))


def select_recorded_winners(rows: Sequence[TuningRow]) -> dict[str, TuningRow]:
    """Apply the study rule: maximum validation Macro-F1, then lower learning rate, then smaller batch."""
    chosen = run_tuning.select_best([
        {
            "model_key": row.model_key,
            "learning_rate": row.learning_rate,
            "batch_size": row.batch_size,
            "val_macro_f1": row.val_macro_f1,
        }
        for row in rows
    ])
    winners = {}
    for model, spec in chosen.items():
        matches = [
            row for row in rows
            if _recipe(row.model_key, row.learning_rate, row.batch_size)
            == _recipe(model, spec["learning_rate"], spec["batch_size"])
        ]
        if len(matches) != 1:
            raise TuningResultsError(f"Expected one tuning row for the selected {model} configuration.")
        winners[model] = matches[0]
    return winners


def load_study(root: Path) -> TuningStudy:
    """Load and validate the three live result files. Archived paths are rejected."""
    root = Path(root)
    _assert_live_path(root)
    tuning_path = root / "tuning_results.csv"
    final_path = root / "final_results.csv"
    selection_path = root / "selection.json"
    tuning_rows = tuple(_parse_tuning_row(row, tuning_path) for row in _read_csv(tuning_path))
    final_rows = tuple(_parse_final_row(row, final_path) for row in _read_csv(final_path))
    if len(tuning_rows) != EXPECTED_TUNING_ROWS:
        raise TuningResultsError(
            f"Expected {EXPECTED_TUNING_ROWS} tuning configurations in {tuning_path}, found {len(tuning_rows)}."
        )
    if len(final_rows) != EXPECTED_FINAL_ROWS:
        raise TuningResultsError(
            f"Expected {EXPECTED_FINAL_ROWS} final evaluation runs in {final_path}, found {len(final_rows)}."
        )
    if not selection_path.is_file():
        raise TuningResultsError(f"Missing selection record: {selection_path}")
    _assert_live_path(selection_path)
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TuningResultsError(f"{selection_path} is not valid JSON.") from exc
    winners = _validate_grid(tuning_rows, final_rows, selection, selection_path)
    return TuningStudy(root=root, tuning=tuning_rows, final=final_rows, selected=winners)


def _validate_grid(tuning_rows: Sequence[TuningRow], final_rows: Sequence[FinalRow],
                   selection: Mapping[str, object], selection_path: Path) -> dict[str, TuningRow]:
    seen = {}
    for row in tuning_rows:
        if row.model_key not in MODELS:
            raise TuningResultsError(f"Unexpected model in the tuning table: {row.model_key}")
        if row.seed != run_tuning.SELECTION_SEED:
            raise TuningResultsError(
                f"Tuning rows must use seed {run_tuning.SELECTION_SEED}; found seed {row.seed}."
            )
        key = _recipe(row.model_key, row.learning_rate, row.batch_size)
        if key in seen:
            raise TuningResultsError(f"Duplicate tuning configuration: {key}.")
        if not any(_matches_rate(row.learning_rate, rate) for rate in LEARNING_RATES):
            raise TuningResultsError(f"Learning rate {row.learning_rate} is outside the study grid.")
        if row.batch_size not in BATCH_SIZES:
            raise TuningResultsError(f"Batch size {row.batch_size} is outside the study grid.")
        seen[key] = row
    expected = {
        _recipe(model, rate, batch)
        for model in MODELS
        for rate in LEARNING_RATES
        for batch in BATCH_SIZES
    }
    if set(seen) != expected:
        raise TuningResultsError("The tuning table is not the full 2 x 2 grid for both models.")
    winners = select_recorded_winners(tuning_rows)
    recorded = selection.get("selected") if isinstance(selection, Mapping) else None
    if not isinstance(recorded, Mapping):
        raise TuningResultsError(f"{selection_path} has no selected object.")
    for model in MODELS:
        spec = recorded.get(model)
        winner = winners[model]
        if not isinstance(spec, Mapping):
            raise TuningResultsError(f"{selection_path} has no selected configuration for {model}.")
        try:
            recorded_rate = float(spec["learning_rate"])
            recorded_batch = int(spec["batch_size"])
            recorded_score = float(spec["val_macro_f1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningResultsError(f"{selection_path} selected.{model} is incomplete.") from exc
        if _recipe(model, recorded_rate, recorded_batch) != _recipe(
            winner.model_key, winner.learning_rate, winner.batch_size
        ):
            raise TuningResultsError(
                f"{selection_path} disagrees with validation Macro-F1 selection for {model}."
            )
        if not math.isclose(recorded_score, winner.val_macro_f1, rel_tol=0.0, abs_tol=1e-6):
            raise TuningResultsError(
                f"{selection_path} validation Macro-F1 for {model} does not match the tuning table."
            )
        marked = [row for row in tuning_rows if row.model_key == model and row.selected]
        if len(marked) != 1 or marked[0] != winner:
            raise TuningResultsError(f"The selected flag in the tuning table does not match {model}.")
    for model in MODELS:
        rows = [row for row in final_rows if row.model_key == model]
        seeds = tuple(sorted(row.seed for row in rows))
        if seeds != tuple(sorted(SEEDS)):
            raise TuningResultsError(
                f"Expected final seeds {list(SEEDS)} for {model}, found {list(seeds)}."
            )
        winner = winners[model]
        for row in rows:
            if _recipe(row.model_key, row.learning_rate, row.batch_size) != _recipe(
                winner.model_key, winner.learning_rate, winner.batch_size
            ):
                raise TuningResultsError(
                    f"Final run {row.run_dir} does not use the selected {model} configuration."
                )
    return winners


def tables_present(root: Path) -> bool:
    root = Path(root)
    return all((root / name).is_file() for name in TABLE_NAMES)


def resolve_results_root(local_root: Path, drive_root: Path | None = None) -> tuple[Path, str | None]:
    """Use the local tables when they validate. Otherwise use the Drive backup."""
    local_root = Path(local_root)
    note = None
    local_error: TuningResultsError | None = None
    if tables_present(local_root):
        try:
            load_study(local_root)
            return local_root, None
        except TuningResultsError as exc:
            local_error = exc
    if drive_root is not None and tables_present(drive_root):
        load_study(drive_root)
        if local_error is not None:
            note = (
                f"Local tables in {local_root} were not used ({local_error}). "
                f"Figures use {drive_root}."
            )
        return Path(drive_root), note
    if local_error is not None:
        raise TuningResultsError(
            f"Local results at {local_root} cannot be plotted ({local_error}). "
            "The Drive backup does not contain a complete replacement."
        )
    raise TuningResultsError(
        "Required results were not found. Expected tuning_results.csv (8 configurations), "
        "final_results.csv (6 runs), and selection.json in results/tuning or "
        "/content/drive/MyDrive/essential_text_tuning. Archived runs are not used."
    )


def load_original_fixed_recipe(step2_runs: Path) -> tuple[OriginalScores | None, str | None]:
    """Read the fixed-recipe Step 2 runs. Return a reason when they cannot be matched."""
    step2_runs = Path(step2_runs)
    test: dict[str, tuple[float, ...]] = {}
    external: dict[str, tuple[float, ...]] = {}
    problems = []
    for model in MODELS:
        test_scores = []
        external_scores = []
        for seed in SEEDS:
            path = step2_runs / PHRASEBANK_CONFIG / model / f"seed{seed}" / "result.json"
            score_pair = _fixed_recipe_scores(path, model, seed)
            if isinstance(score_pair, str):
                problems.append(score_pair)
                continue
            test_scores.append(score_pair[0])
            external_scores.append(score_pair[1])
        if len(test_scores) == len(SEEDS):
            test[model] = tuple(test_scores)
            external[model] = tuple(external_scores)
    if problems:
        reason = (
            "Figure 3 was not drawn. The original fixed-recipe files "
            f"(learning rate {run_step2.format_learning_rate(run_step2.DEFAULT_LEARNING_RATE)}, "
            f"batch size {run_step2.DEFAULT_BATCH_SIZE}, seeds {list(SEEDS)}, "
            f"{PHRASEBANK_CONFIG}) could not all be matched: " + " ".join(problems)
        )
        return None, reason
    return OriginalScores(test=test, external=external), None


def _fixed_recipe_scores(path: Path, model: str, seed: int) -> tuple[float, float] | str:
    if not path.is_file():
        return f"{path} is missing."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"{path} is not valid JSON."
    if payload.get("status") != "completed":
        return f"{path} is not a completed run."
    if payload.get("config") != PHRASEBANK_CONFIG or payload.get("model_key") != model:
        return f"{path} is not {PHRASEBANK_CONFIG}/{model}."
    if int(payload.get("seed", -1)) != int(seed):
        return f"{path} does not record seed {seed}."
    hyper = payload.get("hyperparams") or {}
    try:
        learning_rate = float(hyper["learning_rate"])
        batch_size = int(hyper["batch_size"])
    except (KeyError, TypeError, ValueError):
        return f"{path} has no fixed-recipe hyperparameters."
    if not run_step2.is_default_recipe(learning_rate, batch_size):
        return (
            f"{path} records learning rate {learning_rate} and batch size {batch_size}, "
            "not the original fixed recipe."
        )
    metrics = payload.get("metrics") or {}
    try:
        test_score = float((metrics.get("test") or {})["macro_f1"])
        external_score = float((metrics.get("external") or {})["macro_f1"])
    except (KeyError, TypeError, ValueError):
        return f"{path} has no test and external Macro-F1."
    if not math.isfinite(test_score) or not math.isfinite(external_score):
        return f"{path} has a non-finite Macro-F1."
    return test_score, external_score


def _scores_for(rows: Sequence[FinalRow], model: str, split: str) -> ScoreSummary:
    ordered = []
    for seed in SEEDS:
        match = [row for row in rows if row.model_key == model and row.seed == seed]
        if len(match) != 1:
            raise TuningResultsError(f"Expected one final row for {model} seed {seed}.")
        ordered.append(getattr(match[0], f"{split}_macro_f1"))
    return mean_and_sample_sd(ordered)


def _format_pm(summary: ScoreSummary) -> str:
    return f"{summary.mean:.3f} ± {summary.sample_sd:.3f}"


def _format_rate(learning_rate: float) -> str:
    return run_step2.format_learning_rate(learning_rate)


def _higher_model(left_model: str, left: float, right_model: str, right: float) -> str:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=5e-4):
        return (
            f"{MODEL_LABELS[left_model]} and {MODEL_LABELS[right_model]} "
            f"have the same three-seed mean at the reported precision ({left:.3f})."
        )
    winner, score, other, other_score = (
        (left_model, left, right_model, right) if left > right
        else (right_model, right, left_model, left)
    )
    return (
        f"{MODEL_LABELS[winner]} has the higher three-seed mean "
        f"({score:.3f} versus {other_score:.3f} for {MODEL_LABELS[other]})."
    )


def build_summary(study: TuningStudy, original: OriginalScores | None, skip_reason: str | None) -> str:
    """Markdown note. Every reported score is taken from ``study`` or ``original``."""
    final_test = {model: _scores_for(study.final, model, "test") for model in MODELS}
    final_external = {model: _scores_for(study.final, model, "external") for model in MODELS}
    lines = [
        "# Tuning results",
        "",
        "Selection used seed 42 and validation Macro-F1 only. "
        "The final rows are the locked Financial PhraseBank test split and the external SEntFiN holdout. "
        "Means use seeds 42, 100, and 2026. The ± term is the sample standard deviation.",
        "",
        "## Selected configurations",
        "",
        "| Model | Learning rate | Batch size | Validation Macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        winner = study.selected[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {_format_rate(winner.learning_rate)} | "
            f"{winner.batch_size} | {winner.val_macro_f1:.3f} |"
        )
    lines.extend([
        "",
        "## Final evaluation",
        "",
        "| Model | PhraseBank test Macro-F1 | SEntFiN Macro-F1 |",
        "| --- | ---: | ---: |",
    ])
    for model in MODELS:
        lines.append(
            f"| {MODEL_LABELS[model]} | {_format_pm(final_test[model])} | {_format_pm(final_external[model])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "On the PhraseBank test set, "
        + _higher_model("bert_base", final_test["bert_base"].mean, "finbert_mlm", final_test["finbert_mlm"].mean)
    )
    lines.append(
        "On SEntFiN, "
        + _higher_model(
            "bert_base", final_external["bert_base"].mean, "finbert_mlm", final_external["finbert_mlm"].mean
        )
    )
    lines.append(
        "These comparisons describe the stored runs. "
        "They are not a significance test, and the seed-level differences in Figure 4 "
        "should not be read as statistically significant."
    )
    if original is not None:
        original_test = {model: mean_and_sample_sd(original.test[model]) for model in MODELS}
        original_external = {model: mean_and_sample_sd(original.external[model]) for model in MODELS}
        phrasebank_gap_original = original_test["bert_base"].mean - original_test["finbert_mlm"].mean
        phrasebank_gap_tuned = final_test["bert_base"].mean - final_test["finbert_mlm"].mean
        sentfin_gap_original = original_external["bert_base"].mean - original_external["finbert_mlm"].mean
        sentfin_gap_tuned = final_external["bert_base"].mean - final_external["finbert_mlm"].mean
        lines.append(
            "The original bars use the fixed recipe stored in the Step 2 result files "
            f"(learning rate {_format_rate(run_step2.DEFAULT_LEARNING_RATE)}, "
            f"batch size {run_step2.DEFAULT_BATCH_SIZE}). "
            "The tuned bars use the configuration selected independently for each model."
        )
        lines.append(
            f"On the PhraseBank test set, the BERT-base minus FinBERT-MLM gap is "
            f"{phrasebank_gap_original:.3f} under the fixed recipe and {phrasebank_gap_tuned:.3f} after independent tuning. "
            f"On SEntFiN, the same gap is {sentfin_gap_original:.3f} under the fixed recipe and "
            f"{sentfin_gap_tuned:.3f} after independent tuning."
        )
    elif skip_reason:
        lines.extend(["", skip_reason])
    lines.extend(["", "## README images", ""])
    figures = list(FIGURE_STEMS)
    if original is not None:
        figures.insert(2, OPTIONAL_FIGURE_STEM)
    captions = {
        "hyperparameter_heatmaps": "Hyperparameter tuning",
        "final_model_comparison": "Final model comparison",
        "original_vs_tuned": "Original fixed recipe versus independently tuned models",
        "per_seed_performance": "Per-seed Macro-F1",
    }
    for stem in figures:
        lines.append(f"![{captions[stem]}](results/tuning/figures/{stem}.png)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _style():
    import matplotlib.pyplot as plt

    return plt.rc_context({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _rate_index(learning_rate: float) -> int:
    for index, rate in enumerate(LEARNING_RATES):
        if _matches_rate(learning_rate, rate):
            return index
    raise TuningResultsError(f"Learning rate {learning_rate} is outside the study grid.")


def plot_heatmaps(study: TuningStudy, stem: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    matrices = []
    for model in MODELS:
        grid = np.full((len(LEARNING_RATES), len(BATCH_SIZES)), np.nan)
        for row in study.tuning:
            if row.model_key != model:
                continue
            grid[_rate_index(row.learning_rate), BATCH_SIZES.index(row.batch_size)] = row.val_macro_f1
        matrices.append(grid)
    vmin = float(min(np.nanmin(grid) for grid in matrices))
    vmax = float(max(np.nanmax(grid) for grid in matrices))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-9):
        vmin -= 0.01
        vmax += 0.01
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)
        image = None
        for ax, model, grid in zip(axes, MODELS, matrices):
            image = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="Blues", origin="upper", aspect="equal")
            ax.set_title(MODEL_LABELS[model])
            ax.set_xlabel("Batch size")
            ax.set_xticks(range(len(BATCH_SIZES)), [str(batch) for batch in BATCH_SIZES])
            ax.set_ylabel("Learning rate")
            ax.set_yticks(
                range(len(LEARNING_RATES)),
                [f"{rate:.0e}" for rate in LEARNING_RATES],
            )
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#d0d0d0")
            winner = study.selected[model]
            for row_index, rate in enumerate(LEARNING_RATES):
                for col_index, batch in enumerate(BATCH_SIZES):
                    value = float(grid[row_index, col_index])
                    shade = (value - vmin) / (vmax - vmin)
                    ax.text(
                        col_index, row_index, f"{value:.3f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if shade > 0.62 else "#1a1a1a",
                    )
                    if _matches_rate(rate, winner.learning_rate) and batch == winner.batch_size:
                        ax.add_patch(Rectangle(
                            (col_index - 0.5, row_index - 0.5), 1, 1,
                            fill=False, edgecolor="black", linewidth=2.0,
                        ))
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="Validation Macro-F1")
        fig.suptitle("Validation Macro-F1 by learning rate and batch size", fontsize=13)
        fig.text(
            0.5, -0.04,
            "Both panels share one color scale. The outlined cell is the selected configuration.",
            ha="center", va="top", fontsize=9, color="#333333",
        )
        _save(fig, stem)


def _grouped_bars(ax, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
                  *, show_error: bool = True) -> None:
    import numpy as np

    positions = np.arange(len(labels))
    width = 0.36
    offsets = (-width / 2, width / 2)
    highest = 0.0
    for offset, (model, means, sds) in zip(offsets, series):
        bars = ax.bar(
            positions + offset, means, width=width,
            yerr=sds if show_error else None,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#222222"},
            zorder=2,
        )
        for bar, mean, sd in zip(bars, means, sds):
            top = mean + (sd if show_error else 0.0)
            highest = max(highest, top)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + 0.02,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=8, color="#222222",
            )
    ax.set_xticks(positions, list(labels))
    ax.set_ylim(0, max(F1_AXIS_TOP, highest + 0.08))
    ax.set_ylabel("Macro-F1")
    ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)


def plot_final_comparison(study: TuningStudy, stem: Path) -> None:
    import matplotlib.pyplot as plt

    series = []
    for model in MODELS:
        test = _scores_for(study.final, model, "test")
        external = _scores_for(study.final, model, "external")
        series.append((model, (test.mean, external.mean), (test.sample_sd, external.sample_sd)))
    with _style():
        fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
        _grouped_bars(ax, ["Financial PhraseBank test", "External SEntFiN"], series)
        ax.legend(loc="upper right")
        ax.set_title("Final Macro-F1 after independent hyperparameter selection")
        ax.set_xlabel("Evaluation set")
        fig.text(
            0.5, -0.06,
            "Bars are means over seeds 42, 100, and 2026. Error bars are the sample standard deviation.",
            ha="center", va="top", fontsize=9, color="#333333",
        )
        _save(fig, stem)


def plot_per_seed(study: TuningStudy, stem: Path) -> None:
    import matplotlib.pyplot as plt

    panels = (
        ("test_macro_f1", "Financial PhraseBank test"),
        ("external_macro_f1", "External SEntFiN"),
    )
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharey=True)
        for ax, (attribute, title) in zip(axes, panels):
            series = []
            for model in MODELS:
                values = []
                for seed in SEEDS:
                    row = next(item for item in study.final if item.model_key == model and item.seed == seed)
                    values.append(getattr(row, attribute))
                series.append((model, tuple(values), (0.0, 0.0, 0.0)))
            _grouped_bars(ax, [str(seed) for seed in SEEDS], series, show_error=False)
            ax.set_title(title)
            ax.set_xlabel("Seed")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
        fig.suptitle("Macro-F1 for each final-evaluation seed", fontsize=13, y=1.06)
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
        fig.text(
            0.5, -0.05,
            "Each bar is one stored seed. Differences across seeds are descriptive, not a significance test.",
            ha="center", va="top", fontsize=9, color="#333333",
        )
        _save(fig, stem)


def plot_original_vs_tuned(study: TuningStudy, original: OriginalScores, stem: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        "PhraseBank\nfixed recipe",
        "PhraseBank\ntuned",
        "SEntFiN\nfixed recipe",
        "SEntFiN\ntuned",
    ]
    series = []
    for model in MODELS:
        tuned_test = _scores_for(study.final, model, "test")
        tuned_external = _scores_for(study.final, model, "external")
        original_test = mean_and_sample_sd(original.test[model])
        original_external = mean_and_sample_sd(original.external[model])
        means = (
            original_test.mean, tuned_test.mean,
            original_external.mean, tuned_external.mean,
        )
        sds = (
            original_test.sample_sd, tuned_test.sample_sd,
            original_external.sample_sd, tuned_external.sample_sd,
        )
        series.append((model, means, sds))
    with _style():
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        _grouped_bars(ax, labels, series)
        ax.legend(loc="upper right")
        ax.set_title("Fixed recipe and independently tuned models")
        ax.set_xlabel("Dataset and hyperparameter source")
        fig.text(
            0.5, -0.08,
            "Fixed-recipe scores are read from the Step 2 result files. "
            "Tuned scores use the selected learning rate and batch size. "
            "Error bars are the sample standard deviation.",
            ha="center", va="top", fontsize=9, color="#333333",
        )
        _save(fig, stem)


def generate_report(results_root: Path, figures_dir: Path, step2_runs: Path) -> dict[str, object]:
    """Write the figures and RESULTS_SUMMARY.md for one validated results directory."""
    study = load_study(results_root)
    figures_dir = Path(figures_dir)
    _assert_live_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    original, skip_reason = load_original_fixed_recipe(step2_runs)
    plot_heatmaps(study, figures_dir / "hyperparameter_heatmaps")
    plot_final_comparison(study, figures_dir / "final_model_comparison")
    plot_per_seed(study, figures_dir / "per_seed_performance")
    written = [
        figures_dir / f"{stem}.png" for stem in FIGURE_STEMS
    ] + [
        figures_dir / f"{stem}.pdf" for stem in FIGURE_STEMS
    ]
    if original is not None:
        plot_original_vs_tuned(study, original, figures_dir / OPTIONAL_FIGURE_STEM)
        written.extend([
            figures_dir / f"{OPTIONAL_FIGURE_STEM}.png",
            figures_dir / f"{OPTIONAL_FIGURE_STEM}.pdf",
        ])
    summary = build_summary(study, original, skip_reason)
    summary_path = figures_dir / SUMMARY_NAME
    summary_path.write_text(summary, encoding="utf-8")
    return {
        "results_root": str(study.root),
        "figures_dir": str(figures_dir),
        "summary_path": str(summary_path),
        "summary_markdown": summary,
        "figures": [str(path) for path in written],
        "original_comparison": original is not None,
        "original_skip_reason": skip_reason,
    }


def generate_from_available_results(local_root: Path, figures_dir: Path, step2_runs: Path,
                                    drive_root: Path | None = None) -> dict[str, object]:
    """Plot local results, or the Drive backup when the local tables are unavailable."""
    root, note = resolve_results_root(local_root, drive_root)
    report = generate_report(root, figures_dir, step2_runs)
    report["source_note"] = note
    return report
