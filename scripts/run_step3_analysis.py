"""Step 3: annotator agreement, per-phenomenon error rates and architecture/domain attribution.

Reads the diagnostic directory (key, two annotation files, rule tags) and the Step 2 run layout,
writes `annotation_agreement.json`, `phenomenon_analysis.json`, `STEP3_REPORT.md` and three figures.
Bootstraps resample sentences and seeds; error rates are reported raw and label-balanced; the
attribution p-values are Holm-adjusted within each table. Exit code 2 when an input file is
missing or malformed. Nothing is downloaded or trained here.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Any, Callable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attribution import (  # noqa: E402
    PHENOMENA,
    agreement_table,
    consensus_tags,
    loss_only_ids,
    phenomenon_attribution,
    phenomenon_error_rates,
    read_annotations,
    rq1_cue_breakdown,
    rq1_terminology_comparison,
)

CONFIGS = ("sentences_allagree", "sentences_75agree", "sentences_50agree")
BASELINES = ("vader_official", "lm_official", "vader", "loughran_mcdonald", "tfidf_logreg",
             "tfidf_svm")
FINETUNED = ("bert_base", "finbert_mlm")
MODELS = BASELINES + FINETUNED
ZERO_SHOT = ("external_only", "prosus_finbert", "zeroshot")
_FALLBACK_NAMES = {"prosus_finbert": "ProsusAI/finbert (zero-shot)"}

# fallback cue matcher, used only when phenomena.financial_terms is not available
_FALLBACK_TERMS = re.compile(
    r"\b(?:costs?|expenses?|liabilit(?:y|ies)|debts?|depreciation|amorti[sz]ation|impairments?|"
    r"write[- ]?downs?|write[- ]?offs?|provisions?|charges?|tax(?:es)?|loss(?:es)?|deficits?|"
    r"leverage[d]?|exposures?|risks?|restructuring|bearish|bears?|volatility|claims?)\b",
    re.IGNORECASE,
)


def _display_names() -> dict[str, str]:
    try:  # lazy: the registry is owned by another module
        from src.baselines.registry import DISPLAY_NAMES
        return dict(DISPLAY_NAMES)
    except ImportError:
        return {}


def _name(model: str) -> str:
    return _display_names().get(model) or _FALLBACK_NAMES.get(model, model)


def _cue_function() -> Callable[[str], list[str]]:
    try:
        from src.analysis.phenomena import financial_terms
        return financial_terms
    except ImportError:
        return lambda text: [m.group(0).lower() for m in _FALLBACK_TERMS.finditer(text)]


class MissingInput(Exception):
    """A required input file is absent or unreadable."""


def _require(path: str) -> str:
    if not os.path.isfile(path):
        raise MissingInput(f"Missing input file: {path}")
    return path


def _load_json(path: str) -> Any:
    _require(path)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as exc:
        raise MissingInput(f"Unreadable JSON {path}: {exc}") from exc


def _read_ann(path: str) -> dict[str, dict[str, int]]:
    _require(path)
    try:
        return read_annotations(path)
    except ValueError as exc:
        raise MissingInput(str(exc)) from exc


def _run_preds(run_dir: str, split: str) -> dict[str, int]:
    """Predicted label per id for one run directory and split."""
    npz_path = _require(os.path.join(run_dir, "predictions.npz"))
    with np.load(npz_path, allow_pickle=False) as data:
        if f"{split}_ids" not in data.files:
            raise MissingInput(f"Missing split '{split}' in {npz_path}")
        ids = [str(i) for i in data[f"{split}_ids"]]
        pred = np.argmax(data[f"{split}_prob"], axis=1)
    return {i: int(p) for i, p in zip(ids, pred)}


def _model_run_dirs(runs_dir: str, config: str, model: str) -> list[str]:
    base = os.path.join(runs_dir, config, model)
    if model in BASELINES:
        dirs = [os.path.join(base, "seed42")]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(base, "seed*"))
                      if os.path.isdir(d) and not d.endswith(".tmp"))
    if not dirs or not all(os.path.isfile(os.path.join(d, "result.json")) for d in dirs):
        raise MissingInput(f"Missing run(s): {os.path.join(base, 'seed*', 'result.json')}")
    for d in dirs:
        if _load_json(os.path.join(d, "result.json")).get("status") != "completed":
            raise MissingInput(f"Run not completed: {d}")
    return dirs


def _config_preds(runs_dir: str, config: str, split: str) -> dict[str, list[dict[str, int]]]:
    return {m: [_run_preds(d, split) for d in _model_run_dirs(runs_dir, config, m)]
            for m in MODELS}


def _pick_tfidf_best(runs_dir: str, config: str) -> tuple[str, dict[str, float]]:
    scores = {}
    for model in ("tfidf_logreg", "tfidf_svm"):
        result = _load_json(os.path.join(runs_dir, config, model, "seed42", "result.json"))
        scores[model] = float(((result.get("metrics") or {}).get("val") or {})["macro_f1"])
    best = "tfidf_svm" if scores["tfidf_svm"] > scores["tfidf_logreg"] else "tfidf_logreg"
    return best, scores


def _clean(obj: Any) -> Any:
    """JSON-safe copy: nan/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_ci(entry: dict, key: str = "error_rate") -> str:
    if entry.get(key) is None:
        return "n/a"
    prefix = "balanced_" if key == "balanced_error" else ""
    return f"{entry[key]:.3f} [{entry[prefix + 'ci_low']:.3f}, {entry[prefix + 'ci_high']:.3f}]"


def _split_analysis(tags: dict, y_true: dict, preds: dict, tfidf_best: str, B: int,
                    seed: int, texts: dict | None = None) -> dict[str, Any]:
    out = {
        "n": len(tags),
        "error_rates": phenomenon_error_rates(tags, y_true, preds, B=B, seed=seed),
        "attribution": phenomenon_attribution(tags, y_true, preds, tfidf_best, B=B, seed=seed),
        "rq1_financial_terminology": rq1_terminology_comparison(tags, y_true, preds, B=B,
                                                                 seed=seed),
    }
    if texts is not None:
        cue_fn = _cue_function()
        cues = {i: cue_fn(texts[i]) for i in tags
                if i in texts and i in y_true and int(tags[i]["financial_terminology"]) == 1}
        drop = loss_only_ids(cues)
        out["rq1_cue_breakdown"] = rq1_cue_breakdown(cues, y_true, preds, B=B, seed=seed)
        out["rq1_excluding_loss"] = rq1_terminology_comparison(tags, y_true, preds, B=B,
                                                               seed=seed, drop_ids=drop)
        out["n_loss_only_neutral_dropped"] = sum(1 for i in drop if int(y_true[i]) == 1)
    return out


def _full_split(runs_dir: str, splits_dir: str, tagger: Callable, B: int, seed: int,
                warnings: list[str]) -> dict[str, Any]:
    """Rule-tag analysis on every available config's test split and on SEntFiN."""
    available = [c for c in CONFIGS if os.path.isdir(os.path.join(runs_dir, c))]
    out: dict[str, Any] = {}
    if not available:
        return out
    payload = _load_json(os.path.join(splits_dir, "sentfin_dedup.json"))
    sentfin_rows = [r for r in payload["records"] if "text" in r and "label" in r]
    sentfin_texts = {str(r["id"]): r["text"] for r in sentfin_rows}
    sentfin_labels = {str(r["id"]): int(r["label"]) for r in sentfin_rows}
    sids = sorted(sentfin_texts)
    sentfin_tags = dict(zip(sids, tagger([sentfin_texts[i] for i in sids])))
    for config in available:
        manifest = _load_json(os.path.join(splits_dir, f"{config}_manifest.json"))
        best, _ = _pick_tfidf_best(runs_dir, config)
        entry: dict[str, Any] = {"tfidf_best": best}
        test_preds = _config_preds(runs_dir, config, "test")
        ids = sorted(next(iter(test_preds.values()))[0])
        records = manifest["records_by_id"]
        tag_list = tagger([records[i]["text"] for i in ids])
        tags = dict(zip(ids, tag_list))
        y_true = {i: int(records[i]["label"]) for i in ids}
        entry["test"] = _split_analysis(tags, y_true, test_preds, best, B, seed,
                                        texts={i: records[i]["text"] for i in ids})
        # SEntFiN, models trained on this config
        ext_preds = _config_preds(runs_dir, config, "external")
        zs_dir = os.path.join(runs_dir, *ZERO_SHOT)
        if os.path.isfile(os.path.join(zs_dir, "predictions.npz")):
            ext_preds["prosus_finbert"] = [_run_preds(zs_dir, "external")]
        else:
            warnings.append("prosus_finbert zero-shot run not found; omitted from SEntFiN tables.")
        entry["external"] = _split_analysis(sentfin_tags, sentfin_labels, ext_preds, best,
                                            B, seed, texts=sentfin_texts)
        out[config] = entry
    return out


def _heatmap(path: str, rates: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((len(MODELS), len(PHENOMENA)), np.nan)
    for j, name in enumerate(PHENOMENA):
        for i, model in enumerate(MODELS):
            value = rates[name]["models"][model]["tag1"]["balanced_error"]
            if value is not None:
                grid[i, j] = value
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(grid, cmap="viridis_r", vmin=0, vmax=max(0.05, np.nanmax(grid)
                                                             if np.isfinite(grid).any() else 1))
    ax.set_xticks(range(len(PHENOMENA)))
    ax.set_xticklabels(
        [f"{p.replace('_', ' ')}\n(n={rates[p]['n_tag1']}{'*' if rates[p]['small_n'] else ''})"
         for p in PHENOMENA], fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([_name(m) for m in MODELS], fontsize=8)
    for i in range(len(MODELS)):
        for j in range(len(PHENOMENA)):
            if np.isfinite(grid[i, j]):
                vmax = im.get_clim()[1]
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] > 0.45 * vmax else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="label-balanced error on tagged sentences")
    ax.set_title(title + "  (* = fewer than 20 tagged sentences)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _attribution_plot(path: str, attribution: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(PHENOMENA))
    width = 0.38
    for k, (key, label) in enumerate((("delta_arch", "architecture (BERT - TF-IDF)"),
                                      ("delta_domain", "domain pretraining (FinBERT-MLM - BERT)"))):
        vals = [attribution[p][key]["delta"] for p in PHENOMENA]
        low = [attribution[p][key]["ci_low"] for p in PHENOMENA]
        high = [attribution[p][key]["ci_high"] for p in PHENOMENA]
        v = np.array([np.nan if a is None else a for a in vals], dtype=float)
        lo = np.array([np.nan if a is None else a for a in low], dtype=float)
        hi = np.array([np.nan if a is None else a for a in high], dtype=float)
        ax.bar(x + (k - 0.5) * width, np.nan_to_num(v), width, label=label,
               yerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)], capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={attribution[p]['n']}{'*' if attribution[p]['small_n'] else ''})"
                        for p in PHENOMENA], fontsize=8)
    ax.set_ylabel("accuracy difference (95% CI)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rq1_panel(ax: Any, rq1: dict, title: str) -> None:
    models = list(rq1)
    x = np.arange(len(models))
    width = 0.38
    ax.set_facecolor("white")
    for k, (key, label) in enumerate((("term1", "financial terminology = 1"),
                                      ("term0", "financial terminology = 0"))):
        def col(field: str) -> np.ndarray:
            return np.array([np.nan if rq1[m][key][field] is None else rq1[m][key][field]
                             for m in models], dtype=float)
        rate, lo, hi = col("rate"), col("ci_low"), col("ci_high")
        n = next(iter(rq1.values()))[key]["n"]
        ax.bar(x + (k - 0.5) * width, np.nan_to_num(rate), width, label=f"{label} (n = {n})",
               yerr=[np.nan_to_num(rate - lo), np.nan_to_num(hi - rate)], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_name(m) for m in models], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("neutral-to-negative rate (gold neutral)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def _rq1_plot(path: str, panels: list[tuple[str, dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.0), facecolor="white",
                             squeeze=False)
    for ax, (title, rq1) in zip(axes[0], panels):
        _rq1_panel(ax, rq1, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _hdr(name: str, n: int, small: bool) -> str:
    return f"{name} (n={n}{'*' if small else ''})"


def _fmt_p(pv: float | None, B: int | None, label: str = "p") -> str:
    if pv is None:
        return ""
    if pv == 0 and B:
        return f"{label}<{2.0 / B:.4g}"
    if pv < 0.0005:
        return f"{label}<0.001"
    return f"{label}={pv:.3f}"


def _fmt_delta(d: dict, p: bool = True) -> str:
    if d.get("delta") is None:
        return "n/a"
    text = f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    if not p or d.get("p_value") is None:
        return text + (" (CI only)" if p else "")
    b = d.get("B")
    parts = [_fmt_p(d["p_value"], b)]
    if d.get("p_holm") is not None:
        parts.append(_fmt_p(d["p_holm"], None, "Holm p"))
    return text + " (" + ", ".join(parts) + ")"


def _fmt_rate(d: dict) -> str:
    if d.get("rate") is None or d.get("na"):
        return f"n/a (n={d.get('n', 0)})"
    return f"{d['rate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}] (n={d['n']}" \
           f"{', small' if d['small_n'] else ''})"


def _fmt_diff(r: dict) -> str:
    if r["term1"].get("na") or r["term0"].get("na"):
        return "n/a"
    return _fmt_delta(r["difference"], p=False)


def _error_tables(rates: dict, label: str) -> list[str]:
    """Raw error, label-balanced error and gold-label mix tables for one block."""
    header = "| Model | " + " | ".join(
        _hdr(p, rates[p]["n_tag1"], rates[p]["small_n"]) for p in PHENOMENA) + " |"
    rule = "|---|" + "---|" * len(PHENOMENA)
    models = list(rates[PHENOMENA[0]]["models"])
    lines = [f"{label} raw error rate (1 - accuracy), seed mean [95% CI]:", "", header, rule]
    for m in models:
        lines.append("| " + _name(m) + " | " + " | ".join(
            _fmt_ci(rates[p]["models"][m]["tag1"]) for p in PHENOMENA) + " |")
    lines += ["", f"{label} label-balanced error (mean over the gold classes present of the class "
              "error rate) [95% CI]:", "", header, rule]
    for m in models:
        lines.append("| " + _name(m) + " | " + " | ".join(
            _fmt_ci(rates[p]["models"][m]["tag1"], "balanced_error") for p in PHENOMENA) + " |")
    lines += ["", f"{label} gold-label mix of each subset (negative / neutral / positive):", "",
              "| Phenomenon | tag = 1 | tag = 0 |", "|---|---|---|"]
    for p in PHENOMENA:
        a, b = rates[p]["gold_mix_tag1"], rates[p]["gold_mix_tag0"]
        lines.append(f"| {_hdr(p, rates[p]['n_tag1'], rates[p]['small_n'])} | "
                     f"{a['n_neg']} / {a['n_neu']} / {a['n_pos']} | "
                     f"{b['n_neg']} / {b['n_neu']} / {b['n_pos']} |")
    return lines + [""]


def _attribution_table(attr: dict) -> list[str]:
    lines = ["| Phenomenon | n | delta_arch [95% CI] (p, Holm p) | delta_domain [95% CI] (p, Holm p) |",
             "|---|---|---|---|"]
    for p in PHENOMENA:
        a = attr[p]
        lines.append(f"| {p}{' *' if a['small_n'] else ''} | {a['n']} | "
                     f"{_fmt_delta(a['delta_arch'])} | {_fmt_delta(a['delta_domain'])} |")
    return lines + [""]


def _rq1_table(rq1: dict, with_diff: bool) -> list[str]:
    head = "| Model | RQ1 term=1 | RQ1 term=0 |" + (" difference |" if with_diff else "")
    lines = [head, "|---|---|---|" + ("---|" if with_diff else "")]
    for m, r in rq1.items():
        row = f"| {_name(m)} | {_fmt_rate(r['term1'])} | {_fmt_rate(r['term0'])} |"
        lines.append(row + (f" {_fmt_diff(r)} |" if with_diff else ""))
    return lines + [""]


def _cue_table(cues: dict) -> list[str]:
    if not cues:
        return ["No gold-neutral sentence carries a financial-terminology cue.", ""]
    models = list(next(iter(cues.values()))["models"])
    lines = ["| Cue | n | " + " | ".join(_name(m) for m in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for cue, c in cues.items():
        cells = ["n/a" if c["na"] else f"{c['models'][m]['rate']:.3f}" for m in models]
        lines.append(f"| {cue}{' *' if c['small_n'] else ''} | {c['n']} | " + " | ".join(cells) + " |")
    return lines + [""]


def _report(path: str, *, annotators: dict, key: dict, agreement: dict, mode: str,
            diag: dict, tfidf_best: str, tfidf_scores: dict, full: dict, args: Any,
            warnings: list[str]) -> None:
    a_name = annotators.get("a") or "not recorded"
    b_name = annotators.get("b") or "not recorded"
    kappas = [agreement[p]["kappa_ab"] for p in PHENOMENA if agreement[p]["kappa_ab"] is not None]
    mean_kappa = float(np.mean(kappas)) if kappas else None
    B = args.bootstrap_samples
    lines = [
        "# Step 3 Report: Phenomenon-Level Error and Attribution Analysis",
        "",
        "## 1. Annotation and agreement",
        "",
        "Annotators A and B are two independent LLM annotators, not human annotators. "
        f"Annotator A was `{a_name}` and annotator B was `{b_name}`. Both labelled the same "
        f"{key.get('n', 'N')} sentences blind (no gold label, no model prediction) using the same "
        "guidelines. Both models may have seen Financial PhraseBank during pretraining; blindness "
        "refers to the labels and predictions withheld in this run. All agreement numbers below "
        "therefore describe agreement between two language models and must not be read as "
        "inter-human agreement.",
        "",
        f"The sentences were drawn from the `{key.get('config')}` test split "
        f"(seed {key.get('seed')}). Mean Cohen's kappa between A and B over the five phenomena is "
        f"{_fmt(mean_kappa)}.",
        "",
        "| Phenomenon | kappa A-B | kappa A-rule | kappa B-rule | raw agreement A-B | "
        "prevalence A / B / rule | consensus differs from rule (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in PHENOMENA:
        r = agreement[p]
        lines.append(
            f"| {p} | {_fmt(r['kappa_ab'])} | {_fmt(r['kappa_a_rule'])} | "
            f"{_fmt(r['kappa_b_rule'])} | {_fmt(r['raw_agreement_ab'])} | "
            f"{_fmt(r['prevalence_a'])} / {_fmt(r['prevalence_b'])} / "
            f"{_fmt(r['prevalence_rule'])} | {r['n_consensus_vs_rule_diff']} of {r['n']} |")
    lines += [
        "",
        "The last column counts the sentences where the both-annotators consensus differs from the "
        "rule tag. Where it is close to 0 the LLM pass confirmed the rule tagger, so the kappa "
        "against the rules measures cue-word matching rather than construct validity.",
        "",
        f"Consensus tags use the `{mode}` rule "
        + ("(a sentence carries a phenomenon only when both annotators marked it)."
           if mode == "both" else "(a sentence carries a phenomenon when either annotator marked it)."),
        "",
        "## Protocol",
        "",
        f"- Bootstrap: B = {B}; every resample draws sentences with replacement and, independently "
        "per model, draws its trained seeds with replacement (a baseline has one run), so every "
        "interval includes seed variation.",
        "- p = min(1, 2 * min(P(d <= 0), P(d >= 0))) over the bootstrap draws of the difference d; "
        f"a zero is printed as p<{2.0 / B:.4g} (= 2/B).",
        "- Holm-Bonferroni is applied within each attribution table (one split block, 5 phenomena "
        "x 2 deltas, over the rows that are tested). `Holm p` is the adjusted value.",
        "- Rows marked `*` have fewer than 20 tagged sentences: only the CI is shown, no p-value. "
        "RQ1 cells with fewer than 5 sentences print n/a.",
        "- Error rates are shown raw and label-balanced (mean over the gold classes present of the "
        "class error rate), because the gold-label mix of a tagged subset changes the raw rate. "
        "The gold-label mix of every subset is listed.",
        "",
        "## 2. Per-phenomenon error rates on the diagnostic subset",
        "",
        f"Error on the sentences tagged with the phenomenon (annotator consensus). Rows and "
        "columns marked `*` have fewer than 20 tagged sentences and are unreliable.",
        "",
    ]
    lines += _error_tables(diag["error_rates"], "Diagnostic subset:")
    lines += [
        "![error heatmap](step3_phenomenon_error_heatmap.png)",
        "",
        "The heatmap shows the label-balanced error.",
        "",
        "## 3. Architecture versus domain-pretraining attribution",
        "",
        f"`delta_arch` = accuracy(bert_base) - accuracy({tfidf_best}) and `delta_domain` = "
        "accuracy(finbert_mlm) - accuracy(bert_base) on the tagged sentences. "
        f"`{tfidf_best}` was chosen by validation Macro-F1 (logreg "
        f"{_fmt(tfidf_scores['tfidf_logreg'], 4)}, svm {_fmt(tfidf_scores['tfidf_svm'], 4)}).",
        "",
    ]
    lines += _attribution_table(diag["attribution"])
    lines += ["![attribution](step3_attribution_by_phenomenon.png)", "",
              "## 4. RQ1: financial terminology and neutral-to-negative errors", "",
              "Neutral-to-negative rate on gold-neutral sentences, with and without a "
              "financial-terminology tag. Annotator consensus on the diagnostic subset, then rule "
              "tags on the full splits.", ""]
    lines += _rq1_table(diag["rq1_financial_terminology"], with_diff=False)
    lines += ["![rq1](step3_rq1_accounting_terms.png)", ""]
    if full:
        lines += ["## 5. Full-split analysis with rule tags only", "",
                  "No annotators are involved here: tags come from the rule-based pre-tagger "
                  "(`src/analysis/phenomena.py`), which is cruder than the annotators but covers "
                  "every test sentence.", ""]
        for config, entry in full.items():
            for split, label in (("test", "test split"), ("external", "SEntFiN holdout")):
                block = entry[split]
                lines += [f"### {config}: {label} (N = {block['n']}, "
                          f"tfidf_best = {entry['tfidf_best']})", ""]
                lines += _error_tables(block["error_rates"], "Rule tags:")
                lines += _attribution_table(block["attribution"])
                lines += ["RQ1, all financial-terminology sentences:", ""]
                lines += _rq1_table(block["rq1_financial_terminology"], with_diff=True)
                if "rq1_excluding_loss" in block:
                    lines += [f"RQ1, excluding sentences whose only cue is loss/losses "
                              f"({block.get('n_loss_only_neutral_dropped', 0)} gold-neutral "
                              "sentences dropped from the term=1 group):", ""]
                    lines += _rq1_table(block["rq1_excluding_loss"], with_diff=True)
                    lines += ["RQ1 by cue word (neutral-to-negative rate on gold-neutral sentences, "
                              "n = sentences carrying the cue; a sentence counts once per cue; "
                              "n/a below 5): the effect can be carried by a single cue, and "
                              "label conventions differ between datasets (for example, reported "
                              "losses are mostly negative or positive in Financial PhraseBank "
                              "but often neutral in SEntFiN).", ""]
                    lines += _cue_table(block["rq1_cue_breakdown"])
    if warnings:
        lines += ["## Notes", ""] + [f"- {w}" for w in sorted(set(warnings))] + [""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", default="data/diagnostic")
    parser.add_argument("--runs-dir", default="results/step2/runs")
    parser.add_argument("--out-dir", default="results/step3")
    parser.add_argument("--consensus", choices=("both", "either"), default="both")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Step 1 manifests and sentfin_dedup.json, for the full-split analysis.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, tagger: Callable | None = None) -> int:
    args = _parse(argv)
    if tagger is None:
        from src.analysis.phenomena import tag_many as tagger  # lazy: group 4's module
    warnings: list[str] = []
    try:
        diag_dir = args.diagnostic_dir
        key = _load_json(os.path.join(diag_dir, "diagnostic_key.json"))
        ann_a = _read_ann(os.path.join(diag_dir, "annotations_a.csv"))
        ann_b = _read_ann(os.path.join(diag_dir, "annotations_b.csv"))
        rules = _read_ann(os.path.join(diag_dir, "rule_tags.csv"))
        ann_path = os.path.join(diag_dir, "annotators.json")
        annotators = _load_json(ann_path) if os.path.isfile(ann_path) else {}
        if not annotators:
            warnings.append("annotators.json not found; annotator model names not recorded.")
        config = key["config"]
        items = {it["diag_id"]: it for it in key["items"]}
        for name, ann in (("annotations_a.csv", ann_a), ("annotations_b.csv", ann_b),
                          ("rule_tags.csv", rules)):
            absent = sorted(set(items) - set(ann))
            if absent:
                raise MissingInput(f"{os.path.join(diag_dir, name)} lacks diag_id(s) {absent[:5]}")
        runs = _config_preds(args.runs_dir, config, "test")
        best, scores = _pick_tfidf_best(args.runs_dir, config)

        agreement = agreement_table(ann_a, ann_b, rules)
        consensus = consensus_tags(ann_a, ann_b, args.consensus)
        by_record = {items[d]["record_id"]: tags for d, tags in consensus.items() if d in items}
        y_true = {items[d]["record_id"]: int(items[d]["label"]) for d in consensus if d in items}
        for model, seed_preds in runs.items():
            for preds in seed_preds:
                lost = sorted(set(y_true) - set(preds))
                if lost:
                    raise MissingInput(
                        f"{model} test predictions in {args.runs_dir}/{config} lack "
                        f"record_id(s) {lost[:5]}")
        diag = _split_analysis(by_record, y_true, runs, best, args.bootstrap_samples, args.seed)
        full = _full_split(args.runs_dir, args.splits_dir, tagger, args.bootstrap_samples,
                           args.seed, warnings)
    except MissingInput as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"ERROR: malformed input, missing key {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    _write_json(os.path.join(args.out_dir, "annotation_agreement.json"), {
        "annotators": {"a": annotators.get("a"), "b": annotators.get("b"),
                       "note": "two independent LLM annotators, not human"},
        "config": config, "n_items": len(items), "consensus_mode": args.consensus,
        "agreement": agreement,
    })
    _write_json(os.path.join(args.out_dir, "phenomenon_analysis.json"), {
        "config": config, "consensus_mode": args.consensus, "tfidf_best": best,
        "tfidf_val_macro_f1": scores, "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed, "diagnostic": diag, "full_split_rule_tags": full,
    })
    _heatmap(os.path.join(args.out_dir, "step3_phenomenon_error_heatmap.png"),
             diag["error_rates"], f"Error rate on tagged sentences ({args.consensus} consensus)")
    _attribution_plot(os.path.join(args.out_dir, "step3_attribution_by_phenomenon.png"),
                      diag["attribution"], f"Attribution by phenomenon ({args.consensus} consensus)")
    if config in full:
        ext = full[config]["external"]["rq1_financial_terminology"]
        tst = full[config]["test"]["rq1_financial_terminology"]
        panels = [("SEntFiN holdout, rule tags (models trained on "
                   f"{config.replace('sentences_', '')})", ext),
                  (f"{config.replace('sentences_', '')} test split, rule tags", tst)]
    else:
        panels = [("Diagnostic subset (annotator consensus)", diag["rq1_financial_terminology"])]
    _rq1_plot(os.path.join(args.out_dir, "step3_rq1_accounting_terms.png"), panels)
    _report(os.path.join(args.out_dir, "STEP3_REPORT.md"), annotators=annotators, key=key,
            agreement=agreement, mode=args.consensus, diag=diag, tfidf_best=best,
            tfidf_scores=scores, full=full, args=args, warnings=warnings)
    print(f"Wrote Step 3 outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
