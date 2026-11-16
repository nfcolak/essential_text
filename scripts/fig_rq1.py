"""RQ1 figure: how are gold-neutral sentences labelled?

Diverging horizontal bars per model: the share of gold-neutral sentences that a
model calls negative (left of zero) or positive (right of zero), for the
PhraseBank allagree test split and for SEntFiN. The share labelled neutral
(correct) is printed as a number column. Fine-tuned models are averaged over
their seeds.

Sized for a 15 cm text width (11pt article, 3 cm side margins) and set in
Latin Modern Roman so that it matches the report's body font.

Usage: python fig_rq1.py <repo_root> <out_dir>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path

REPO, OUT = sys.argv[1], sys.argv[2]
RUNS = os.path.join(REPO, "results", "step2", "runs", "sentences_allagree")

# (display name, run directory, seeds, row position). The gap between rows 1 and 2
# separates the two lexicons from the three trained models.
MODELS = [
    ("VADER", "vader_official", ["seed42"], 0.0),
    ("Loughran–McDonald", "lm_official", ["seed42"], 1.0),
    ("TF-IDF + SVM", "tfidf_svm", ["seed42"], 2.35),
    ("BERT-base", "bert_base", ["seed42", "seed100", "seed2026"], 3.35),
    ("FinBERT-MLM", "finbert_mlm", ["seed42", "seed100", "seed2026"], 4.35),
]
PANELS = [("test", "(a) PhraseBank test"), ("external", "(b) SEntFiN")]

# Palette validated with the dataviz skill's validate_palette.js on a white surface.
NEG, POS = "#e34948", "#2a78d6"
INK, INK2 = "#000000", "#444444"
GRID, AXIS = "#dcdcdc", "#555555"

TEXT_WIDTH_IN = 15.0 / 2.54

# Latin Modern Roman at its 10pt optical size (the report's body font). Loaded by file,
# because the family name alone can resolve to the wider 5pt or 7pt optical sizes.
LM_DIR = os.environ.get("LM_FONT_DIR", "/usr/share/texmf/fonts/opentype/public/lm")


def lm(size: float, italic: bool = False) -> FontProperties:
    name = "lmroman10-italic.otf" if italic else "lmroman10-regular.otf"
    return FontProperties(fname=os.path.join(LM_DIR, name), size=size)


def neutral_shares(model: str, seeds: list[str], split: str) -> tuple[int, np.ndarray]:
    """Mean % of gold-neutral sentences predicted negative / neutral / positive."""
    rows, n = [], 0
    for seed in seeds:
        z = np.load(os.path.join(RUNS, model, seed, "predictions.npz"), allow_pickle=True)
        y, pred = z[f"{split}_y"], z[f"{split}_prob"].argmax(1)
        mask = y == 1
        n = int(mask.sum())
        rows.append([((pred == c) & mask).sum() / n * 100 for c in (0, 1, 2)])
    return n, np.mean(rows, axis=0)


def bar(ax, y: float, value: float, height: float, color: str, r_pt: float = 1.0) -> None:
    """Bar from the zero baseline to `value`: square at the baseline, softened at the data end."""
    if abs(value) < 1e-9:
        return
    fig = ax.figure
    r_px = r_pt * fig.dpi / 72.0
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 1.0))
    dx = min(r_px / abs(p1[0] - p0[0]), abs(value))
    dy = min(r_px / abs(p1[1] - p0[1]), height / 2)
    s = 1.0 if value > 0 else -1.0
    lo, hi = y - height / 2, y + height / 2
    verts = [
        (0, lo), (value - s * dx, lo), (value, lo), (value, lo + dy),
        (value, hi - dy), (value, hi), (value - s * dx, hi), (0, hi), (0, lo),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.9), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.995, top=0.875, bottom=0.225, wspace=0.2)
    xlim, col_x = (-16.0, 53.0), 52.5
    height = 0.6
    y_top, y_bottom = -0.42, MODELS[-1][3] + 0.42
    header_y = -0.86

    for ax, (split, title) in zip(axes, PANELS):
        ax.set_xlim(*xlim)
        ax.set_ylim(y_bottom + 0.05, header_y - 0.35)
        ax.set_xticks([-10, 0, 10, 20, 30])
        ax.set_xticklabels(["10", "0", "10", "20", "30"], color=INK)
        ax.tick_params(axis="x", length=2.5, width=0.5, color=AXIS, pad=2)
        ax.tick_params(axis="y", length=0, pad=4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_bounds(-10, 30)
        ax.spines["bottom"].set_position(("data", y_bottom))
        for x in (-10, 10, 20, 30):
            ax.plot([x, x], [y_top, y_bottom], color=GRID, lw=0.5, zorder=1)
        ax.plot([0, 0], [y_top, y_bottom], color=AXIS, lw=0.6, zorder=4)
        fig.canvas.draw()

        n_neutral = 0
        for name, key, seeds, y in MODELS:
            n_neutral, (neg, neu, pos) = neutral_shares(key, seeds, split)
            print(f"{split:9s} {name:20s} neg {neg:5.1f}  neu {neu:5.1f}  pos {pos:5.1f}")
            bar(ax, y, -neg, height, NEG)
            bar(ax, y, pos, height, POS)
            num = {"va": "center", "color": INK, "fontproperties": lm(8.5)}
            ax.text(-neg - 0.8, y, f"{neg:.1f}", ha="right", **num)
            ax.text(pos + 0.8, y, f"{pos:.1f}", ha="left", **num)
            ax.text(col_x, y, f"{neu:.1f}", ha="right", **num)

        for label in ax.get_xticklabels():
            label.set_fontproperties(lm(8.5))
        print(f"{split}: {n_neutral:,} gold-neutral sentences")
        ax.set_title(title, loc="left", color=INK, pad=5, fontproperties=lm(9.5))
        italic = {"fontproperties": lm(8.5, italic=True), "color": INK2, "va": "center"}
        ax.text(-0.8, header_y, "called negative", ha="right", **italic)
        ax.text(0.8, header_y, "called positive", ha="left", **italic)
        ax.text(col_x, header_y, "correct", ha="right", **italic)

    axes[0].set_yticks([m[3] for m in MODELS])
    axes[0].set_yticklabels([m[0] for m in MODELS], color=INK, fontproperties=lm(9.5))
    fig.text(0.61, 0.04, "Share of gold-neutral sentences (%)", ha="center", va="center",
             color=INK, fontproperties=lm(9))

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "rq1_neutral_errors.pdf"), facecolor="white")
    fig.savefig(os.path.join(OUT, "rq1_neutral_errors.png"), dpi=400, facecolor="white")


if __name__ == "__main__":
    main()
