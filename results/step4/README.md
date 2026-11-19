# Step 4: independent hyperparameter tuning

| File | Content |
|---|---|
| `STEP4_REPORT.md` | Protocol, per-seed scores, paired comparisons (seed + sentence bootstrap, McNemar + Holm), gap change; written by `scripts/summarize_step4.py` |
| `step4_statistics.json` | All numbers behind `STEP4_REPORT.md` (`python3 scripts/summarize_step4.py --check` recomputes them) |
| `RESULTS_SUMMARY.md` | Short summary written by `src/analysis/tuning_figures.py` |
| `hyperparameter_heatmaps.png/.pdf` | Validation Macro-F1 of the 2 x 2 grid per model (seed 42) |
| `final_model_comparison.png/.pdf` | Three-seed means of the selected recipes on PhraseBank test and SEntFiN |
| `original_vs_tuned.png/.pdf` | Fixed Step 2 recipe against the tuned recipe |
| `per_seed_performance.png/.pdf` | Per-seed Macro-F1 of the tuned runs |

The runs themselves (with per-sentence predictions) are in `results/tuning/`. All runs, the fixed-recipe Step 2 runs
included, come from one environment (Apple MPS, Python 3.9.6, torch 2.8.0, transformers 4.57.6).
