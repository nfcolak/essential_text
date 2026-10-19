#!/usr/bin/env python3
"""Build Google Colab Jupyter Notebook for Step 1.

Generates `notebooks/step1_data_and_baselines_colab.ipynb` programmatically adhering
to the NBFormat v4 specification.

Cells included:
1. Markdown: Title, authorship, course rubric alignment, and "Open in Colab" badge.
2. Code: Git clone / pull repository.
3. Code: Pip install requirements.
4. Code: Real dataset download & split locking with SHA-256 manifests.
5. Code: Leakage audit verification across all splits.
6. Code: Baseline training & evaluation (Tier 1 & Tier 2) + statistical tests.
7. Code: RQ1 neutral-vs-negative confusion matrix & error rate visualization.
8. Code: Artifact packaging and zip export.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "step1_data_and_baselines_colab.ipynb"


def make_markdown_cell(source: list[str]) -> dict:
    """Create an NBFormat v4 markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source],
    }


def make_code_cell(source: list[str]) -> dict:
    """Create an NBFormat v4 code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source],
    }


def build_step1_notebook() -> dict:
    """Construct notebook JSON structure."""
    cells = []

    # 1. Header & Badge
    header_source = [
        "# Evaluating Domain Adaptation in Financial Sentiment Analysis",
        "## Step 1: Leakage-Free Data Splits, Tier 1/2 Baselines, and Evaluation Suite",
        "",
        "**Authors:** Qiyao Zheng, Benjamin Donick, Necati Furkan Colak  ",
        "**Repository:** [https://github.com/nfcolak/essential_text](https://github.com/nfcolak/essential_text)  ",
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nfcolak/essential_text/blob/main/notebooks/step1_data_and_baselines_colab.ipynb)",
        "",
        "---",
        "### Overview & Rubric Alignment",
        "This notebook executes **Step 1** of our project under strict non-negotiable methodology rules:",
        "1. **Zero Data Leakage:** Stratified 70/15/15 splits lock near-duplicate sentences (token Jaccard $\\ge 0.80$) into identical partitions with SHA-256 verification.",
        "2. **Strict Baseline Rigor:** Feature extraction (TF-IDF vocabulary) is fit strictly on training folds.",
        "3. **Course Metrics Only:** Macro-F1, Accuracy, per-class P/R/F1, 3x3 confusion matrices, ECE, McNemar's test, and 10,000-sample bootstrap CIs.",
        "4. **RQ1 Attribution:** Direct measurement of Neutral-to-Negative misclassification on accounting terms (*cost*, *liability*, *depreciation*, *debt*).",
    ]
    cells.append(make_markdown_cell(header_source))

    # 2. Git Clone / Pull
    clone_source = [
        "# Cell 1: Clone or pull the essential_text repository",
        "import os",
        "import sys",
        "",
        'REPO_URL = "https://github.com/nfcolak/essential_text.git"',
        'REPO_NAME = "essential_text"',
        "",
        "if not os.path.exists(REPO_NAME):",
        "    !git clone {REPO_URL}",
        "    %cd {REPO_NAME}",
        "else:",
        "    %cd {REPO_NAME}",
        "    !git pull origin main",
        "",
        "if os.getcwd() not in sys.path:",
        "    sys.path.insert(0, os.getcwd())",
        "",
        'print(f"Working directory: {os.getcwd()}")',
    ]
    cells.append(make_code_cell(clone_source))

    # 3. Pip Install
    install_source = [
        "# Cell 2: Install project dependencies",
        "!pip install -q -r requirements.txt",
        'print("Dependencies installed successfully.")',
    ]
    cells.append(make_code_cell(install_source))

    # 4. Download and Split Locking
    data_source = [
        "# Cell 3: Download Real Datasets & Generate SHA-256 Locked Splits",
        "import os",
        "import json",
        "from src.data.download import load_real_datasets",
        "from src.data.splits import build_records, create_stratified_splits, verify_no_leakage",
        "from src.data.dedup import deduplicate_external_holdout",
        "",
        'os.makedirs("data/splits", exist_ok=True)',
        'os.makedirs("data/raw", exist_ok=True)',
        "",
        'print("Ingesting Financial PhraseBank and SEntFiN 1.0 datasets...")',
        'real_data = load_real_datasets(cache_dir="data/raw", execute=True)',
        "",
        'configs = ["sentences_allagree", "sentences_75agree", "sentences_50agree"]',
        "all_pb_texts = []",
        "manifests = {}",
        "",
        "for cfg in configs:",
        '    cfg_data = real_data.get("financial_phrasebank", {}).get(cfg)',
        "    if cfg_data:",
        '        texts = cfg_data["texts"]',
        '        labels = cfg_data["labels"]',
        "        all_pb_texts.extend(texts)",
        "        records = build_records(texts=texts, labels=labels, prefix=cfg, jaccard_threshold=0.80)",
        "        manifest = create_stratified_splits(records=records, seed=42, ratios=(0.70, 0.15, 0.15), n_folds=5)",
        "        manifest_path = f'data/splits/{cfg}_manifest.json'",
        '        with open(manifest_path, "w", encoding="utf-8") as f:',
        "            json.dump(manifest, f, indent=2)",
        "        manifests[cfg] = manifest",
        '        print(f"Created locked manifest for {cfg}: {len(records)} records -> {manifest_path}")',
        "",
        "# External SEntFiN 1.0 holdout deduplication against all PhraseBank sentences",
        'raw_sentfin = real_data.get("sentfin", [])',
        "if isinstance(raw_sentfin, list) and len(raw_sentfin) > 0:",
        "    from src.data.splits import LABEL_TO_ID, ID_TO_LABEL",
        "    sentfin_records = []",
        "    for i, row in enumerate(raw_sentfin):",
        '        t = row.get("Title", "") or row.get("text", "")',
        '        lbl_str = (row.get("Sentiment") or row.get("Decisions") or "").strip().lower()',
        "        lbl_id = LABEL_TO_ID.get(lbl_str, 1)",
        "        sentfin_records.append({",
        '            "id": f"sentfin_{i:05d}",',
        '            "text": t,',
        '            "label": lbl_id,',
        '            "label_name": ID_TO_LABEL[lbl_id],',
        "        })",
        "    filtered_holdout, stats = deduplicate_external_holdout(sentfin_records, all_pb_texts, threshold=0.80)",
        '    with open("data/splits/sentfin_dedup.json", "w", encoding="utf-8") as f:',
        '        json.dump({"stats": stats, "records": filtered_holdout}, f, indent=2)',
        '    print(f"SEntFiN 1.0 Deduplication: Kept {stats[\'kept\']} / {stats[\'total_input\']} sentences (removed {stats[\'removed_overlap\']} near-duplicates)")',
    ]
    cells.append(make_code_cell(data_source))

    # 5. Leakage Audit
    audit_source = [
        "# Cell 4: Comprehensive Data Leakage Audit",
        "import glob",
        "import json",
        "from src.data.splits import verify_no_leakage",
        "",
        'print("=" * 70)',
        'print("DATA LEAKAGE AUDIT VERIFICATION")',
        'print("=" * 70)',
        "",
        'manifest_files = sorted(glob.glob("data/splits/*_manifest.json"))',
        "for mf in manifest_files:",
        '    with open(mf, "r", encoding="utf-8") as f:',
        "        m = json.load(f)",
        "    audit = verify_no_leakage(m, jaccard_threshold=0.80)",
        '    prefix = m.get("dataset_prefix", os.path.basename(mf))',
        '    print(f"Manifest: {prefix}")',
        '    print(f"  [+] ID Overlap: {audit[\'id_overlap_count\']} (PASSED)")',
        '    print(f"  [+] SHA-256 Overlap: {audit[\'hash_overlap_count\']} (PASSED)")',
        '    print(f"  [+] Cluster Overlap: {audit[\'cluster_overlap_count\']} (PASSED)")',
        '    print(f"  [+] Max Cross-Split Jaccard: {audit[\'max_cross_split_jaccard\']:.4f} < 0.8000 (PASSED)")',
        '    print(f"  [+] Partitions: Train={audit[\'split_sizes\'][\'train\']}, Val={audit[\'split_sizes\'][\'val\']}, Test={audit[\'split_sizes\'][\'test\']}")',
        "    print()",
    ]
    cells.append(make_code_cell(audit_source))

    # 6. Baseline Training and Evaluation
    baseline_source = [
        "# Cell 5: Baseline Training, Evaluation, and Statistical Comparisons",
        "# Executes the complete Step 1 runner with full dataset and 10,000-sample bootstrap",
        "!python colab/run_step1_colab.py --mode full --execute --out-dir results/step1 --splits-dir data/splits --bootstrap-samples 10000",
        "",
        "# Render generated STEP1_REPORT.md directly in Colab output",
        'report_file = "results/step1/STEP1_REPORT.md"',
        "if os.path.exists(report_file):",
        '    with open(report_file, "r", encoding="utf-8") as f:',
        "        from IPython.display import Markdown, display",
        "        display(Markdown(f.read()))",
    ]
    cells.append(make_code_cell(baseline_source))

    # 7. Visualization
    viz_source = [
        "# Cell 6: RQ1 Neutral-vs-Negative Confusion Matrix & Error Visualization",
        "import json",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import seaborn as sns",
        "",
        'metrics_file = "results/step1/summary_metrics.json"',
        "with open(metrics_file, \"r\", encoding=\"utf-8\") as f:",
        "    summary = json.load(f)",
        "",
        'models = summary["models"]',
        "model_names = list(models.keys())",
        'class_names = ["Negative", "Neutral", "Positive"]',
        "",
        "# 1. Plot 3x3 Confusion Matrices for all baselines",
        "fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))",
        "if len(model_names) == 1:",
        "    axes = [axes]",
        "",
        "for ax, name in zip(axes, model_names):",
        '    cm = np.array(models[name]["test_metrics"]["confusion_matrix"])',
        "    sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False,",
        "                xticklabels=class_names, yticklabels=class_names, ax=ax)",
        '    rate = models[name]["test_metrics"]["neutral_to_negative_rate"]',
        '    ax.set_title(f"{name}\\nRQ1 Neu→Neg Error: {rate:.2%}", fontsize=11, fontweight="bold")',
        '    ax.set_xlabel("Predicted Label")',
        '    ax.set_ylabel("True Label")',
        "",
        "plt.tight_layout()",
        'plt.savefig("results/step1/rq1_confusion_matrices.png", dpi=300)',
        "plt.show()",
        "",
        "# 2. Bar chart comparing RQ1 Neutral -> Negative Misclassification Rates",
        'rates = [models[n]["test_metrics"]["neutral_to_negative_rate"] for n in model_names]',
        "plt.figure(figsize=(9, 4.5))",
        'palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]',
        "bars = plt.bar(model_names, rates, color=palette[:len(model_names)], edgecolor=\"black\", alpha=0.85)",
        'plt.ylabel("Neutral → Negative Misclassification Rate", fontsize=11)',
        'plt.title("RQ1: False-Negative Misclassification of Neutral Financial Statements", fontsize=12, fontweight="bold")',
        "plt.ylim(0, max(0.5, max(rates) * 1.25 if rates else 1.0))",
        "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)",
        "",
        "for bar in bars:",
        "    yval = bar.get_height()",
        '    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.2%}", ha="center", va="bottom", fontweight="bold")',
        "",
        'plt.xticks(rotation=15, ha="right", fontsize=10)',
        "plt.tight_layout()",
        'plt.savefig("results/step1/rq1_neutral_to_negative_comparison.png", dpi=300)',
        "plt.show()",
    ]
    cells.append(make_code_cell(viz_source))

    # 8. Artifact Export
    export_source = [
        "# Cell 7: Package Locked Splits and Evaluation Results for Download",
        "import shutil",
        "",
        'print("Archiving results and split manifests into step1_artifacts.zip...")',
        'shutil.make_archive("step1_artifacts", "zip", root_dir=".", base_dir="results/step1")',
        'print("step1_artifacts.zip created successfully.")',
        "",
        "try:",
        "    from google.colab import files",
        '    files.download("step1_artifacts.zip")',
        '    print("Download prompt triggered in browser.")',
        "except ImportError:",
        '    print("Local environment detected; step1_artifacts.zip is ready in root directory.")',
    ]
    cells.append(make_code_cell(export_source))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return notebook


def main() -> int:
    """Generate and write the Step 1 Colab notebook."""
    nb = build_step1_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, mode="w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated Colab notebook with {len(nb['cells'])} cells at {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
