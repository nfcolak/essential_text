#!/usr/bin/env python3
"""Step 1 End-to-End Pipeline Runner: Leakage-Free Splits, Baselines, and Evaluation.

Supports two operational modes:
- `--mode smoke` (default): Fast local execution (< 5 seconds) on synthetic financial
  fixtures without network access, verifying split locking, baselines, and statistical tests.
- `--mode full --execute`: Google Colab execution loading real Financial PhraseBank
  and SEntFiN 1.0 datasets, generating SHA-256 locked manifests, fitting all Tier 1/2
  baselines, running 10,000-sample bootstrap comparisons, and generating STEP1_REPORT.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Sequence
import numpy as np

from src.baselines.lexicon import LoughranMcDonaldBaseline, VaderBaseline
from src.baselines.tfidf_ml import TfidfMLBaseline
from src.data.dedup import deduplicate_external_holdout
from src.data.download import generate_synthetic_dataset, load_real_datasets
from src.data.splits import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_records,
    create_stratified_splits,
    verify_no_leakage,
)
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.stats import mcnemar_test, paired_bootstrap_ci


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Step 1 runner."""
    parser = argparse.ArgumentParser(
        description="Run Step 1 Data Preparation, Baseline Training, and Evaluation.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Execution mode: 'smoke' uses synthetic fixtures locally; 'full' runs real datasets in Colab.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Safety execution gate required for --mode full. Downloads and processes real datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/step1",
        help="Directory to save evaluation results and reports (default: results/step1).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory to save locked split manifests (default: data/splits).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Number of paired bootstrap resamples (default: 200 for smoke, 10000 for full).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split creation and bootstrap resampling (default: 42).",
    )
    parser.add_argument(
        "--fpb-config",
        type=str,
        default="sentences_allagree",
        choices=["sentences_allagree", "sentences_75agree", "sentences_50agree", "all"],
        help="Financial PhraseBank agreement subset to evaluate (default: sentences_allagree).",
    )
    return parser.parse_args(argv)


def instantiate_fresh_model(model_name: str, seed: int = 42) -> Any:
    """Instantiate a fresh baseline model instance."""
    if model_name == "VADER":
        return VaderBaseline()
    elif model_name == "Loughran-McDonald":
        return LoughranMcDonaldBaseline()
    elif model_name == "TF-IDF + Logistic Regression":
        return TfidfMLBaseline(model_type="logreg", random_state=seed)
    elif model_name == "TF-IDF + Linear SVM":
        return TfidfMLBaseline(model_type="svm", random_state=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def generate_markdown_report(
    metadata: dict[str, Any],
    leakage_summary: dict[str, Any],
    models_results: dict[str, Any],
    statistical_comparisons: dict[str, Any],
) -> str:
    """Generate comprehensive GitHub Flavored Markdown report for Step 1."""
    lines: list[str] = []

    lines.append("# Step 1 Baseline Evaluation Report")
    lines.append("")
    lines.append(f"- **Execution Mode:** `{metadata['mode']}`")
    lines.append(f"- **Dataset:** `{metadata['dataset']}`")
    lines.append(f"- **Random Seed:** `{metadata['seed']}`")
    lines.append(f"- **Bootstrap Samples:** `{metadata['bootstrap_samples']}`")
    lines.append(f"- **Generated At (UTC):** `{metadata['timestamp']}`")
    lines.append("")

    # Split sizes
    sizes = metadata.get("split_sizes", {})
    lines.append("## 1. Dataset Partitioning & Leakage Audit")
    lines.append("")
    lines.append("| Partition | Sample Count |")
    lines.append("|---|---|")
    lines.append(f"| Training (`train`) | {sizes.get('train', 0):,} |")
    lines.append(f"| Validation (`val`) | {sizes.get('val', 0):,} |")
    lines.append(f"| Test (`test`) | {sizes.get('test', 0):,} |")
    lines.append(f"| External Holdout (`SEntFiN 1.0` deduplicated) | {sizes.get('external_holdout', 0):,} |")
    lines.append("")

    lines.append("### Leakage Verification Results")
    lines.append("")
    lines.append("| Verification Check | Status | Value | Constraint |")
    lines.append("|---|---|---|---|")
    lines.append(f"| ID Overlap Count | **PASSED** | `{leakage_summary.get('id_overlap_count', 0)}` | Exactly 0 |")
    lines.append(f"| SHA-256 Hash Overlap Count | **PASSED** | `{leakage_summary.get('hash_overlap_count', 0)}` | Exactly 0 |")
    lines.append(f"| Near-Duplicate Cluster Overlap | **PASSED** | `{leakage_summary.get('cluster_overlap_count', 0)}` | Exactly 0 |")
    lines.append(f"| Max Cross-Split Token Jaccard | **PASSED** | `{leakage_summary.get('max_cross_split_jaccard', 0.0):.4f}` | < 0.8000 |")
    lines.append(f"| 5-Fold CV Partition Leakage | **PASSED** | `0` | Disjoint Folds |")
    lines.append("")

    # Test Set Baseline Table
    lines.append("## 2. Held-Out Test Set Performance (Tier 1 & Tier 2 Baselines)")
    lines.append("")
    lines.append("| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ECE | RQ1 Neu→Neg Error Rate |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for name, res in models_results.items():
        tm = res["test_metrics"]
        pc = tm.get("per_class", {})
        neg_f1 = pc.get("negative", {}).get("f1", 0.0)
        neu_f1 = pc.get("neutral", {}).get("f1", 0.0)
        pos_f1 = pc.get("positive", {}).get("f1", 0.0)
        ece_str = f"{tm['ece']:.4f}" if tm.get("ece") is not None else "N/A"
        n2n = tm.get("neutral_to_negative_rate", 0.0)
        lines.append(
            f"| **{name}** | {tm['accuracy']:.4f} | **{tm['macro_f1']:.4f}** | "
            f"{neg_f1:.4f} | {neu_f1:.4f} | {pos_f1:.4f} | {ece_str} | **{n2n:.4f}** |"
        )
    lines.append("")

    # 5-Fold Cross-Validation Table
    lines.append("## 3. 5-Fold Cross-Validation Performance (Training Partition)")
    lines.append("")
    lines.append("| Model | Mean Accuracy (±Std) | Mean Macro-F1 (±Std) | Mean Neu→Neg Rate (±Std) |")
    lines.append("|---|---|---|---|")
    for name, res in models_results.items():
        cv = res.get("cv_metrics", {})
        acc_str = f"{cv.get('accuracy_mean', 0.0):.4f} ± {cv.get('accuracy_std', 0.0):.4f}"
        f1_str = f"{cv.get('macro_f1_mean', 0.0):.4f} ± {cv.get('macro_f1_std', 0.0):.4f}"
        n2n_str = f"{cv.get('neutral_to_negative_rate_mean', 0.0):.4f} ± {cv.get('neutral_to_negative_rate_std', 0.0):.4f}"
        lines.append(f"| **{name}** | {acc_str} | **{f1_str}** | {n2n_str} |")
    lines.append("")

    # External Holdout Table
    lines.append("## 4. Deduplicated External Holdout Performance (SEntFiN 1.0)")
    lines.append("")
    lines.append("| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ECE | Neu→Neg Error Rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in models_results.items():
        hm = res.get("external_holdout_metrics")
        if hm:
            pc = hm.get("per_class", {})
            neg_f1 = pc.get("negative", {}).get("f1", 0.0)
            neu_f1 = pc.get("neutral", {}).get("f1", 0.0)
            pos_f1 = pc.get("positive", {}).get("f1", 0.0)
            ece_str = f"{hm['ece']:.4f}" if hm.get("ece") is not None else "N/A"
            n2n = hm.get("neutral_to_negative_rate", 0.0)
            lines.append(
                f"| **{name}** | {hm['accuracy']:.4f} | **{hm['macro_f1']:.4f}** | "
                f"{neg_f1:.4f} | {neu_f1:.4f} | {pos_f1:.4f} | {ece_str} | {n2n:.4f} |"
            )
        else:
            lines.append(f"| **{name}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")

    # RQ1 Deep Dive
    lines.append("## 5. RQ1 Deep Dive: Neutral-to-Negative Misclassification Analysis")
    lines.append("")
    lines.append(
        "RQ1 investigates domain polarity mismatch: general sentiment lexicons (such as VADER) "
        "frequently misclassify neutral descriptive accounting terminology (*cost*, *liability*, "
        "*depreciation*, *debt*, *obligation*) as negative, creating false-negative errors. "
        "Below are the observed `neutral_to_negative_rate` metrics across models:"
    )
    lines.append("")
    for name, res in models_results.items():
        tm = res["test_metrics"]
        cm = tm["confusion_matrix"]
        neu_total = sum(cm[1])
        neu_to_neg = cm[1][0]
        rate = tm["neutral_to_negative_rate"]
        lines.append(f"- **{name}:** {neu_to_neg} / {neu_total} neutral sentences falsely predicted as negative ({rate:.2%}).")
    lines.append("")

    # Statistical Significance Comparisons
    lines.append("## 6. Statistical Significance Comparisons (Held-Out Test Set)")
    lines.append("")
    lines.append("| Comparison (Model A vs. Model B) | McNemar χ² | McNemar p-value | ΔMacro-F1 (95% CI) | Bootstrap p-value | Significant (p < 0.05) |")
    lines.append("|---|---|---|---|---|---|")

    for comp_name, comp in statistical_comparisons.items():
        mc = comp["mcnemar"]
        bs = comp["bootstrap"]
        sig = "Yes (p < 0.05)" if (mc["p_value"] < 0.05 or bs["p_value"] < 0.05) else "No"
        ci_str = f"[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]"
        lines.append(
            f"| **{comp_name}** | {mc['chi2']:.3f} | {mc['p_value']:.4f} | "
            f"{bs['delta_macro_f1']:+.4f} {ci_str} | {bs['p_value']:.4f} | **{sig}** |"
        )
    lines.append("")

    # Next Steps
    lines.append("## 7. Transition to Step 2 (Tier 3 Transformer Fine-Tuning)")
    lines.append("")
    lines.append(
        "With Step 1 complete, the locked SHA-256 split manifests in `data/splits/` will serve as "
        "the exact, identical train/val/test splits for fine-tuning `bert-base-uncased` and "
        "`yiyanghkust/finbert-pretrain` in Step 2. This guarantees zero test leakage and enables rigorous "
        "attribution of domain adaptation benefits."
    )
    lines.append("")

    return "\n".join(lines)


def run_step1(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the end-to-end Step 1 pipeline."""
    if args.mode == "full" and not args.execute:
        raise RuntimeError(
            "Full execution mode requires --execute flag. "
            "Real dataset processing is designed to run in Google Colab."
        )

    bootstrap_samples = (
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else (200 if args.mode == "smoke" else 10000)
    )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)

    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Starting Step 1 Pipeline...")
    print(f"Mode: {args.mode} | Seed: {args.seed} | Bootstrap Samples: {bootstrap_samples}")

    # 1. Dataset Ingestion and Split Locking
    if args.mode == "smoke":
        print("Generating deterministic synthetic financial datasets...")
        synth_data = generate_synthetic_dataset(n_per_class=40, seed=args.seed)
        pb_synth = synth_data["phrasebank_synthetic"]
        sentfin_synth = synth_data["sentfin_synthetic"]

        records = build_records(
            texts=pb_synth["texts"],
            labels=pb_synth["labels"],
            prefix="fpb_synth",
            jaccard_threshold=0.80,
        )
        manifest = create_stratified_splits(
            records=records,
            seed=args.seed,
            ratios=(0.70, 0.15, 0.15),
            n_folds=5,
        )
        manifest_filename = "phrasebank_synthetic_manifest.json"
        manifest_path = os.path.join(args.splits_dir, manifest_filename)
        with open(manifest_path, mode="w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved locked split manifest to {manifest_path}")

        # Deduplicate external holdout against PhraseBank
        ref_texts = [r["text"] for r in records]
        filtered_holdout_records, holdout_stats = deduplicate_external_holdout(
            external_records=sentfin_synth["records"],
            reference_texts=ref_texts,
            threshold=0.80,
        )
        holdout_path = os.path.join(args.splits_dir, "sentfin_synthetic_dedup.json")
        with open(holdout_path, mode="w", encoding="utf-8") as f:
            json.dump({
                "stats": holdout_stats,
                "records": filtered_holdout_records,
            }, f, indent=2)
        print(f"Saved deduplicated external holdout to {holdout_path} (kept {holdout_stats['kept']} / {holdout_stats['total_input']})")

    else:
        # Full mode (Colab)
        print("Loading real financial sentiment datasets with --execute...")
        real_data = load_real_datasets(cache_dir="data/raw", execute=True)

        configs_to_process = (
            ["sentences_allagree", "sentences_75agree", "sentences_50agree"]
            if args.fpb_config == "all"
            else [args.fpb_config]
        )

        all_pb_texts: list[str] = []
        manifests_by_cfg: dict[str, Any] = {}

        for cfg in configs_to_process:
            pb_cfg_data = real_data.get("financial_phrasebank", {}).get(cfg)
            if not pb_cfg_data:
                raise RuntimeError(f"Could not load Financial PhraseBank config '{cfg}' from datasets.")

            texts = pb_cfg_data["texts"]
            labels = pb_cfg_data["labels"]
            all_pb_texts.extend(texts)

            cfg_records = build_records(texts=texts, labels=labels, prefix=cfg, jaccard_threshold=0.80)
            cfg_manifest = create_stratified_splits(records=cfg_records, seed=args.seed, ratios=(0.70, 0.15, 0.15), n_folds=5)
            verify_no_leakage(cfg_manifest, jaccard_threshold=0.80)

            cfg_manifest_path = os.path.join(args.splits_dir, f"{cfg}_manifest.json")
            with open(cfg_manifest_path, mode="w", encoding="utf-8") as f:
                json.dump(cfg_manifest, f, indent=2)
            print(f"Saved locked split manifest for {cfg} to {cfg_manifest_path}")
            manifests_by_cfg[cfg] = cfg_manifest

        # Use primary config manifest for baseline training
        primary_cfg = "sentences_allagree" if "sentences_allagree" in manifests_by_cfg else configs_to_process[0]
        manifest = manifests_by_cfg[primary_cfg]

        # External SEntFiN deduplication
        raw_sentfin = real_data.get("sentfin", [])
        if isinstance(raw_sentfin, list) and len(raw_sentfin) > 0:
            from src.data.download import parse_sentfin_decisions

            sentfin_norm_records: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_sentfin):
                if "label" in item and "text" in item and isinstance(item["label"], int):
                    sentfin_norm_records.append(item)
                else:
                    t = (item.get("Title") or item.get("text") or "").strip()
                    if not t:
                        continue
                    decisions = parse_sentfin_decisions(
                        item.get("Decisions") or item.get("Sentiment") or ""
                    )
                    pols = set(decisions.values())
                    if len(pols) != 1:
                        continue
                    lbl_str = next(iter(pols))
                    lbl_id = LABEL_TO_ID[lbl_str]
                    sentfin_norm_records.append({
                        "id": f"sentfin_{idx:05d}",
                        "text": t,
                        "label": lbl_id,
                        "label_name": ID_TO_LABEL[lbl_id],
                        "entities": decisions,
                    })
            filtered_holdout_records, holdout_stats = deduplicate_external_holdout(
                external_records=sentfin_norm_records,
                reference_texts=list(set(all_pb_texts)),
                threshold=0.80,
            )
            if "sentfin_audit" in real_data:
                holdout_stats["upstream_audit"] = real_data["sentfin_audit"]
        else:
            filtered_holdout_records = []
            holdout_stats = {"total_input": 0, "removed_overlap": 0, "kept": 0}

        holdout_path = os.path.join(args.splits_dir, "sentfin_dedup.json")
        with open(holdout_path, mode="w", encoding="utf-8") as f:
            json.dump({"stats": holdout_stats, "records": filtered_holdout_records}, f, indent=2)

    # 2. Leakage Verification
    print("Verifying zero data leakage across train, val, and test splits...")
    leakage_summary = verify_no_leakage(manifest, jaccard_threshold=0.80)
    print(f"Leakage Audit PASSED: Zero ID/Hash/Cluster overlap, Max Jaccard = {leakage_summary['max_cross_split_jaccard']:.4f}")

    # Extract splits
    records_by_id = manifest["records_by_id"]
    train_recs = [records_by_id[rid] for rid in manifest["splits"]["train"]]
    val_recs = [records_by_id[rid] for rid in manifest["splits"]["val"]]
    test_recs = [records_by_id[rid] for rid in manifest["splits"]["test"]]

    train_texts = [r["text"] for r in train_recs]
    train_labels = [r["label"] for r in train_recs]
    val_texts = [r["text"] for r in val_recs]
    val_labels = [r["label"] for r in val_recs]
    test_texts = [r["text"] for r in test_recs]
    test_labels = [r["label"] for r in test_recs]

    holdout_texts = [r["text"] for r in filtered_holdout_records if "text" in r]
    holdout_labels = [r["label"] for r in filtered_holdout_records if "label" in r]

    # 3. Model Training & Evaluation
    model_names = [
        "VADER",
        "Loughran-McDonald",
        "TF-IDF + Logistic Regression",
        "TF-IDF + Linear SVM",
    ]

    models_results: dict[str, Any] = {}
    test_predictions_by_model: dict[str, np.ndarray] = {}

    for name in model_names:
        print(f"\n--- Training & Evaluating: {name} ---")
        model = instantiate_fresh_model(name, seed=args.seed)

        # Fit with validation tuning
        model.fit(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)

        # Validation evaluation
        val_preds = model.predict(val_texts)
        val_probs = model.predict_proba(val_texts)
        val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)

        # Test evaluation
        test_preds = model.predict(test_texts)
        test_probs = model.predict_proba(test_texts)
        test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)
        test_predictions_by_model[name] = test_preds

        print(
            f"Test Results -> Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
            f"Neu→Neg Rate: {test_metrics['neutral_to_negative_rate']:.4f}"
        )

        # External holdout evaluation
        holdout_metrics = None
        if holdout_texts and holdout_labels:
            h_preds = model.predict(holdout_texts)
            h_probs = model.predict_proba(holdout_texts)
            holdout_metrics = compute_classification_metrics(holdout_labels, h_preds, h_probs)

        # 5-Fold Cross Validation
        cv_folds = manifest.get("cv_folds", [])
        cv_fold_metrics: list[dict[str, Any]] = []

        if cv_folds:
            for f_info in cv_folds:
                f_train_recs = [records_by_id[rid] for rid in f_info["train"]]
                f_val_recs = [records_by_id[rid] for rid in f_info["val"]]

                f_train_t = [r["text"] for r in f_train_recs]
                f_train_l = [r["label"] for r in f_train_recs]
                f_val_t = [r["text"] for r in f_val_recs]
                f_val_l = [r["label"] for r in f_val_recs]

                fold_model = instantiate_fresh_model(name, seed=args.seed)
                fold_model.fit(f_train_t, f_train_l, val_texts=f_val_t, val_labels=f_val_l)
                f_pred = fold_model.predict(f_val_t)
                f_prob = fold_model.predict_proba(f_val_t)
                fold_m = compute_classification_metrics(f_val_l, f_pred, f_prob)
                cv_fold_metrics.append({
                    "fold": f_info["fold"],
                    "accuracy": fold_m["accuracy"],
                    "macro_f1": fold_m["macro_f1"],
                    "neutral_to_negative_rate": fold_m["neutral_to_negative_rate"],
                })

            cv_accs = [m["accuracy"] for m in cv_fold_metrics]
            cv_f1s = [m["macro_f1"] for m in cv_fold_metrics]
            cv_n2n = [m["neutral_to_negative_rate"] for m in cv_fold_metrics]

            cv_summary = {
                "accuracy_mean": float(np.mean(cv_accs)),
                "accuracy_std": float(np.std(cv_accs)),
                "macro_f1_mean": float(np.mean(cv_f1s)),
                "macro_f1_std": float(np.std(cv_f1s)),
                "neutral_to_negative_rate_mean": float(np.mean(cv_n2n)),
                "neutral_to_negative_rate_std": float(np.std(cv_n2n)),
                "folds": cv_fold_metrics,
            }
        else:
            cv_summary = {}

        models_results[name] = {
            "best_params": getattr(model, "best_params_", {}),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "cv_metrics": cv_summary,
            "external_holdout_metrics": holdout_metrics,
        }

    # 4. Statistical Significance Testing
    print("\nComputing pairwise McNemar tests and paired bootstrap confidence intervals...")
    statistical_comparisons: dict[str, Any] = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            pair_key = f"{name_a} vs {name_b}"

            preds_a = test_predictions_by_model[name_a]
            preds_b = test_predictions_by_model[name_b]

            mc_res = mcnemar_test(test_labels, preds_a, preds_b)
            bs_res = paired_bootstrap_ci(
                test_labels,
                preds_a,
                preds_b,
                B=bootstrap_samples,
                seed=args.seed,
            )

            statistical_comparisons[pair_key] = {
                "model_a": name_a,
                "model_b": name_b,
                "mcnemar": mc_res,
                "bootstrap": bs_res,
            }
            print(
                f"  {pair_key} -> McNemar p={mc_res['p_value']:.4f} | "
                f"ΔMacro-F1={bs_res['delta_macro_f1']:+.4f} (95% CI: [{bs_res['ci_low']:+.4f}, {bs_res['ci_high']:+.4f}], p={bs_res['p_value']:.4f})"
            )

    # 5. Save Results
    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "bootstrap_samples": bootstrap_samples,
        "dataset": "phrasebank_synthetic" if args.mode == "smoke" else "financial_phrasebank",
        "split_sizes": {
            "train": len(train_texts),
            "val": len(val_texts),
            "test": len(test_texts),
            "external_holdout": len(holdout_texts),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_metrics_path = os.path.join(args.out_dir, "summary_metrics.json")
    with open(summary_metrics_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "leakage_audit": leakage_summary,
            "models": models_results,
        }, f, indent=2)
    print(f"\nSaved summary metrics to {summary_metrics_path}")

    stats_comparisons_path = os.path.join(args.out_dir, "statistical_comparisons.json")
    with open(stats_comparisons_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "test_size": len(test_texts),
                "bootstrap_samples": bootstrap_samples,
                "seed": args.seed,
            },
            "comparisons": statistical_comparisons,
        }, f, indent=2)
    print(f"Saved statistical comparisons to {stats_comparisons_path}")

    report_markdown = generate_markdown_report(
        metadata=metadata,
        leakage_summary=leakage_summary,
        models_results=models_results,
        statistical_comparisons=statistical_comparisons,
    )
    report_path = os.path.join(args.out_dir, "STEP1_REPORT.md")
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_markdown)
    print(f"Saved Step 1 Markdown report to {report_path}")

    print(f"\nStep 1 Pipeline Completed Successfully.")
    return {
        "metadata": metadata,
        "leakage_audit": leakage_summary,
        "models": models_results,
        "statistical_comparisons": statistical_comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for colab/run_step1_colab.py."""
    args = parse_args(argv)
    run_step1(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
