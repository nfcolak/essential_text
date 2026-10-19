"""Data ingestion, synthetic dataset generation, and Colab-gated real dataset loading.

Provides deterministic in-memory synthetic financial datasets for local smoke and unit
testing without network calls, and an --execute-gated loader for full datasets
(Financial PhraseBank, SEntFiN 1.0, Loughran-McDonald lexicon) in Google Colab.
"""

from __future__ import annotations

import os
import random
from typing import Any

from src.data.splits import ID_TO_LABEL, LABEL_TO_ID

# Seed sentences for generating realistic synthetic financial sentiment data
SYNTHETIC_TEMPLATES: dict[int, list[str]] = {
    0: [  # Negative
        "Operating loss widened to {num1} million EUR as raw material costs spiked unexpectedly.",
        "Sales dropped by {num1}% in the third quarter following severe supply chain disruptions.",
        "The company issued a profit warning and cut its full-year operating profit outlook sharply.",
        "Net profit fell from {num2} million EUR to {num1} million EUR amid mounting competitive pressure.",
        "Restructuring charges and impairment of goodwill severely hurt annual performance.",
        "The firm suffered a net loss due to surging debt financing and inflation.",
        "Deteriorating market demand triggered a sharp decline in quarterly operating margin.",
        "Lower shipment volumes and higher logistics costs resulted in a severe revenue downturn.",
        "Cash flow from operations deteriorated to negative {num1} million EUR in the first half.",
        "The board decided to slash the annual dividend following heavy asset write-downs.",
        "Operating loss widened to {num1} million EUR because raw material costs spiked unexpectedly.",  # Near-duplicate variant
        "Sales dropped by {num1}% in the third period following severe supply chain disruptions.",  # Near-duplicate variant
    ],
    1: [  # Neutral (including accounting/balance sheet terms like cost, liability, depreciation, debt)
        "Depreciation expense for the machinery was reported at {num1}.{num2} million EUR according to accounting standards.",
        "Total debt and lease liabilities remained unchanged during the second quarter.",
        "The company disclosed operating cost figures in the annual balance sheet filing.",
        "Research and development costs represented approximately {num1}% of total annual expenditure.",
        "The board reviewed debt maturity schedules and capital expenditure plans for next year.",
        "Operating expenses and administrative liabilities were in line with quarterly forecasts.",
        "The carrying value of inventory and plant assets was maintained at book value.",
        "Management commented on the liability provisions in accordance with IFRS standards.",
        "Total personnel expenses and pension obligations amounted to {num2} million EUR.",
        "The financial report details interest-bearing debt and amortized loan schedules.",
        "Depreciation expense for the machinery was disclosed at {num1}.{num2} million EUR according to accounting standards.",  # Near-duplicate variant
        "Total debt and lease obligations remained unchanged during the second quarter.",  # Near-duplicate variant
    ],
    2: [  # Positive (including negations like 'did not result in a loss', numerical comparisons)
        "Operating margin rose from {num1}% to {num2}% driven by robust demand across European markets.",
        "Net sales surged by {num1}% year-on-year to {num2} million EUR, exceeding consensus estimates.",
        "The group secured a lucrative multi-year supply agreement with significant profitability.",
        "Cash flow from operations increased substantially to {num2} million EUR due to volume growth.",
        "Cost rationalization did not result in a loss and boosted core operating earnings.",
        "Operating profit expanded by {num1}% as order intake reached an all-time record high.",
        "Strong retail performance lifted quarterly revenues well above initial guidance.",
        "Profit before taxes grew by {num1}% to {num2}.5 million EUR compared with the prior year.",
        "Earnings per share jumped to {num1}.{num2} EUR supported by strong organic growth.",
        "The management raised its full-year guidance following double-digit revenue expansion.",
        "Operating margin rose from {num1}% to {num2}% driven by strong demand across European markets.",  # Near-duplicate variant
        "Net sales surged by {num1}% year-on-year to {num2} million EUR, surpassing consensus estimates.",  # Near-duplicate variant
    ],
}


def generate_synthetic_dataset(
    n_per_class: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate deterministic in-memory synthetic financial datasets for smoke tests.

    Creates realistic financial sentences containing accounting terms (*cost*, *liability*,
    *depreciation*, *debt*), negations (*did not result in a loss*), numerical comparisons
    (*margin rose from 8% to 12%*), and near-duplicates (Jaccard >= 0.80).

    Returns:
        Dictionary containing:
        - "phrasebank_synthetic": dict with "texts", "labels", and "records"
        - "sentfin_synthetic": dict with "texts", "labels", and "records"
    """
    rng = random.Random(seed)

    pb_texts: list[str] = []
    pb_labels: list[int] = []

    for lbl in (0, 1, 2):
        templates = SYNTHETIC_TEMPLATES[lbl]
        for i in range(n_per_class):
            tmpl = templates[i % len(templates)]
            n1 = (i * 3 + 7) % 25 + 1
            n2 = (i * 5 + 13) % 70 + 20
            sentence = tmpl.format(num1=n1, num2=n2)
            pb_texts.append(sentence)
            pb_labels.append(lbl)

    # Sentfin synthetic: include some distinct entities, and 5 intentional overlaps with phrasebank
    # to test cross-dataset deduplication
    sentfin_records: list[dict[str, Any]] = []
    sentfin_texts: list[str] = []
    sentfin_labels: list[int] = []

    entities = ["Nokia", "Konecranes", "Sampo", "Fortum", "Wartsila", "Outokumpu", "UPM", "Stora Enso"]

    # 1. First add 5 overlapping sentences (near-duplicate or exact to phrasebank)
    for i in range(min(5, len(pb_texts))):
        base_text = pb_texts[i]
        lbl = pb_labels[i]
        ent = entities[i % len(entities)]
        overlap_text = f"{ent} reports: {base_text}"  # Near duplicate / overlap
        sentfin_records.append({
            "id": f"sentfin_synth_overlap_{i:03d}",
            "text": base_text,  # exact overlap
            "entity": ent,
            "label": lbl,
            "label_name": ID_TO_LABEL[lbl],
        })
        sentfin_texts.append(base_text)
        sentfin_labels.append(lbl)

    # 2. Add novel external sentences
    novel_templates = {
        0: "{entity} shares plunged {num1}% after cutting dividend forecasts.",
        1: "{entity} announced completion of the scheduled facility inspection.",
        2: "{entity} shares rallied {num1}% following record quarterly deliveries.",
    }

    for lbl in (0, 1, 2):
        for i in range(n_per_class):
            ent = entities[(i + lbl) % len(entities)]
            num = (i * 4 + 5) % 30 + 2
            sent = novel_templates[lbl].format(entity=ent, num1=num)
            rec_id = f"sentfin_synth_novel_{lbl}_{i:03d}"
            sentfin_records.append({
                "id": rec_id,
                "text": sent,
                "entity": ent,
                "label": lbl,
                "label_name": ID_TO_LABEL[lbl],
            })
            sentfin_texts.append(sent)
            sentfin_labels.append(lbl)

    return {
        "phrasebank_synthetic": {
            "texts": pb_texts,
            "labels": pb_labels,
        },
        "sentfin_synthetic": {
            "texts": sentfin_texts,
            "labels": sentfin_labels,
            "records": sentfin_records,
        },
    }


def load_real_datasets(
    cache_dir: str = "data/raw",
    execute: bool = False,
) -> dict[str, Any]:
    """Load real financial sentiment datasets (Financial PhraseBank, SEntFiN 1.0, Loughran-McDonald).

    Gated execution: raises RuntimeError if execute is False to ensure real dataset
    downloads and processing only run in Google Colab per project rules.

    Args:
        cache_dir: Directory where downloaded datasets are cached.
        execute: Boolean gate; must be True to execute downloads.

    Returns:
        Dictionary containing loaded real datasets.
    """
    if not execute:
        raise RuntimeError(
            "Real dataset download requires --execute (run in Google Colab)."
        )

    # Execution path for Google Colab
    os.makedirs(cache_dir, exist_ok=True)
    results: dict[str, Any] = {}

    # 1. Financial PhraseBank via Hugging Face datasets
    try:
        from datasets import load_dataset  # type: ignore

        fpb_configs = ["sentences_allagree", "sentences_75agree", "sentences_50agree"]
        results["financial_phrasebank"] = {}
        for cfg in fpb_configs:
            ds = load_dataset("financial_phrasebank", cfg, split="train", cache_dir=cache_dir)
            results["financial_phrasebank"][cfg] = {
                "texts": [item["sentence"] for item in ds],
                "labels": [item["label"] for item in ds],  # 0: neg, 1: neu, 2: pos in HF
            }
    except Exception as exc:
        results["financial_phrasebank_error"] = str(exc)

    # 2. SEntFiN 1.0
    sentfin_path = os.path.join(cache_dir, "SEntFiN.csv")
    if os.path.exists(sentfin_path):
        import csv

        records = []
        with open(sentfin_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        results["sentfin"] = records
    else:
        results["sentfin"] = {"note": f"SEntFiN.csv not found at {sentfin_path}"}

    # 3. Loughran-McDonald Master Dictionary
    lm_path = os.path.join(cache_dir, "Loughran-McDonald_MasterDictionary_1993-2023.csv")
    if os.path.exists(lm_path):
        results["loughran_mcdonald_path"] = lm_path
    else:
        results["loughran_mcdonald_path"] = None

    return results
