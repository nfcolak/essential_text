"""Data ingestion, synthetic dataset generation, and Colab-gated real dataset loading.

Provides deterministic in-memory synthetic financial datasets for local smoke and unit
testing without network calls, and an --execute-gated loader for full datasets
(Financial PhraseBank, SEntFiN 1.0, Loughran-McDonald lexicon) in Google Colab.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import random
import re
import urllib.request
import zipfile
from typing import Any

from src.data.dedup import sentence_sha256
from src.data.splits import ID_TO_LABEL, LABEL_TO_ID

# Pinned primary-source URLs and SHA-256 checksums (verified via nlp-dataset-audits)
FPB_COMMIT_SHA = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
FPB_ZIP_URL = (
    f"https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
    f"{FPB_COMMIT_SHA}/data/FinancialPhraseBank-v1.0.zip"
)
FPB_ZIP_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"

FPB_CONFIG_TO_FILENAME: dict[str, str] = {
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
}

SENTFIN_COMMIT_SHA = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
SENTFIN_CSV_URL = (
    f"https://raw.githubusercontent.com/pyRis/SEntFiN/"
    f"{SENTFIN_COMMIT_SHA}/SEntFiN.csv"
)
SENTFIN_CSV_SHA256 = "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"

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


def parse_sentfin_decisions(raw_decisions: str) -> dict[str, str]:
    """Parse SEntFiN 1.0 entity-sentiment `Decisions` column into a clean dict.

    Handles both JSON-formatted strings (`{"Nokia": "positive"}`) and Python-literal
    single-quoted dicts including the 6 upstream rows with unescaped apostrophes in
    entity names (e.g. `{'Tata Steel UK': 'negative', 'Moody's': 'neutral'}`).
    """
    if not raw_decisions or not raw_decisions.strip():
        return {}
    val = raw_decisions.strip()
    if val in LABEL_TO_ID:
        return {"_sentence": val}
    try:
        parsed = json.loads(val)
    except Exception:
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pattern = r"\"([^\"]+)\"\s*:\s*\"(positive|neutral|negative)\"|'(.*?)'\s*:\s*'(positive|neutral|negative)'(?=\s*[,}])"
            matches = re.findall(pattern, val)
            parsed = {(m[0] or m[2]).strip(): (m[1] or m[3]).strip() for m in matches if (m[0] or m[2])}
    if not isinstance(parsed, dict):
        return {}
    clean: dict[str, str] = {}
    for k, v in parsed.items():
        v_norm = str(v).strip().lower()
        if str(k).strip() and v_norm in LABEL_TO_ID:
            clean[str(k).strip()] = v_norm
    return clean


def _download_with_sha256(url: str, dest_path: str, expected_sha256: str | None = None) -> str:
    """Download a file from `url` to `dest_path` if not already cached and verify SHA-256."""
    if os.path.exists(dest_path):
        with open(dest_path, "rb") as f:
            existing_sha = hashlib.sha256(f.read()).hexdigest()
        if expected_sha256 is None or existing_sha == expected_sha256:
            return existing_sha

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    actual_sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha}"
        )

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)
    return actual_sha


def load_real_datasets(
    cache_dir: str = "data/raw",
    execute: bool = False,
) -> dict[str, Any]:
    """Load real financial sentiment datasets (Financial PhraseBank v1.0 & SEntFiN 1.0).

    Gated execution: raises RuntimeError if execute is False to ensure real dataset
    downloads and processing only run when explicitly invoked (e.g., in Google Colab).

    Downloads from pinned upstream commits and verifies SHA-256 hashes:
    1. Financial PhraseBank v1.0 (`takala/financial_phrasebank` @ `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`)
       parsed directly from `FinancialPhraseBank-v1.0.zip` (`iso-8859-1`, `@` delimiter) so it
       never fails on `datasets >= 3.0` where legacy dataset loading scripts are disabled.
    2. SEntFiN 1.0 (`pyRis/SEntFiN` @ `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`), parsing
       entity-level `Decisions` JSON/dicts, retaining unambiguous single-polarity headlines
       (9,520 of 10,753 rows; 9,460 unique normalized headlines) and recording full audit counts.
    """
    if not execute:
        raise RuntimeError(
            "Real dataset download requires --execute (run in Google Colab)."
        )

    os.makedirs(cache_dir, exist_ok=True)
    results: dict[str, Any] = {
        "provenance": {
            "financial_phrasebank": {
                "repo": "takala/financial_phrasebank",
                "revision": FPB_COMMIT_SHA,
                "url": FPB_ZIP_URL,
                "expected_sha256": FPB_ZIP_SHA256,
            },
            "sentfin": {
                "repo": "pyRis/SEntFiN",
                "revision": SENTFIN_COMMIT_SHA,
                "url": SENTFIN_CSV_URL,
                "expected_sha256": SENTFIN_CSV_SHA256,
            },
        }
    }

    # 1. Financial PhraseBank v1.0 (direct pinned ZIP download + ISO-8859-1 parsing)
    fpb_zip_path = os.path.join(cache_dir, "FinancialPhraseBank-v1.0.zip")
    fpb_sha = _download_with_sha256(FPB_ZIP_URL, fpb_zip_path, FPB_ZIP_SHA256)
    results["provenance"]["financial_phrasebank"]["actual_sha256"] = fpb_sha

    results["financial_phrasebank"] = {}
    with zipfile.ZipFile(fpb_zip_path, mode="r") as zf:
        for cfg, member_name in FPB_CONFIG_TO_FILENAME.items():
            raw_lines = zf.read(member_name).decode("iso-8859-1").splitlines()
            texts: list[str] = []
            labels: list[int] = []
            for line in raw_lines:
                if "@" not in line:
                    continue
                sentence, label_str = line.rsplit("@", 1)
                sentence = sentence.strip()
                label_norm = label_str.strip().lower()
                if sentence and label_norm in LABEL_TO_ID:
                    texts.append(sentence)
                    labels.append(LABEL_TO_ID[label_norm])
            results["financial_phrasebank"][cfg] = {
                "texts": texts,
                "labels": labels,
            }

    # 2. SEntFiN 1.0 (direct pinned CSV download + entity-decision parsing)
    sentfin_path = os.path.join(cache_dir, "SEntFiN.csv")
    sentfin_sha = _download_with_sha256(SENTFIN_CSV_URL, sentfin_path, SENTFIN_CSV_SHA256)
    results["provenance"]["sentfin"]["actual_sha256"] = sentfin_sha

    sentfin_records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    total_raw_rows = 0
    mixed_multi_entity_rows = 0
    duplicate_internal_rows = 0

    with open(sentfin_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            total_raw_rows += 1
            title = (row.get("Title") or row.get("text") or "").strip()
            if not title:
                continue
            decisions_raw = row.get("Decisions") or row.get("Sentiment") or ""
            decisions = parse_sentfin_decisions(decisions_raw)
            polarities = set(decisions.values())
            if len(polarities) != 1:
                mixed_multi_entity_rows += 1
                continue
            lbl_str = next(iter(polarities))
            lbl_id = LABEL_TO_ID[lbl_str]
            h = sentence_sha256(title)
            if h in seen_hashes:
                duplicate_internal_rows += 1
                continue
            seen_hashes.add(h)
            sentfin_records.append({
                "id": f"sentfin_{idx:05d}",
                "text": title,
                "sha256": h,
                "label": lbl_id,
                "label_name": ID_TO_LABEL[lbl_id],
                "entities": decisions,
            })

    results["sentfin"] = sentfin_records
    results["sentfin_audit"] = {
        "total_raw_rows": total_raw_rows,
        "mixed_multi_entity_excluded": mixed_multi_entity_rows,
        "internal_duplicates_excluded": duplicate_internal_rows,
        "unambiguous_unique_kept": len(sentfin_records),
    }

    # 3. Loughran-McDonald Master Dictionary (embedded in src.baselines.lexicon + optional CSV override)
    lm_path = os.path.join(cache_dir, "Loughran-McDonald_MasterDictionary_1993-2023.csv")
    results["loughran_mcdonald_path"] = lm_path if os.path.exists(lm_path) else "embedded_curated_lm_v2023"

    return results

