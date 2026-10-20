"""Tier 1 lexicon baselines built on the published lexicons, applied with fixed rules.

- `VaderOfficialBaseline`: the official vaderSentiment lexicon (`SentimentIntensityAnalyzer`) with the fixed
  Hutto & Gilbert compound thresholds (+/-0.05). No tuning, no lexicon edits.
- `LMOfficialBaseline`: the official Loughran-McDonald Master Dictionary, plain counting (P > N positive,
  N > P negative, else neutral). No negation rule, no edits.

Both return probabilities on the 3-class simplex (columns `[negative, neutral, positive]`) whose argmax
equals `predict` exactly, because the Step 2 runner scores the argmax of the probabilities.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from typing import Any, Optional, Sequence

import numpy as np

from src.baselines.lexicon import _logits_to_probability_simplex

LM_CSV_DEFAULT = "data/raw/LM_MasterDictionary_1993-2025.csv"
LM_CSV_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"
LM_CSV_URL = (
    "https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
)

VADER_THRESHOLD = 0.05
# Offset that pushes a score sitting exactly on a (inclusive) threshold off the simplex tie.
_BOUNDARY_NUDGE = 1e-6
_LM_THRESHOLD = 1e-9
_WORD_RE = re.compile(r"[A-Za-z]+")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lm_master(
    path: str = LM_CSV_DEFAULT, expected_sha256: Optional[str] = LM_CSV_SHA256
) -> tuple[set[str], set[str]]:
    """Return `(positive_words, negative_words)` (upper-case) of the LM Master Dictionary CSV.

    A word is a member when its `Positive` / `Negative` column is non-zero (the value is the year it was added).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Loughran-McDonald Master Dictionary not found at {path}. Download it from {LM_CSV_URL} "
            "(listed on https://sraf.nd.edu/loughranmcdonald-master-dictionary/)."
        )
    if expected_sha256 is not None:
        actual = _file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}."
            )
    positive: set[str] = set()
    negative: set[str] = set()
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            word = (row.get("Word") or "").strip().upper()
            if not word:
                continue
            if _is_member(row.get("Positive")):
                positive.add(word)
            if _is_member(row.get("Negative")):
                negative.add(word)
    return positive, negative


def _is_member(value: Optional[str]) -> bool:
    try:
        return float(value or 0) != 0
    except ValueError:
        return False


def _rule_probabilities(scores: np.ndarray, t_neg: float, t_pos: float) -> np.ndarray:
    """Simplex probabilities with `<= t_neg` -> negative and `>= t_pos` -> positive as the argmax."""
    scores = np.asarray(scores, dtype=np.float64)
    nudged = np.where(scores >= t_pos, scores + _BOUNDARY_NUDGE, scores)
    nudged = np.where(scores <= t_neg, scores - _BOUNDARY_NUDGE, nudged)
    return _logits_to_probability_simplex(nudged, t_neg, t_pos)


def _vader_version() -> str:
    try:
        from importlib.metadata import version

        return f"vaderSentiment {version('vaderSentiment')}"
    except Exception:
        return "vaderSentiment"


class VaderOfficialBaseline:
    """Official VADER lexicon, positive if compound >= 0.05, negative if <= -0.05, else neutral."""

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold_neg_ = -VADER_THRESHOLD
        self.threshold_pos_ = VADER_THRESHOLD
        self.best_params_: dict[str, Any] = {
            "thresholds": [self.threshold_neg_, self.threshold_pos_],
            "tuned": False,
            "lexicon": _vader_version(),
            "lexicon_size": len(self.analyzer.lexicon),
        }

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(
            [self.analyzer.polarity_scores(str(t))["compound"] for t in texts], dtype=np.float64
        )

    def fit(
        self,
        train_texts: Any = None,
        train_labels: Any = None,
        val_texts: Any = None,
        val_labels: Any = None,
    ) -> "VaderOfficialBaseline":
        """No-op: the thresholds are the published ones and are not tuned."""
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return _rule_probabilities(self.score_texts(texts), self.threshold_neg_, self.threshold_pos_)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)


class LMOfficialBaseline:
    """Official Loughran-McDonald Master Dictionary, positive if P > N, negative if N > P, else neutral."""

    def __init__(
        self,
        lm_path: str = LM_CSV_DEFAULT,
        expected_sha256: Optional[str] = LM_CSV_SHA256,
        words: Optional[tuple[Any, Any]] = None,
    ) -> None:
        if words is not None:
            positive, negative = words
            positive = {str(w).upper() for w in positive}
            negative = {str(w).upper() for w in negative}
        else:
            positive, negative = load_lm_master(lm_path, expected_sha256)
        self.positive_words = positive
        self.negative_words = negative
        self.best_params_: dict[str, Any] = {
            "thresholds": [-_LM_THRESHOLD, _LM_THRESHOLD],
            "tuned": False,
            "lexicon": "Loughran-McDonald Master Dictionary 1993-2025",
            "lexicon_size": {"positive": len(positive), "negative": len(negative)},
        }

    def count_sentence(self, text: str) -> tuple[int, int]:
        tokens = [t.upper() for t in _WORD_RE.findall(str(text))]
        return (
            sum(t in self.positive_words for t in tokens),
            sum(t in self.negative_words for t in tokens),
        )

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        counts = [self.count_sentence(t) for t in texts]
        return np.array([(p - n) / (p + n + 1) for p, n in counts], dtype=np.float64)

    def fit(
        self,
        train_texts: Any = None,
        train_labels: Any = None,
        val_texts: Any = None,
        val_labels: Any = None,
    ) -> "LMOfficialBaseline":
        """No-op: plain counting, nothing to tune."""
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return _rule_probabilities(self.score_texts(texts), -_LM_THRESHOLD, _LM_THRESHOLD)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)
