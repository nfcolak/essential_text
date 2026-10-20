"""Rule-based pre-tagger for the five linguistic phenomena of the Step 3 diagnostic analysis.

The tags are an approximation used (a) as a third rater next to the two annotators and (b) for the larger
full-split analysis where no annotators are available. A sentence can carry several tags.
"""

from __future__ import annotations

import re

PHENOMENA = ("financial_terminology", "negation", "hedging", "numerical_comparison", "context_dependent")

_FLAGS = re.IGNORECASE

# --- financial_terminology -------------------------------------------------------------------------------------
_FIN_TERMS = re.compile(
    r"\b(?:"
    r"costs?|expenses?|liabilit(?:y|ies)|debts?|depreciation|amorti[sz]ation|impairments?|"
    r"write[- ]?downs?|write[- ]?offs?|provisions?|charges?|tax(?:es)?|loss(?:es)?|deficits?|leverage[d]?|"
    r"exposures?|risks?|restructuring|bearish|bears?|bear market|volatility|claims?|"
    r"short (?:positions?|selling|sell)"
    r")\b",
    _FLAGS,
)

# --- negation --------------------------------------------------------------------------------------------------
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|neither|nor|none|cannot|failed to|fails to|fail to|lack(?:s|ed|ing)?)\b"
    r"|n['’]t\b",
    _FLAGS,
)

# --- hedging ---------------------------------------------------------------------------------------------------
_HEDGING = re.compile(
    r"\b(?:"
    r"(?-i:may)|might|could|would|"
    r"expect(?:s|ed|ing)?|anticipat(?:e|es|ed|ing)|estimat(?:e|es|ed|ing)|forecasts?|forecasted|guidance|outlook|"
    r"likely|possible|possibly|potential|potentially|approximately|believ(?:e|es|ed)|"
    r"plans? to|aims? to|targets?|targeted|projected|projects? to|seen as"
    r")\b"
    r"|\b(?:about|around)\s+(?:[$€£]|eur|usd|gbp)?\s*\d",
    _FLAGS,
)

# --- numerical_comparison --------------------------------------------------------------------------------------
_HAS_NUMBER = re.compile(r"\d")
_CHANGE = re.compile(
    r"\b(?:rose|rise[sn]?|rising|fell|fall(?:s|en|ing)?|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
    r"grew|grow(?:s|n|ing)?|declin(?:e|es|ed|ing)|dropp?(?:ed|ing|s)?|jump(?:ed|s|ing)?|up|down|"
    r"(?:higher|lower|more|less|greater|smaller) than|compared (?:with|to)|versus|vs\.?|"
    r"year[- ]on[- ]year|y[- ]o[- ]y|yoy|doubled|halved|tripled)\b"
    r"|\bfrom\b[^.;]*?\d[^.;]*?\bto\b",
    _FLAGS,
)
_RATIO_WORDS = re.compile(r"\b(?:doubled|halved|tripled)\b", _FLAGS)

# --- context_dependent -----------------------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+", _FLAGS)
_DIRECTION = {
    "decrease", "decreased", "decreases", "decreasing", "reduce", "reduced", "reduces", "reducing", "reduction",
    "fell", "fall", "falls", "falling", "fallen", "drop", "dropped", "drops", "dropping", "decline", "declined",
    "declines", "declining", "narrow", "narrowed", "narrows", "narrowing", "lower", "lowered", "lowering",
    "cut", "cuts", "cutting", "increase", "increased", "increases", "increasing", "rise", "rose", "risen",
    "rises", "rising", "grow", "grew", "grown", "growth", "widen", "widened", "widens", "widening", "higher",
    "shrink", "shrank", "shrunk", "improve", "improved", "worsen", "worsened", "eliminate", "eliminated",
}
_NEGATIVE_NOUNS = {
    "loss", "losses", "cost", "costs", "debt", "debts", "deficit", "deficits", "expense", "expenses",
    "liability", "liabilities", "impairment", "impairments", "unemployment", "layoff", "layoffs",
}
_CUT_JOBS = re.compile(r"\b(?:cut|cuts|cutting|eliminat(?:e|es|ed|ing))\b(?:\W+\w+){0,2}?\W+jobs?\b", _FLAGS)
_WINDOW = 4


def _context_dependent(text: str) -> bool:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    directions = [i for i, t in enumerate(tokens) if t in _DIRECTION]
    nouns = [i for i, t in enumerate(tokens) if t in _NEGATIVE_NOUNS]
    for i in directions:
        for j in nouns:
            if i != j and abs(i - j) <= _WINDOW:
                return True
    return bool(_CUT_JOBS.search(text))


def _numerical_comparison(text: str) -> bool:
    if not _CHANGE.search(text):
        return False
    return bool(_HAS_NUMBER.search(text)) or bool(_RATIO_WORDS.search(text))


def tag_sentence(text: str) -> dict[str, int]:
    """Return a 0/1 tag per phenomenon (keys = PHENOMENA)."""
    return {
        "financial_terminology": int(bool(_FIN_TERMS.search(text))),
        "negation": int(bool(_NEGATION.search(text))),
        "hedging": int(bool(_HEDGING.search(text))),
        "numerical_comparison": int(_numerical_comparison(text)),
        "context_dependent": int(_context_dependent(text)),
    }


def tag_many(texts: list[str]) -> list[dict[str, int]]:
    return [tag_sentence(t) for t in texts]
