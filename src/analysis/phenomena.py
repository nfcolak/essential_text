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

# Role-sense "charge" (take charge as CEO, in charge of, charge of ...) is not a financial charge.
_CHARGE_ROLE = re.compile(
    r"\b(?:takes?|took|taking)\s+charge\b|\bin\s+charge\b|\bcharge\s+of\b|\badditional\s+charges?\s+of\b",
    _FLAGS,
)


def financial_terms(text: str) -> list[str]:
    """Lowercased financial-terminology terms matched in the text (role-sense "charge" excluded)."""
    masked = _CHARGE_ROLE.sub(" ", text)
    return [re.sub(r"\s+", " ", m.group(0).lower()) for m in _FIN_TERMS.finditer(masked)]


# --- negation --------------------------------------------------------------------------------------------------
# "No. 1 supplier" is an abbreviation of "number", not a negation.
_NO_ABBREV = re.compile(r"\bno\.\s*(?=\d)", _FLAGS)
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
    r"grew|grow(?:s|n|ing)?|declin(?:e|es|ed|ing)|dropp?(?:ed|ing|s)?|jump(?:ed|s|ing)?|"
    r"(?:higher|lower|more|less|greater|smaller) than|compared (?:with|to)|versus|vs\.?|"
    r"year[- ]on[- ]year|y[- ]o[- ]y|yoy|doubled|halved|tripled)\b"
    r"|\bfrom\b[^.;]*?\d[^.;]*?\bto\b",
    _FLAGS,
)
_RATIO_WORDS = re.compile(r"\b(?:doubled|halved|tripled)\b", _FLAGS)
# bare "up"/"down" count only next to a number; phrasal forms are masked first
_UP_DOWN = re.compile(r"\b(?:up|down)\b", _FLAGS)
_PHRASAL_UP = re.compile(r"\bstart[- ]?ups?\b|\b(?:sign|signs|signed|signing|pick|picks|picked|picking)\s+up\b", _FLAGS)
_NUM_TOKEN = re.compile(r"[a-z]+|\d[\d.,]*|%", _FLAGS)
_UP_DOWN_WINDOW = 3

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
_WINDOW = 4

# v2: polarity-loaded targets of both valences; the direction word flips the sentence polarity with the target.
_NEGATIVE_TARGETS = {
    "loss", "losses", "cost", "costs", "debt", "debts", "deficit", "deficits", "expense", "expenses",
    "liability", "liabilities", "impairment", "impairments", "unemployment", "layoff", "layoffs",
    "write-down", "write-downs", "writedown", "writedowns", "provision", "provisions",
}
_POSITIVE_TARGETS = {
    "profit", "profits", "earnings", "income", "sales", "revenue", "revenues", "turnover", "margin", "margins",
    "ebit", "ebitda", "result", "results", "orders", "demand", "dividend", "dividends", "ebt",
}
_TARGET_PHRASES = (("order", "intake"), ("market", "share"))
_JOB_NOUNS = r"(?:jobs?|staff|employees|people|positions|workforce|personnel)"
_CUT_JOBS = re.compile(
    r"\b(?:cut|cuts|cutting|eliminat(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|shed|sheds|shedding|"
    r"lay(?:s|ing)? off|laid off)\b(?:\W+\w+){0,2}?\W+" + _JOB_NOUNS + r"\b",
    _FLAGS,
)
_PEOPLE_LAID_OFF = re.compile(r"\b\d+\s+(?:people|employees|workers|staff)\b[^.;]*?\b(?:laid off|lay off)\b", _FLAGS)
_TOKEN_V2 = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", _FLAGS)


def _context_dependent_v1(text: str) -> bool:
    """Narrow v1 rule (direction word within 4 tokens of a negative-valence noun), kept for the record."""
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    directions = [i for i, t in enumerate(tokens) if t in _DIRECTION]
    nouns = [i for i, t in enumerate(tokens) if t in _NEGATIVE_NOUNS]
    for i in directions:
        for j in nouns:
            if i != j and abs(i - j) <= _WINDOW:
                return True
    return bool(re.search(r"\b(?:cut|cuts|cutting|eliminat(?:e|es|ed|ing))\b(?:\W+\w+){0,2}?\W+jobs?\b", text, _FLAGS))


def context_dependent_v1(text: str) -> int:
    return int(_context_dependent_v1(text))


def _context_dependent(text: str) -> bool:
    tokens = [t.lower() for t in _TOKEN_V2.findall(text)]
    targets = _NEGATIVE_TARGETS | _POSITIVE_TARGETS
    target_idx = [i for i, t in enumerate(tokens) if t in targets]
    for k in range(len(tokens) - 1):
        if (tokens[k], tokens[k + 1]) in _TARGET_PHRASES:
            target_idx.append(k)
            target_idx.append(k + 1)
    # hyphenated forms such as "write-down" are single tokens; also split "cost-cutting" style tokens
    flat = []
    for i, t in enumerate(tokens):
        if "-" in t:
            flat.extend((i, part) for part in t.split("-"))
        else:
            flat.append((i, t))
    directions = [i for i, t in flat if t in _DIRECTION]
    for i in directions:
        for j in target_idx:
            if i != j and abs(i - j) <= _WINDOW:
                return True
    return bool(_CUT_JOBS.search(text) or _PEOPLE_LAID_OFF.search(text))


def _updown_with_number(text: str) -> bool:
    masked = _PHRASAL_UP.sub(" ", text)
    if not _UP_DOWN.search(masked):
        return False
    tokens = [t.lower() for t in _NUM_TOKEN.findall(masked)]
    numeric = [i for i, t in enumerate(tokens) if t[0].isdigit() or t == "%"]
    for i, t in enumerate(tokens):
        if t in ("up", "down") and any(abs(i - j) <= _UP_DOWN_WINDOW for j in numeric):
            return True
    return False


def _numerical_comparison(text: str) -> bool:
    if _CHANGE.search(text) and (_HAS_NUMBER.search(text) or _RATIO_WORDS.search(text)):
        return True
    return _updown_with_number(text)


def tag_sentence(text: str) -> dict[str, int]:
    """Return a 0/1 tag per phenomenon (keys = PHENOMENA)."""
    return {
        "financial_terminology": int(bool(financial_terms(text))),
        "negation": int(bool(_NEGATION.search(_NO_ABBREV.sub(" ", text)))),
        "hedging": int(bool(_HEDGING.search(text))),
        "numerical_comparison": int(_numerical_comparison(text)),
        "context_dependent": int(_context_dependent(text)),
    }


def tag_many(texts: list[str]) -> list[dict[str, int]]:
    return [tag_sentence(t) for t in texts]
