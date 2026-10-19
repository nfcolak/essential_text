"""Tier 1 Lexicon Baselines: VADER and Loughran-McDonald.

Implements financial sentiment lexicon scoring with:
- 3-token window negation handling (e.g., 'did not achieve profit', 'did not suffer losses')
- Domain-specific neutral accounting term handling (cost, liability, depreciation, debt)
- Validation threshold tuning to maximize Macro-F1
- Calibrated 3-class predict and predict_proba on the probability simplex
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence
import numpy as np

LABEL_TO_ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# Domain-specific neutral accounting terms that must NOT be scored as negative
# in financial reporting (Loughran & McDonald 2011).
NEUTRAL_ACCOUNTING_TERMS: set[str] = {
    "cost", "costs",
    "liability", "liabilities",
    "depreciation", "depreciations",
    "debt", "debts",
    "expense", "expenses",
    "amortization", "amortizations",
    "capex", "capital expenditure", "capital expenditures",
    "overhead", "overheads",
    "payable", "payables",
    "obligation", "obligations",
    "interest", "balance",
}

# Negation words for 3-token window lookback
NEGATION_TOKENS: set[str] = {
    "not", "no", "never", "without", "neither", "nor", "none",
    "n't", "hardly", "barely", "scarcely", "cannot", "non",
}

# Punctuation marks that break negation scope
CLAUSE_BREAKERS: set[str] = {".", ";", ",", "!", "?", ":", "-", "--", "(", ")"}

# Curated core Loughran-McDonald financial positive words
LM_FIN_POS: set[str] = {
    "able", "abundance", "abundant", "acclaimed", "accomplish", "accomplished", "accomplishes",
    "accomplishing", "accomplishment", "accomplishments", "achieve", "achieved", "achieves",
    "achieving", "achievement", "achievements", "adequately", "advancement", "advancements",
    "advances", "advancing", "advantage", "advantaged", "advantageous", "advantageously",
    "advantages", "alliance", "alliances", "assure", "assured", "assures", "assuring",
    "attain", "attained", "attaining", "attainment", "attainments", "attains", "attractive",
    "attractiveness", "beautiful", "benefactor", "benefactors", "beneficial", "beneficially",
    "benefit", "benefited", "benefiting", "benefits", "best", "better", "bolster", "bolstered",
    "bolstering", "bolsters", "boom", "booming", "boost", "boosted", "boosting", "boosts",
    "breakthrough", "breakthroughs", "bright", "brilliant", "collaborate", "collaborated",
    "collaborates", "collaborating", "collaboration", "collaborations", "collaborative",
    "collaborator", "collaborators", "compliment", "complimentary", "complimented",
    "complimenting", "compliments", "conclusive", "conclusively", "conducive", "confident",
    "constructive", "constructively", "courteous", "creative", "creatively", "creativeness",
    "creativity", "delight", "delighted", "delightful", "delightfully", "delighting", "delights",
    "dependability", "dependable", "desirable", "desired", "despite", "destined", "diligent",
    "diligently", "distinction", "distinctions", "distinctive", "distinctively", "distinctiveness",
    "distinguished", "durability", "durable", "dynamic", "eager", "eagerly", "eagerness",
    "effective", "effectively", "effectiveness", "efficiencies", "efficiency", "efficient",
    "efficiently", "empower", "empowered", "empowering", "empowers", "enable", "enabled",
    "enables", "enabling", "enhance", "enhanced", "enhancement", "enhancements", "enhances",
    "enhancing", "enjoy", "enjoyable", "enjoyably", "enjoyed", "enjoying", "enjoyment", "enjoys",
    "enthusiasm", "enthusiastic", "enthusiastically", "excel", "excelled", "excelling", "excels",
    "excellence", "excellent", "excellently", "exceptional", "exceptionally", "exceed",
    "exceeded", "exceeding", "exceeds", "excited", "excitement", "exciting", "exclusive",
    "exclusively", "exclusiveness", "exclusives", "exclusivity", "exemplary", "fantastic",
    "favorable", "favorably", "favored", "favoring", "favorite", "favorites", "flawless",
    "flawlessly", "fortune", "fortunate", "fortunately", "foster", "fostered", "fostering",
    "fosters", "fruitful", "fruitfully", "fruitfulness", "gain", "gained", "gaining", "gains",
    "generate", "generated", "generates", "generating", "glad", "good", "great", "greater",
    "greatest", "greatly", "greatness", "grow", "growing", "grown", "grows", "growth",
    "highest", "honor", "honorable", "honored", "honoring", "honors", "ideal", "ideally",
    "impress", "impressed", "impresses", "impressing", "impressive", "impressively", "improve",
    "improved", "improvement", "improvements", "improves", "improving", "incredible",
    "incredibly", "influential", "informative", "ingenuity", "innovate", "innovated",
    "innovates", "innovating", "innovation", "innovations", "innovative", "innovativeness",
    "innovator", "innovators", "insightful", "inspiration", "inspirational", "inspire",
    "inspired", "inspires", "inspiring", "integrity", "invent", "invented", "inventing",
    "invention", "inventions", "inventive", "inventiveness", "inventor", "inventors",
    "leadership", "leading", "lucrative", "lucratively", "merit", "merited", "meriting",
    "meritorious", "merits", "milestone", "milestones", "obtain", "obtained", "obtaining",
    "obtains", "optimism", "optimistic", "optimistically", "outperform", "outperformed",
    "outperforming", "outperforms", "outstrip", "outstripped", "outstripping", "outstrips",
    "paramount", "partnership", "partnerships", "patiently", "perfect", "perfected",
    "perfecting", "perfection", "perfects", "pleasant", "pleasantly", "pleased", "pleasure",
    "plentiful", "popular", "popularity", "positive", "positively", "positives", "preeminent",
    "premier", "premiere", "prestige", "prestigious", "proactive", "proactively", "proficiency",
    "proficient", "proficiently", "profit", "profitability", "profitable", "profitably",
    "profited", "profiting", "profits", "progress", "progressed", "progresses", "progressing",
    "progression", "progressive", "progressively", "prosper", "prospered", "prospering",
    "prosperity", "prosperous", "prospers", "rebound", "rebounded", "rebounding", "rebounds",
    "record", "records", "recover", "recovered", "recovering", "recovers", "recovery",
    "resolve", "resolved", "resolves", "resolving", "reward", "rewarded", "rewarding",
    "rewards", "rise", "risen", "rises", "rising", "robust", "robustly", "robustness",
    "satisfaction", "satisfactorily", "satisfactory", "satisfied", "satisfies", "satisfy",
    "satisfying", "smooth", "smoother", "smoothest", "smoothly", "soar", "soared", "soaring",
    "soars", "solid", "solidity", "solidly", "solve", "solved", "solves", "solving", "spectacular",
    "spectacularly", "stability", "stabilization", "stabilize", "stabilized", "stabilizes",
    "stabilizing", "stable", "strength", "strengthen", "strengthened", "strengthening",
    "strengthens", "strengths", "strong", "stronger", "strongest", "strongly", "succeed",
    "succeeded", "succeeding", "succeeds", "success", "successes", "successful", "successfully",
    "superior", "superiority", "surpass", "surpassed", "surpasses", "surpassing", "sustain",
    "sustainable", "sustained", "sustaining", "sustains", "synergy", "tremendous", "tremendously",
    "triumph", "triumphant", "triumphantly", "unmatched", "unparalleled", "unsurpassed",
    "upbeat", "upgrade", "upgraded", "upgrades", "upgrading", "upside", "upturn", "upturns",
    "upward", "upwards", "valuable", "versatile", "versatility", "vibrant", "victorious",
    "victory", "vigor", "vigorous", "vigorously", "vital", "vitality", "warm", "win", "winner",
    "winners", "winning", "wins", "worthwhile", "worthy", "yield", "yielding", "yields"
}

# Curated core Loughran-McDonald financial negative words
LM_FIN_NEG: set[str] = {
    "abandon", "abandoned", "abandoning", "abandonment", "abandons", "adversity", "adversities",
    "adverse", "adversely", "against", "aggravate", "aggravated", "aggravates", "aggravating",
    "aggravation", "alerted", "allegation", "allegations", "allege", "alleged", "allegedly",
    "alleges", "alleging", "annul", "annulled", "annulling", "annulment", "annuls", "anomalous",
    "anomalously", "anomaly", "anxieties", "anxiety", "anxious", "anxiously", "arrears",
    "bad", "badly", "bailout", "bankrupt", "bankruptcies", "bankruptcy", "bankrupted",
    "bankrupting", "bankrupts", "banner", "barrier", "barriers", "bearish", "blockage",
    "blockages", "bottleneck", "bottlenecks", "boycott", "boycotted", "boycotting", "boycotts",
    "breach", "breached", "breaches", "breaching", "breakdown", "breakdowns", "cancel",
    "canceled", "canceling", "cancellation", "cancellations", "cancels", "catastrophe",
    "catastrophic", "caution", "cautionary", "cautioned", "cautioning", "cautions", "cease",
    "ceased", "ceases", "ceasing", "censure", "censured", "censures", "censuring", "closure",
    "closures", "collapse", "collapsed", "collapses", "collapsing", "collision", "collisions",
    "complaint", "complaints", "complicate", "complicated", "complicates", "complicating",
    "complication", "complications", "compulsion", "compulsory", "concede", "conceded",
    "concedes", "conceding", "concern", "concerned", "concerning", "concerns", "condemn",
    "condemnation", "condemnations", "condemned", "condemning", "condemns", "conflict",
    "conflicted", "conflicting", "conflicts", "confront", "confrontation", "confrontational",
    "confrontations", "confronted", "confronting", "confronts", "confuse", "confused",
    "confuses", "confusing", "confusion", "constrain", "constrained", "constraining",
    "constrains", "constraint", "constraints", "contradict", "contradicted", "contradicting",
    "contradiction", "contradictions", "contradictory", "contradicts", "corrective", "corrupt",
    "corrupted", "corrupting", "corruption", "corrupts", "curtail", "curtailed", "curtailing",
    "curtailment", "curtailments", "curtails", "damage", "damaged", "damages", "damaging",
    "danger", "dangerous", "dangerously", "dangers", "deadlock", "deadlocked", "deadlocks",
    "decline", "declined", "declines", "declining", "decrease", "decreased", "decreases",
    "decreasing", "default", "defaulted", "defaulting", "defaults", "defeat", "defeated",
    "defeats", "defect", "defective", "defects", "deficiency", "deficient", "deficit",
    "deficits", "delinquency", "delinquent", "delinquencies", "demise", "demolish",
    "demolished", "demolishing", "demolition", "demolitions", "depress", "depressed",
    "depresses", "depressing", "depression", "derail", "derailed", "derailing", "derails",
    "destabilize", "destabilized", "destabilizing", "destabilization", "destroy", "destroyed",
    "destroying", "destroys", "destruction", "destructive", "destructively", "deteriorate",
    "deteriorated", "deteriorates", "deteriorating", "deterioration", "deteriorations",
    "detriment", "detrimental", "detrimentally", "detriments", "devalue", "devalued",
    "devalues", "devaluing", "devaluation", "devaluations", "difficult", "difficulties",
    "difficulty", "diminish", "diminished", "diminishes", "diminishing", "disadvantage",
    "disadvantaged", "disadvantages", "disadvantageous", "disagree", "disagreeable",
    "disagreed", "disagreeing", "disagreement", "disagreements", "disagrees", "disappoint",
    "disappointed", "disappointing", "disappointingly", "disappointment", "disappointments",
    "disappoints", "disaster", "disasters", "disastrous", "disastrously", "disclose",
    "disconnect", "disconnected", "discontinuance", "discontinue", "discontinued",
    "discontinues", "discontinuing", "discourage", "discouraged", "discourages",
    "discouraging", "discredit", "discredited", "discrediting", "discredits", "discrepancy",
    "discrepancies", "dismiss", "dismissal", "dismissals", "dismissed", "dismisses",
    "dismissing", "disorder", "disorderly", "disorders", "dispute", "disputed", "disputes",
    "disputing", "disruption", "disruptions", "disruptive", "disrupt", "disrupted", "disrupts",
    "downfall", "downgrade", "downgraded", "downgrades", "downgrading", "downsizing",
    "downturn", "downturns", "downward", "downwards", "drag", "dragged", "dragging",
    "drain", "drained", "draining", "drop", "dropped", "dropping", "drops", "drought",
    "erode", "eroded", "erodes", "eroding", "erosion", "error", "errors", "erroneous",
    "fail", "failed", "failing", "fails", "failure", "failures", "fall", "fallen", "falling",
    "falls", "fallout", "fatality", "fatalities", "fault", "faulty", "fear", "feared",
    "fearing", "fears", "fines", "fired", "flaw", "flawed", "flaws", "forfeit", "forfeited",
    "forfeiture", "fraud", "frauds", "fraudulent", "fraudulence", "freeze", "freezing",
    "froze", "frozen", "grievance", "grievances", "halt", "halted", "halting", "halts",
    "hazard", "hazardous", "hazards", "hinder", "hindered", "hindering", "hindrance",
    "hindrances", "hinders", "hostile", "hostility", "hurt", "hurting", "hurts", "idle",
    "idled", "idling", "impair", "impaired", "impairing", "impairment", "impairments",
    "impairs", "impasse", "impede", "impeded", "impedes", "impeding", "impediment",
    "inability", "inadequate", "inadequately", "inappropriate", "incident", "incidents",
    "incur", "incurred", "incurring", "incurs", "indict", "indicted", "indictment",
    "indictments", "ineffective", "ineffectively", "inefficiency", "inefficient",
    "inefficiently", "infringe", "infringed", "infringement", "infringements", "infringes",
    "injunction", "injunctions", "injure", "injured", "injuries", "injury", "insolvency",
    "insolvent", "instability", "interrupt", "interrupted", "interruption", "interruptions",
    "investigate", "investigated", "investigation", "investigations", "jeopardize",
    "jeopardized", "jeopardizes", "jeopardizing", "jeopardy", "lapse", "lapsed", "lapses",
    "lawsuit", "lawsuits", "layoff", "layoffs", "liquidate", "liquidated", "liquidating",
    "liquidation", "liquidations", "liquidates", "litigate", "litigated", "litigates",
    "litigating", "litigation", "litigations", "lose", "loses", "losing", "loss", "losses",
    "lost", "lower", "lowered", "lowering", "lowers", "lowest", "malfunction", "malfunctions",
    "misconduct", "misleading", "mismanagement", "misrepresent", "misrepresented",
    "misrepresentation", "negative", "negatively", "neglect", "neglected", "negligence",
    "negligent", "obstacle", "obstacles", "offence", "offense", "offenses", "paucity",
    "penalties", "penalty", "peril", "perils", "pessimism", "pessimistic", "petition",
    "plaintiff", "plaintiffs", "plea", "pleaded", "pleading", "plummet", "plummeted",
    "plummeting", "plummets", "plunge", "plunged", "plunges", "plunging", "poor", "poorer",
    "poorest", "poorly", "postpone", "postponed", "postponement", "postponements", "postpones",
    "problem", "problematic", "problems", "prosecute", "prosecuted", "prosecuting",
    "prosecution", "prosecutions", "protest", "protested", "protests", "punish", "punished",
    "punishment", "punitive", "questionable", "recession", "recessions", "recessionary",
    "redundancy", "redundancies", "redundant", "refuse", "refused", "refuses", "refusing",
    "reject", "rejected", "rejecting", "rejection", "rejections", "rejects", "relinquish",
    "relinquished", "restructure", "restructured", "restructures", "restructuring",
    "retaliation", "retaliatory", "retract", "retracted", "retraction", "revoke", "revoked",
    "revoking", "risk", "risks", "risky", "sanction", "sanctioned", "sanctions", "scandal",
    "scandals", "scrutiny", "settlement", "settlements", "sever", "severe", "severely",
    "severity", "shortage", "shortages", "shortfall", "shortfalls", "shrink", "shrinking",
    "shrinks", "shrunk", "slash", "slashed", "slashes", "slashing", "slowdown", "slowdowns",
    "slump", "slumped", "slumping", "slumps", "stagnant", "stagnation", "stagnate",
    "stagnated", "strain", "strained", "strains", "stress", "stressed", "stresses",
    "strike", "strikes", "striking", "stringent", "struggle", "struggled", "struggles",
    "struggling", "subpoena", "subpoenas", "sue", "sued", "sues", "suffer", "suffered",
    "suffering", "suffers", "suing", "suspend", "suspended", "suspending", "suspends",
    "suspension", "suspensions", "tarnish", "tarnished", "terminate", "terminated",
    "terminates", "terminating", "termination", "terminations", "threat", "threaten",
    "threatened", "threatening", "threatens", "threats", "turmoil", "uncertain", "uncertainties",
    "uncertainty", "underperform", "underperformed", "underperforming", "underperforms",
    "unfavorable", "unfavorably", "unfortunate", "unfortunately", "unlawful", "unprofitable",
    "unrest", "unstable", "unsuccessful", "unsuccessfully", "unwelcome", "violate",
    "violated", "violates", "violating", "violation", "violations", "volatility", "volatile",
    "vulnerability", "vulnerable", "warn", "warned", "warning", "warnings", "warns",
    "waste", "wasted", "wasting", "weak", "weaken", "weakened", "weakening", "weakens",
    "weaker", "weakest", "weakness", "weaknesses", "worsen", "worsened", "worsening",
    "worsens", "worst", "worthless", "wreck", "wrecked", "writedown", "writedowns", "writeoff", "writeoffs"
}

# Purge neutral accounting terms from positive and negative sets
LM_FIN_POS = LM_FIN_POS - NEUTRAL_ACCOUNTING_TERMS
LM_FIN_NEG = LM_FIN_NEG - NEUTRAL_ACCOUNTING_TERMS

# Embedded deterministic VADER core lexicon
VADER_CORE_LEXICON: dict[str, float] = {
    # Strong positive
    "great": 3.1, "excellent": 2.7, "superb": 2.8, "outstanding": 2.6,
    "fantastic": 2.6, "exceptional": 2.5, "triumph": 2.5, "brilliant": 2.4,
    "success": 2.3, "successful": 2.4, "profit": 2.0, "profitable": 2.2,
    "growth": 1.8, "grow": 1.6, "grows": 1.6, "grown": 1.6, "growing": 1.7,
    "gain": 1.7, "gains": 1.7, "gained": 1.7, "gaining": 1.7,
    "improve": 1.7, "improved": 1.9, "improves": 1.8, "improving": 1.8, "improvement": 1.8,
    "increase": 1.3, "increased": 1.4, "increases": 1.3, "increasing": 1.4,
    "exceed": 1.8, "exceeded": 1.9, "exceeds": 1.8, "exceeding": 1.8,
    "outperform": 2.1, "outperformed": 2.1, "outperforming": 2.1,
    "boost": 1.7, "boosted": 1.7, "boosting": 1.7,
    "soar": 2.1, "soared": 2.2, "soaring": 2.2,
    "rise": 1.4, "risen": 1.5, "rising": 1.5, "rises": 1.4,
    "strong": 1.8, "stronger": 1.9, "strongest": 2.0, "robust": 1.9,
    "solid": 1.5, "record": 1.6, "rebound": 1.7, "rebounded": 1.7,
    "recover": 1.5, "recovered": 1.6, "recovery": 1.6,
    "good": 1.9, "positive": 1.8, "positively": 1.8, "favorable": 1.8, "favorably": 1.8,
    "achieve": 1.7, "achieved": 1.8, "achievement": 1.9,
    "win": 2.0, "won": 2.0, "winning": 2.0, "reward": 1.7, "lucrative": 2.1,
    "highest": 1.7, "higher": 1.3, "high": 1.2,
    "up": 1.0, "upward": 1.2, "upside": 1.5,

    # Strong negative
    "terrible": -3.0, "horrible": -2.8, "disaster": -2.9, "bankruptcy": -2.9,
    "bankrupt": -2.8, "crisis": -2.5, "catastrophe": -2.8,
    "fail": -2.5, "failed": -2.4, "fails": -2.4, "failing": -2.5, "failure": -2.6,
    "loss": -2.3, "losses": -2.4, "lost": -2.0, "losing": -2.1,
    "decline": -1.8, "declined": -1.8, "declines": -1.7, "declining": -1.8,
    "drop": -1.5, "dropped": -1.6, "drops": -1.5, "dropping": -1.6,
    "decrease": -1.3, "decreased": -1.4, "decreases": -1.3, "decreasing": -1.4,
    "fall": -1.4, "fallen": -1.5, "falls": -1.4, "falling": -1.5,
    "plunge": -2.2, "plunged": -2.3, "plunging": -2.3,
    "plummet": -2.5, "plummeted": -2.6, "plummeting": -2.6,
    "slump": -2.0, "slumped": -2.1, "slumping": -2.1,
    "downfall": -2.2, "downturn": -1.9, "downward": -1.4,
    "shrink": -1.6, "shrunk": -1.6, "shrinking": -1.7,
    "slash": -1.9, "slashed": -2.0, "slashing": -2.0,
    "cut": -1.2, "cuts": -1.3, "cutting": -1.3,
    "weak": -1.8, "weaker": -1.9, "weakest": -2.0, "weakness": -1.8,
    "poor": -2.0, "poorer": -2.1, "poorest": -2.2, "poorly": -2.0,
    "bad": -2.5, "badly": -2.4, "worse": -2.2, "worst": -2.6, "worsen": -2.2, "worsened": -2.3,
    "suffer": -2.1, "suffered": -2.2, "suffering": -2.2, "suffers": -2.1,
    "damage": -2.0, "damaged": -2.1, "damages": -2.0, "damaging": -2.1,
    "harm": -1.9, "harmed": -2.0, "harmful": -2.0,
    "deficit": -1.8, "deficits": -1.8, "negative": -1.6, "negatively": -1.7,
    "unfavorable": -1.7, "unfavorably": -1.7, "adversity": -1.8, "adverse": -1.8, "adversely": -1.8,
    "threat": -1.7, "threaten": -1.8, "threatened": -1.8,
    "collapse": -2.4, "collapsed": -2.4, "collapsing": -2.4,
    "layoff": -2.0, "layoffs": -2.1, "deteriorate": -2.0, "deteriorated": -2.1,
    "down": -1.0, "lower": -1.2, "lowest": -1.6,
}

# Neutralize accounting terms in VADER lexicon
for _term in NEUTRAL_ACCOUNTING_TERMS:
    VADER_CORE_LEXICON[_term] = 0.0

VADER_BOOSTER_DICT: dict[str, float] = {
    "absolutely": 0.293, "amazingly": 0.293, "awfully": 0.293, "completely": 0.293,
    "considerably": 0.293, "decidedly": 0.293, "deeply": 0.293, "enormously": 0.293,
    "entirely": 0.293, "especially": 0.293, "exceptionally": 0.293, "extremely": 0.391,
    "greatly": 0.293, "highly": 0.293, "hugely": 0.293, "incredibly": 0.293,
    "intensely": 0.293, "majorly": 0.293, "more": 0.1, "most": 0.2,
    "particularly": 0.293, "purely": 0.293, "quite": 0.293, "really": 0.293,
    "remarkably": 0.293, "so": 0.293, "substantially": 0.293, "thoroughly": 0.293,
    "totally": 0.293, "tremendously": 0.293, "utterly": 0.293, "very": 0.293,
    "almost": -0.1, "barely": -0.293, "hardly": -0.293, "just": -0.1,
    "less": -0.1, "little": -0.1, "marginally": -0.2, "partly": -0.1,
    "scarcely": -0.293, "slightly": -0.2, "somewhat": -0.1,
}


def _tokenize_text(text: str) -> list[str]:
    """Tokenize sentence preserving words and contractions."""
    return re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b|[.,;:!?]", text)


def _is_negation_token(token: str) -> bool:
    t = token.lower()
    return t in NEGATION_TOKENS or t.endswith("n't")


def _logits_to_probability_simplex(
    scores: np.ndarray,
    threshold_neg: float,
    threshold_pos: float,
) -> np.ndarray:
    """Map continuous polarity scores into a valid 3-class probability simplex.

    Output shape (N, 3), rows sum to 1.0, columns ordered [negative, neutral, positive].
    Uses temperature-scaled piecewise linear logits centered on threshold boundaries.
    """
    scores = np.asarray(scores, dtype=np.float64)
    width = max(1e-4, threshold_pos - threshold_neg)
    center = (threshold_pos + threshold_neg) / 2.0
    half_width = width / 2.0
    scale = 3.0 / half_width

    z_neg = scale * (threshold_neg - scores)
    z_pos = scale * (scores - threshold_pos)
    z_neu = scale * (half_width - np.abs(scores - center))

    logits = np.column_stack([z_neg, z_neu, z_pos])
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Clean numeric precision to guarantee strict simplex
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs


def _compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Macro-F1 across classes 0, 1, 2."""
    f1s: list[float] = []
    for c in (0, 1, 2):
        tp = int(np.count_nonzero((y_true == c) & (y_pred == c)))
        fp = int(np.count_nonzero((y_true != c) & (y_pred == c)))
        fn = int(np.count_nonzero((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


class LoughranMcDonaldBaseline:
    """Tier 1 Loughran-McDonald Financial Sentiment Lexicon Baseline.

    Computes net polarity score: (N_pos - N_neg) / (token_count + 1).
    Features:
    - 3-token window negation handling (inverts polarity)
    - Domain-specific neutral accounting term handling (cost, liability, depreciation, debt)
    - Threshold tuning on validation Macro-F1
    - 3-class predict and predict_proba on probability simplex
    """

    def __init__(
        self,
        pos_words: set[str] | None = None,
        neg_words: set[str] | None = None,
        default_thresholds: tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        self.pos_words: set[str] = set(pos_words or LM_FIN_POS) - NEUTRAL_ACCOUNTING_TERMS
        self.neg_words: set[str] = set(neg_words or LM_FIN_NEG) - NEUTRAL_ACCOUNTING_TERMS
        self.threshold_neg_: float = float(default_thresholds[0])
        self.threshold_pos_: float = float(default_thresholds[1])
        self.best_params_: dict[str, Any] = {
            "threshold_neg": self.threshold_neg_,
            "threshold_pos": self.threshold_pos_,
            "val_macro_f1": None,
        }

    def score_sentence(self, text: str) -> float:
        """Compute net polarity score for a single sentence."""
        raw_tokens = _tokenize_text(text)
        if not raw_tokens:
            return 0.0

        n_pos = 0.0
        n_neg = 0.0
        content_word_count = 0

        # Maintain word tokens and their positions
        word_tokens: list[str] = []
        for tok in raw_tokens:
            if tok in CLAUSE_BREAKERS:
                # Reset word context on clause boundaries
                word_tokens.append("<BREAK>")
            else:
                word_tokens.append(tok.lower())

        for i, word in enumerate(word_tokens):
            if word == "<BREAK>":
                continue
            content_word_count += 1

            if word in NEUTRAL_ACCOUNTING_TERMS:
                # Strictly neutral; does not contribute to positive or negative counts
                continue

            # Lookback up to 3 tokens for negation (stopping at clause breaks)
            window_start = max(0, i - 3)
            prior_tokens = word_tokens[window_start:i]
            if "<BREAK>" in prior_tokens:
                prior_tokens = prior_tokens[prior_tokens.index("<BREAK>") + 1 :]
            is_negated = any(_is_negation_token(t) for t in prior_tokens)

            is_pos = word in self.pos_words
            is_neg = word in self.neg_words

            if is_pos and is_neg:
                # In rare overlap, treat as neutral
                continue
            elif is_pos:
                if is_negated:
                    n_neg += 1.0  # Negated positive acts as negative
                else:
                    n_pos += 1.0
            elif is_neg:
                if is_negated:
                    n_pos += 1.0  # Negated negative acts as positive/favorable
                else:
                    n_neg += 1.0

        total_tokens = max(1, content_word_count)
        return float((n_pos - n_neg) / (total_tokens + 1.0))

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Compute net polarity scores for a sequence of texts."""
        return np.array([self.score_sentence(t) for t in texts], dtype=np.float64)

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
    ) -> LoughranMcDonaldBaseline:
        """Tune decision thresholds (tau_neg, tau_pos) on validation set for Macro-F1."""
        target_texts = val_texts if (val_texts is not None and len(val_texts) > 0) else None
        target_labels = val_labels if (val_labels is not None and len(val_labels) > 0) else None

        if target_texts is None or target_labels is None:
            self.best_params_ = {
                "threshold_neg": self.threshold_neg_,
                "threshold_pos": self.threshold_pos_,
                "val_macro_f1": None,
            }
            return self

        val_scores = self.score_texts(target_texts)
        y_true = np.asarray(target_labels, dtype=int)

        best_f1 = -1.0
        best_neg = self.threshold_neg_
        best_pos = self.threshold_pos_

        # Candidate threshold sweep
        neg_candidates = [-0.30, -0.20, -0.15, -0.10, -0.05, -0.02, -0.01, -0.005, 0.0]
        pos_candidates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

        # Add empirical score percentiles
        if len(val_scores) > 0:
            for q in [10, 20, 30, 40]:
                neg_candidates.append(float(np.percentile(val_scores, q)))
            for q in [60, 70, 80, 90]:
                pos_candidates.append(float(np.percentile(val_scores, q)))

        neg_candidates = sorted(set(round(x, 4) for x in neg_candidates))
        pos_candidates = sorted(set(round(x, 4) for x in pos_candidates))

        for t_neg in neg_candidates:
            for t_pos in pos_candidates:
                if t_neg >= t_pos:
                    continue
                # Predict
                preds = np.where(val_scores <= t_neg, 0, np.where(val_scores >= t_pos, 2, 1))
                f1 = _compute_macro_f1(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_neg = t_neg
                    best_pos = t_pos
                elif math.isclose(f1, best_f1, abs_tol=1e-5):
                    # Tie-break: prefer thresholds closer to (-0.05, 0.05)
                    cur_dist = (best_neg + 0.05) ** 2 + (best_pos - 0.05) ** 2
                    cand_dist = (t_neg + 0.05) ** 2 + (t_pos - 0.05) ** 2
                    if cand_dist < cur_dist:
                        best_neg = t_neg
                        best_pos = t_pos

        self.threshold_neg_ = best_neg
        self.threshold_pos_ = best_pos
        self.best_params_ = {
            "threshold_neg": best_neg,
            "threshold_pos": best_pos,
            "val_macro_f1": float(best_f1),
        }
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Output 3-class probability distribution on simplex."""
        scores = self.score_texts(texts)
        return _logits_to_probability_simplex(scores, self.threshold_neg_, self.threshold_pos_)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Output 3-class predicted integer labels [0, 1, 2]."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)


class VaderBaseline:
    """Tier 1 VADER Financial Sentiment Lexicon Baseline.

    Computes continuous valence score s in [-1, 1] per sentence using
    an embedded deterministic financial/general valence lexicon with booster
    and 3-token negation rules, with neutral accounting term handling.
    """

    def __init__(
        self,
        default_thresholds: tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        self.lexicon: dict[str, float] = dict(VADER_CORE_LEXICON)
        # Ensure accounting terms are strictly neutralized
        for term in NEUTRAL_ACCOUNTING_TERMS:
            self.lexicon[term] = 0.0

        self.boosters: dict[str, float] = dict(VADER_BOOSTER_DICT)
        self.threshold_neg_: float = float(default_thresholds[0])
        self.threshold_pos_: float = float(default_thresholds[1])
        self.best_params_: dict[str, Any] = {
            "threshold_neg": self.threshold_neg_,
            "threshold_pos": self.threshold_pos_,
            "val_macro_f1": None,
        }

    def score_sentence(self, text: str) -> float:
        """Compute VADER continuous compound valence score in [-1, 1]."""
        raw_tokens = _tokenize_text(text)
        if not raw_tokens:
            return 0.0

        word_tokens: list[str] = []
        for tok in raw_tokens:
            if tok in CLAUSE_BREAKERS:
                word_tokens.append("<BREAK>")
            else:
                word_tokens.append(tok)

        sentiments: list[float] = []

        for i, token in enumerate(word_tokens):
            if token == "<BREAK>":
                continue

            token_lower = token.lower()

            if token_lower in NEUTRAL_ACCOUNTING_TERMS:
                # Zero valence for neutral accounting entries
                sentiments.append(0.0)
                continue

            valence = self.lexicon.get(token_lower, 0.0)
            if valence == 0.0:
                continue

            # Check ALL CAPS booster
            if token.isupper() and len(token) > 1:
                if valence > 0:
                    valence += 0.733
                else:
                    valence -= 0.733

            # Lookback up to 3 tokens for negation and boosters
            window_start = max(0, i - 3)
            prior_tokens = word_tokens[window_start:i]
            if "<BREAK>" in prior_tokens:
                prior_tokens = prior_tokens[prior_tokens.index("<BREAK>") + 1 :]

            # Apply booster words in window
            for b_idx, b_tok in enumerate(prior_tokens):
                b_val = self.boosters.get(b_tok.lower(), 0.0)
                if b_val != 0.0:
                    # Decay booster effect with distance
                    dist = len(prior_tokens) - b_idx
                    decay = 0.95 if dist == 1 else (0.90 if dist == 2 else 0.85)
                    if valence > 0:
                        valence += b_val * decay
                    else:
                        valence -= b_val * decay

            # Apply 3-token window negation
            is_negated = any(_is_negation_token(t) for t in prior_tokens)
            if is_negated:
                valence *= -0.74

            sentiments.append(valence)

        if not sentiments:
            return 0.0

        sum_s = float(sum(sentiments))
        alpha = 15.0
        compound = sum_s / math.sqrt((sum_s * sum_s) + alpha)
        return float(np.clip(compound, -1.0, 1.0))

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Compute compound valence scores for a sequence of texts."""
        return np.array([self.score_sentence(t) for t in texts], dtype=np.float64)

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
    ) -> VaderBaseline:
        """Tune decision thresholds on validation set for Macro-F1."""
        target_texts = val_texts if (val_texts is not None and len(val_texts) > 0) else None
        target_labels = val_labels if (val_labels is not None and len(val_labels) > 0) else None

        if target_texts is None or target_labels is None:
            self.best_params_ = {
                "threshold_neg": self.threshold_neg_,
                "threshold_pos": self.threshold_pos_,
                "val_macro_f1": None,
            }
            return self

        val_scores = self.score_texts(target_texts)
        y_true = np.asarray(target_labels, dtype=int)

        best_f1 = -1.0
        best_neg = self.threshold_neg_
        best_pos = self.threshold_pos_

        neg_candidates = [-0.30, -0.20, -0.15, -0.10, -0.05, -0.02, -0.01, -0.005, 0.0]
        pos_candidates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

        if len(val_scores) > 0:
            for q in [10, 20, 30, 40]:
                neg_candidates.append(float(np.percentile(val_scores, q)))
            for q in [60, 70, 80, 90]:
                pos_candidates.append(float(np.percentile(val_scores, q)))

        neg_candidates = sorted(set(round(x, 4) for x in neg_candidates))
        pos_candidates = sorted(set(round(x, 4) for x in pos_candidates))

        for t_neg in neg_candidates:
            for t_pos in pos_candidates:
                if t_neg >= t_pos:
                    continue
                preds = np.where(val_scores <= t_neg, 0, np.where(val_scores >= t_pos, 2, 1))
                f1 = _compute_macro_f1(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_neg = t_neg
                    best_pos = t_pos
                elif math.isclose(f1, best_f1, abs_tol=1e-5):
                    cur_dist = (best_neg + 0.05) ** 2 + (best_pos - 0.05) ** 2
                    cand_dist = (t_neg + 0.05) ** 2 + (t_pos - 0.05) ** 2
                    if cand_dist < cur_dist:
                        best_neg = t_neg
                        best_pos = t_pos

        self.threshold_neg_ = best_neg
        self.threshold_pos_ = best_pos
        self.best_params_ = {
            "threshold_neg": best_neg,
            "threshold_pos": best_pos,
            "val_macro_f1": float(best_f1),
        }
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Output 3-class probability distribution on simplex."""
        scores = self.score_texts(texts)
        return _logits_to_probability_simplex(scores, self.threshold_neg_, self.threshold_pos_)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Output 3-class predicted integer labels [0, 1, 2]."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)
