# Annotation guidelines: five linguistic phenomena in financial sentences

## Purpose

You receive `diagnostic_items.csv` (columns `diag_id,text`, 300 English financial sentences). For each sentence,
decide which of five linguistic phenomena it contains. The tags are later used to see where sentiment models make
more errors. A sentence can carry several tags, or none.

**Judge the text only.** No sentiment label is given and none is needed. Do not guess a label, do not use outside
knowledge about the company, and do not try to decide whether the sentence is good or bad news. Mark a phenomenon
only if the words of the sentence contain it. When unsure, use 0 and write a short `note`.

## The five phenomena

### 1. financial_terminology (financial_terminology)

A finance or accounting term whose everyday-English polarity differs from its mostly descriptive use in finance:
cost(s), expense(s), liability/liabilities, debt, depreciation, amortisation/amortization, impairment,
write-down/write-off, provision(s), charge(s), tax(es), loss(es) used as a line item, deficit, leverage,
exposure, risk(s), restructuring, short (position), bear/bearish, volatility, claims.

- Include: the term is used as a financial or accounting item or concept.
- Exclude: the same word in a non-financial sense (a "charge" of assault, "bear" the animal, "claims" in an
  argument unrelated to insurance or legal claims against a company).

Positive examples (invented):
1. "Quarterly depreciation of the plant rose with the new equipment."
2. "The firm booked a provision for pending tax claims."

Negative examples (invented):
1. "The bakery opened a second shop near the station."
2. "Management welcomed the new board members at the meeting."

### 2. negation (negation)

Explicit negation that scopes over a clause: not, no, never, without, n't, neither, nor, none, cannot, failed to,
lack/lacked.

- Include: the negator changes what is claimed in the clause.
- Exclude: fixed names or labels ("No. 1 supplier", "Not Applicable" column headers), "no" as an abbreviation
  of "number".

Positive examples (invented):
1. "The company did not meet its delivery schedule."
2. "Sales never recovered without the new contract."

Negative examples (invented):
1. "The company met its delivery schedule."
2. "Sales recovered after the new contract."

### 3. hedging (hedging)

Uncertainty, expectation or forward-looking modality: may, might, could, would, expect(s/ed), anticipate(s/d),
estimate(s/d), forecast(s), guidance, outlook, likely, possible/possibly, potential(ly), approximately,
about/around + a number, believe(s), plan(s) to, aim(s) to, target(s), projected, seen as.

- Include: the sentence presents something as expected, planned, estimated or uncertain instead of as a fact.
- Exclude: statements of completed facts that merely use a hedging word as a name ("Potential Inc.").

Positive examples (invented):
1. "The board expects demand to pick up next year."
2. "Revenue could reach around 40 million by the end of the period."

Negative examples (invented):
1. "The board approved the budget on Tuesday."
2. "The factory employs 200 people."

### 4. numerical_comparison (numerical_comparison)

A number, percentage or currency amount combined with a comparison or change: rose, fell, increased, decreased,
grew, declined, dropped, jumped, up, down, from X to Y, compared with/to, versus/vs, higher/lower than,
year-on-year, y-o-y, doubled, halved.

- Include: a quantity is placed against another quantity, a past value or a change in value.
- Exclude: a number without any comparison or change ("Sales were 12 million"), a comparison without any
  number ("sales were higher"), except doubled/halved which imply a number.

Positive examples (invented):
1. "Net sales rose to 52 million from 47 million a year earlier."
2. "Output was 8 percent lower than in the previous quarter."

Negative examples (invented):
1. "Net sales were 52 million."
2. "Output was lower than in the previous quarter."

### 5. context_dependent (context_dependent)

The polarity of the sentence depends on the direction applied to a polarity-loaded target, so word-level polarity
misleads: decrease in losses, costs fell, loss narrowed, debt reduced, increased loss, cut jobs, lower costs.
Rule of thumb: a direction word (increase, decrease, fall, rise, reduce, narrow, widen, cut, lower, higher)
within 4 tokens of a negative-valence financial noun (loss, cost, debt, deficit, expense, liability,
impairment, unemployment, layoffs).

- Include: you need the direction AND the target to tell whether the news is good or bad.
- Exclude: a direction word with a neutral target ("sales rose"), or a negative noun without any direction.

Positive examples (invented):
1. "The deficit narrowed during the second half."
2. "Operating costs fell after the move to the new site."

Negative examples (invented):
1. "Sales rose during the second half."
2. "Operating costs are listed in the annex."

## Output format

One CSV file (UTF-8, header row, one row per sentence, all 300 sentences, same `diag_id` values as the input):

```
diag_id,financial_terminology,negation,hedging,numerical_comparison,context_dependent,note
d001,1,0,0,1,1,
d002,0,0,1,0,0,short free text
```

- The five phenomenon columns hold 0 or 1 only.
- `note` is optional free text. It must not contain commas, and it must not be quoted. Leave it empty if unused.
- Work independently of the other annotator and of any rule-based tags.
