# Diagnostic set: phenomenon-tagged sentences from Financial PhraseBank

This folder holds a 300-sentence diagnostic set used for the Step 3 phenomenon analysis (RQ1, RQ3) of the project
*Evaluating Domain Adaptation in Financial Sentiment Analysis*. The sentences are verbatim Financial PhraseBank text,
so the folder carries the dataset's licence (see "Licence and citation" below).

## Files

| File | What it is |
|---|---|
| `diagnostic_items.csv` | The 300 sentences: `diag_id` (d001 to d300) and `text`. This is the only file that annotators and taggers read. |
| `diagnostic_key.json` | Provenance key: the source config, seed, size, the SHA-256 of the source manifest, and for every item the source `record_id`, the gold `label` (0 negative, 1 neutral, 2 positive), and the SHA-256 of the text. |
| `rule_tags.csv` | Phenomenon tags from the rule tagger (`src/analysis/phenomena.py`), one row per item: financial_terminology, negation, hedging, numerical_comparison, context_dependent (0 or 1). Regenerated from the v2 tagger with `scripts/build_diagnostic_set.py --rule-tags-only --execute`, which keeps `diagnostic_items.csv` and the key unchanged. |
| `annotations_a.csv` | Tags from annotator A (LLM, see below) for the same five phenomena, with an optional `note`. `context_dependent` holds the v2 re-annotation. |
| `annotations_b.csv` | Tags from annotator B (LLM, see below), same format and same v2 re-annotation of `context_dependent`. |
| `annotators.json` | Who the two annotators are and the blind protocol they followed. |
| `annotations_a_v1.csv`, `annotations_b_v1.csv` | The first-round annotations, kept for the record, before the widened `context_dependent` definition (guideline v1). |

The tagging rules are in `annotation/GUIDELINES.md` (repository root, changelog at its end).

## How the set was made

- **Source:** the `sentences_50agree` config of Financial PhraseBank (Malo et al., 2014), **test split only**, from the
  locked 70/15/15 split manifest of Step 1 (manifest SHA-256 `d11c912d24b8ea2db5c553bd042d204a96e9ed21d61459b1719d47dde4e195e2`).
- **Why 50agree:** chosen before any annotation was made. Its test split is the largest of the four agreement levels
  and it contains more Step 1 model errors than the others, so the phenomenon analysis has more cases to explain.
  No model was changed after the annotation.
- **Sampling:** label-stratified. Counts per gold class are set by the largest-remainder rule over the test split,
  then drawn without replacement with NumPy `default_rng(42)`; the chosen rows are shuffled with the same generator.
- **Result:** 300 sentences: 178 neutral, 85 positive, 37 negative.

## Annotators (not human)

Both annotators are large language model agents, **not human annotators**. No human annotation was done for this set.

- Annotator A: Claude Sonnet 5.5 (`claude-sonnet-5-5`, Anthropic).
- Annotator B: Gemini 3.8 Flash (`gemini-3.8-flash-medium`, Google, run through the Antigravity CLI).

Both were run blind and independently: each saw only `diagnostic_items.csv` and the guideline, never the other
annotator's file, the gold labels, the model predictions or the rule tags. Each read the sentences in batches of about
25 and wrote the tags by hand (no keyword scripts). Caveat: both models may have seen Financial PhraseBank during
pretraining, so "blind" means that the labels and predictions were withheld in this run, not that the text was new to
the models.

## v2 re-annotation of `context_dependent`

The first round used a narrow definition of `context_dependent` (3 of 300 sentences tagged). Guideline v2 (2026-10-20)
widens it: a direction word (for example "fell", "rose", "cut") applied to a polarity-loaded financial target (for
example "profit" versus "costs"), plus job-cut patterns. Only this one tag was re-annotated, blind, by both annotators
with the v2 guideline. The other four tags are unchanged from round one. The round-one files are kept as
`annotations_*_v1.csv`.

## Licence and citation

- The sentences come from **Financial PhraseBank**, licensed under **CC BY-NC-SA 3.0**. The licence is recorded in
  `data/dataset_audit_verdict.json`.
- This derived file is therefore shared under the same licence: **non-commercial use only, with attribution, and any
  derivative must be shared under the same licence (ShareAlike).**
- Citation: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P. (2014). "Good debt or bad debt: Detecting
  semantic orientations in economic texts." *Journal of the Association for Information Science and Technology*, 65(4).
- The SEntFiN 1.0 data used for the external test is not redistributed here; only its ids and predictions are stored in the results.
