# Step 4: tuned recipe against the fixed Step 2 recipe

Written by `scripts/summarize_step4.py` from stored predictions; no model is trained or loaded.

## Protocol

Models `bert_base` (BERT) and `finbert_mlm` (FinBERT) on `sentences_allagree`, seeds 42, 100, 2026. Fixed recipe = the Step 2 runs; tuned recipe = the recipe that `results/tuning/selection.json` selected on validation Macro-F1 with seed 42, retrained for all three seeds.

- BERT: fixed learning rate 2e-05, batch size 16; tuned learning rate 3e-05, batch size 32 (selection validation Macro-F1 0.9576).
- FinBERT: fixed learning rate 2e-05, batch size 16; tuned learning rate 3e-05, batch size 32 (selection validation Macro-F1 0.9568).

All 12 runs (fixed and tuned) record the same device and library versions in their `result.json`: device `mps`; numpy 1.26.2, python 3.9.6, sklearn 1.6.1, torch 2.8.0, transformers 4.57.6. The fixed-vs-tuned comparison is therefore inside one software environment.

Splits: PhraseBank test (n = 339) and SEntFiN (external) (n = 9460). All runs were checked to share the same sentence ids and labels per split.

Intervals: seed+sentence bootstrap (sentences and seeds resampled), B = 10000, random seed 2026, 95% percentile interval, two-sided p. McNemar: exact test per seed pairing, Holm-Bonferroni inside each seed family over the 8 tests (4 pairs x 2 splits), alpha = 0.05. Every difference is A - B. In every interval, including the gap-change interval, the seeds of each run set are resampled independently of the other run sets. With three seeds the seed-to-seed spread is estimated from very few runs, so the intervals are approximate.

## Scores

| Split | Model | Recipe | Macro-F1 seed 42 | Macro-F1 seed 100 | Macro-F1 seed 2026 | Macro-F1 mean | sd | Accuracy mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PhraseBank test | BERT | fixed | 0.9691 | 0.9639 | 0.9888 | 0.9739 | 0.0132 | 0.9803 |
| PhraseBank test | BERT | tuned | 0.9695 | 0.9672 | 0.9752 | 0.9706 | 0.0041 | 0.9764 |
| PhraseBank test | FinBERT | fixed | 0.9389 | 0.9493 | 0.9452 | 0.9445 | 0.0052 | 0.9587 |
| PhraseBank test | FinBERT | tuned | 0.9628 | 0.9454 | 0.9539 | 0.9540 | 0.0087 | 0.9646 |
| SEntFiN (external) | BERT | fixed | 0.6747 | 0.7087 | 0.6716 | 0.6850 | 0.0205 | 0.6841 |
| SEntFiN (external) | BERT | tuned | 0.6743 | 0.6785 | 0.6806 | 0.6778 | 0.0032 | 0.6784 |
| SEntFiN (external) | FinBERT | fixed | 0.6979 | 0.6422 | 0.6672 | 0.6691 | 0.0279 | 0.6703 |
| SEntFiN (external) | FinBERT | tuned | 0.7235 | 0.6751 | 0.6863 | 0.6949 | 0.0254 | 0.6938 |

## Paired comparisons (Macro-F1, A - B)

| Split | A | B | Delta | Delta seed 42 | Delta seed 100 | Delta seed 2026 | 95% interval | Bootstrap p | McNemar p min | McNemar p max | Seeds significant (Holm) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| PhraseBank test | tuned BERT | tuned FinBERT | +0.0166 | +0.0067 | +0.0218 | +0.0213 | [-0.0050, +0.0425] | 0.1230 | 0.1797 | 0.7266 | 0/3 |
| PhraseBank test | fixed BERT | fixed FinBERT | +0.0295 | +0.0302 | +0.0146 | +0.0437 | [+0.0061, +0.0572] | 0.0138 | 0.0020 | 0.4240 | 1/3 |
| PhraseBank test | tuned BERT | fixed BERT | -0.0033 | +0.0004 | +0.0034 | -0.0136 | [-0.0227, +0.0180] | 0.7492 | 0.2188 | 1.0000 | 0/3 |
| PhraseBank test | tuned FinBERT | fixed FinBERT | +0.0096 | +0.0238 | -0.0039 | +0.0087 | [-0.0095, +0.0312] | 0.3084 | 0.0625 | 1.0000 | 0/3 |
| SEntFiN (external) | tuned BERT | tuned FinBERT | -0.0171 | -0.0492 | +0.0035 | -0.0057 | [-0.0444, +0.0048] | 0.1838 | <0.0001 | 0.4508 | 1/3 |
| SEntFiN (external) | fixed BERT | fixed FinBERT | +0.0159 | -0.0232 | +0.0665 | +0.0044 | [-0.0164, +0.0497] | 0.3586 | <0.0001 | 0.3877 | 2/3 |
| SEntFiN (external) | tuned BERT | fixed BERT | -0.0072 | -0.0004 | -0.0301 | +0.0089 | [-0.0297, +0.0083] | 0.5462 | <0.0001 | 0.8719 | 2/3 |
| SEntFiN (external) | tuned FinBERT | fixed FinBERT | +0.0258 | +0.0256 | +0.0329 | +0.0191 | [-0.0077, +0.0614] | 0.1534 | <0.0001 | <0.0001 | 3/3 |

In plain words:

- On PhraseBank test, tuned BERT scores 0.0166 Macro-F1 higher than tuned FinBERT (seed means); the 95% interval [-0.0050, +0.0425] contains zero, and McNemar is significant after Holm in 0 of 3 seed pairings.
- On PhraseBank test, fixed BERT scores 0.0295 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [+0.0061, +0.0572] does not contain zero, and McNemar is significant after Holm in 1 of 3 seed pairings.
- On PhraseBank test, tuned BERT scores 0.0033 Macro-F1 lower than fixed BERT (seed means); the 95% interval [-0.0227, +0.0180] contains zero, and McNemar is significant after Holm in 0 of 3 seed pairings.
- On PhraseBank test, tuned FinBERT scores 0.0096 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0095, +0.0312] contains zero, and McNemar is significant after Holm in 0 of 3 seed pairings.
- On SEntFiN (external), tuned BERT scores 0.0171 Macro-F1 lower than tuned FinBERT (seed means); the 95% interval [-0.0444, +0.0048] contains zero, and McNemar is significant after Holm in 1 of 3 seed pairings.
- On SEntFiN (external), fixed BERT scores 0.0159 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0164, +0.0497] contains zero, and McNemar is significant after Holm in 2 of 3 seed pairings.
- On SEntFiN (external), tuned BERT scores 0.0072 Macro-F1 lower than fixed BERT (seed means); the 95% interval [-0.0297, +0.0083] contains zero, and McNemar is significant after Holm in 2 of 3 seed pairings.
- On SEntFiN (external), tuned FinBERT scores 0.0258 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0077, +0.0614] contains zero, and McNemar is significant after Holm in 3 of 3 seed pairings.

## Change of the BERT-FinBERT gap

Gap change = (tuned BERT - tuned FinBERT) - (fixed BERT - fixed FinBERT). Its interval comes from one bootstrap in which the sentences are resampled once per draw for all four run sets and the seeds of the four run sets are resampled independently: tuned and fixed runs with the same seed number are different trainings, so they are not paired.

| Split | Gap change | 95% interval | Bootstrap p |
|---|---:|---|---:|
| PhraseBank test | -0.0129 | [-0.0387, +0.0130] | 0.2840 |
| SEntFiN (external) | -0.0330 | [-0.0734, +0.0066] | 0.1096 |

- On PhraseBank test, tuning changes the BERT-minus-FinBERT gap by -0.0129 Macro-F1; the 95% interval [-0.0387, +0.0130] contains zero.
- On SEntFiN (external), tuning changes the BERT-minus-FinBERT gap by -0.0330 Macro-F1; the 95% interval [-0.0734, +0.0066] contains zero.

## Run directories

- fixed BERT: `results/step2/runs/sentences_allagree/bert_base/seed42`, `results/step2/runs/sentences_allagree/bert_base/seed100`, `results/step2/runs/sentences_allagree/bert_base/seed2026`
- fixed FinBERT: `results/step2/runs/sentences_allagree/finbert_mlm/seed42`, `results/step2/runs/sentences_allagree/finbert_mlm/seed100`, `results/step2/runs/sentences_allagree/finbert_mlm/seed2026`
- tuned BERT: `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed42`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed100`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed2026`
- tuned FinBERT: `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed42`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed100`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed2026`
