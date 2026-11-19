# Step 4: tuned recipe against the fixed Step 2 recipe

Written by `scripts/summarize_step4.py` from stored predictions; no model is trained or loaded.

## Protocol

Models `bert_base` (BERT) and `finbert_mlm` (FinBERT) on `sentences_allagree`, seeds 42, 100, 2026, 7, 123. Fixed recipe = the Step 2 runs; tuned recipe = the recipe that `results/tuning/selection.json` selected on validation Macro-F1 with seed 42, retrained for all 5 seeds.

Pre-specified seeds in this analysis were 42, 100, 2026. Seeds added later were 7, 123.

- BERT: fixed learning rate 2e-05, batch size 16; tuned learning rate 3e-05, batch size 32 (selection validation Macro-F1 0.9576).
- FinBERT: fixed learning rate 2e-05, batch size 16; tuned learning rate 3e-05, batch size 32 (selection validation Macro-F1 0.9568).

All 20 runs (fixed and tuned) record the same device and library versions in their `result.json`: device `mps`; numpy 1.26.2, python 3.9.6, sklearn 1.6.1, torch 2.8.0, transformers 4.57.6. The fixed-vs-tuned comparison is therefore inside one software environment.

Splits: PhraseBank test (n = 339) and SEntFiN (external) (n = 9460). All runs were checked to share the same sentence ids and labels per split.

Intervals: seed+sentence bootstrap (sentences and seeds resampled), B = 10000, random seed 2026, 95% percentile interval, two-sided p. McNemar: exact test per seed pairing, Holm-Bonferroni inside each seed family over the 8 tests (4 pairs x 2 splits), alpha = 0.05. Every difference is A - B. In every interval, including the gap-change interval, the seeds of each run set are resampled independently of the other run sets. With 5 seeds the seed-to-seed spread is estimated from very few runs, so the intervals are approximate.

## Scores

| Split | Model | Recipe | Macro-F1 seed 42 | Macro-F1 seed 100 | Macro-F1 seed 2026 | Macro-F1 seed 7 | Macro-F1 seed 123 | Macro-F1 mean | sd | Accuracy mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PhraseBank test | BERT | fixed | 0.9691 | 0.9639 | 0.9888 | 0.9605 | 0.9625 | 0.9690 | 0.0116 | 0.9764 |
| PhraseBank test | BERT | tuned | 0.9695 | 0.9672 | 0.9752 | 0.9570 | 0.9593 | 0.9656 | 0.0075 | 0.9746 |
| PhraseBank test | FinBERT | fixed | 0.9389 | 0.9493 | 0.9452 | 0.9707 | 0.9246 | 0.9457 | 0.0168 | 0.9593 |
| PhraseBank test | FinBERT | tuned | 0.9628 | 0.9454 | 0.9539 | 0.9570 | 0.9517 | 0.9542 | 0.0064 | 0.9646 |
| SEntFiN (external) | BERT | fixed | 0.6747 | 0.7087 | 0.6716 | 0.6680 | 0.6926 | 0.6831 | 0.0171 | 0.6824 |
| SEntFiN (external) | BERT | tuned | 0.6743 | 0.6785 | 0.6806 | 0.6377 | 0.6940 | 0.6730 | 0.0211 | 0.6754 |
| SEntFiN (external) | FinBERT | fixed | 0.6979 | 0.6422 | 0.6672 | 0.6907 | 0.7062 | 0.6808 | 0.0260 | 0.6811 |
| SEntFiN (external) | FinBERT | tuned | 0.7235 | 0.6751 | 0.6863 | 0.7048 | 0.6925 | 0.6964 | 0.0186 | 0.6950 |

## Paired comparisons (Macro-F1, A - B)

| Split | A | B | Delta | Delta seed 42 | Delta seed 100 | Delta seed 2026 | Delta seed 7 | Delta seed 123 | 95% interval | Bootstrap p | McNemar p min | McNemar p max | Seeds significant (Holm) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| PhraseBank test | tuned BERT | tuned FinBERT | +0.0115 | +0.0067 | +0.0218 | +0.0213 | -0.0000 | +0.0076 | [-0.0101, +0.0335] | 0.2794 | 0.1797 | 0.8145 | 0/5 |
| PhraseBank test | fixed BERT | fixed FinBERT | +0.0232 | +0.0302 | +0.0146 | +0.0437 | -0.0102 | +0.0379 | [-0.0017, +0.0495] | 0.0688 | 0.0020 | 0.7539 | 1/5 |
| PhraseBank test | tuned BERT | fixed BERT | -0.0033 | +0.0004 | +0.0034 | -0.0136 | -0.0035 | -0.0032 | [-0.0198, +0.0121] | 0.6916 | 0.2188 | 1.0000 | 0/5 |
| PhraseBank test | tuned FinBERT | fixed FinBERT | +0.0084 | +0.0238 | -0.0039 | +0.0087 | -0.0136 | +0.0272 | [-0.0114, +0.0310] | 0.4266 | 0.0625 | 1.0000 | 0/5 |
| SEntFiN (external) | tuned BERT | tuned FinBERT | -0.0234 | -0.0492 | +0.0035 | -0.0057 | -0.0671 | +0.0015 | [-0.0479, -0.0015] | 0.0348 | <0.0001 | 0.5604 | 2/5 |
| SEntFiN (external) | fixed BERT | fixed FinBERT | +0.0023 | -0.0232 | +0.0665 | +0.0044 | -0.0227 | -0.0136 | [-0.0218, +0.0293] | 0.8896 | <0.0001 | 0.3877 | 4/5 |
| SEntFiN (external) | tuned BERT | fixed BERT | -0.0101 | -0.0004 | -0.0301 | +0.0089 | -0.0303 | +0.0014 | [-0.0331, +0.0099] | 0.3870 | <0.0001 | 0.8719 | 3/5 |
| SEntFiN (external) | tuned FinBERT | fixed FinBERT | +0.0156 | +0.0256 | +0.0329 | +0.0191 | +0.0142 | -0.0137 | [-0.0081, +0.0423] | 0.2086 | <0.0001 | <0.0001 | 5/5 |

In plain words:

- On PhraseBank test, tuned BERT scores 0.0115 Macro-F1 higher than tuned FinBERT (seed means); the 95% interval [-0.0101, +0.0335] contains zero, and McNemar is significant after Holm in 0 of 5 seed pairings.
- On PhraseBank test, fixed BERT scores 0.0232 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0017, +0.0495] contains zero, and McNemar is significant after Holm in 1 of 5 seed pairings.
- On PhraseBank test, tuned BERT scores 0.0033 Macro-F1 lower than fixed BERT (seed means); the 95% interval [-0.0198, +0.0121] contains zero, and McNemar is significant after Holm in 0 of 5 seed pairings.
- On PhraseBank test, tuned FinBERT scores 0.0084 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0114, +0.0310] contains zero, and McNemar is significant after Holm in 0 of 5 seed pairings.
- On SEntFiN (external), tuned BERT scores 0.0234 Macro-F1 lower than tuned FinBERT (seed means); the 95% interval [-0.0479, -0.0015] does not contain zero, and McNemar is significant after Holm in 2 of 5 seed pairings.
- On SEntFiN (external), fixed BERT scores 0.0023 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0218, +0.0293] contains zero, and McNemar is significant after Holm in 4 of 5 seed pairings.
- On SEntFiN (external), tuned BERT scores 0.0101 Macro-F1 lower than fixed BERT (seed means); the 95% interval [-0.0331, +0.0099] contains zero, and McNemar is significant after Holm in 3 of 5 seed pairings.
- On SEntFiN (external), tuned FinBERT scores 0.0156 Macro-F1 higher than fixed FinBERT (seed means); the 95% interval [-0.0081, +0.0423] contains zero, and McNemar is significant after Holm in 5 of 5 seed pairings.

## Change of the BERT-FinBERT gap

Gap change = (tuned BERT - tuned FinBERT) - (fixed BERT - fixed FinBERT). Its interval comes from one bootstrap in which the sentences are resampled once per draw for all four run sets and the seeds of the four run sets are resampled independently: tuned and fixed runs with the same seed number are different trainings, so they are not paired.

| Split | Gap change | 95% interval | Bootstrap p |
|---|---:|---|---:|
| PhraseBank test | -0.0118 | [-0.0395, +0.0132] | 0.3532 |
| SEntFiN (external) | -0.0257 | [-0.0598, +0.0051] | 0.1106 |

- On PhraseBank test, tuning changes the BERT-minus-FinBERT gap by -0.0118 Macro-F1; the 95% interval [-0.0395, +0.0132] contains zero.
- On SEntFiN (external), tuning changes the BERT-minus-FinBERT gap by -0.0257 Macro-F1; the 95% interval [-0.0598, +0.0051] contains zero.

## Run directories

- fixed BERT: `results/step2/runs/sentences_allagree/bert_base/seed42`, `results/step2/runs/sentences_allagree/bert_base/seed100`, `results/step2/runs/sentences_allagree/bert_base/seed2026`, `results/tuning/fixed_extra/sentences_allagree/bert_base/seed7`, `results/tuning/fixed_extra/sentences_allagree/bert_base/seed123`
- fixed FinBERT: `results/step2/runs/sentences_allagree/finbert_mlm/seed42`, `results/step2/runs/sentences_allagree/finbert_mlm/seed100`, `results/step2/runs/sentences_allagree/finbert_mlm/seed2026`, `results/tuning/fixed_extra/sentences_allagree/finbert_mlm/seed7`, `results/tuning/fixed_extra/sentences_allagree/finbert_mlm/seed123`
- tuned BERT: `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed42`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed100`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed2026`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed7`, `results/tuning/final/sentences_allagree/bert_base/lr3e-05_bs32_seed123`
- tuned FinBERT: `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed42`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed100`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed2026`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed7`, `results/tuning/final/sentences_allagree/finbert_mlm/lr3e-05_bs32_seed123`
