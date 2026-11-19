# Zero-shot financial sentiment classification

| Dataset | N | Macro-F1 | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| phrasebank_allagree_test | 339 | 0.9340 | 0.9351 | 0.9684 | 0.9487 | 0.8851 | 0 |
| phrasebank_50agree_test | 726 | 0.8194 | 0.8209 | 0.8523 | 0.8469 | 0.7591 | 0 |
| sentfin | 9460 | 0.8249 | 0.8216 | 0.8621 | 0.7692 | 0.8436 | 0 |
| fiqa | 1110 | 0.6031 | 0.6559 | 0.8138 | 0.2594 | 0.7362 | 0 |
| twitter | 11737 | 0.7432 | 0.7674 | 0.7056 | 0.8060 | 0.7179 | 0 |

For the PhraseBank test sets, the LLM may have seen Financial PhraseBank during pretraining.

## Comparisons with fixed-recipe transformers

Δ is the zero-shot Macro-F1 minus the mean fine-tuned Macro-F1. The 95% intervals use 10000 paired sentence and seed resamples. Each McNemar test pairs the same zero-shot run with one fine-tuning seed. Holm adjustment covers the two reference models within each dataset and seed.

| Dataset | Reference | Δ Macro-F1 | 95% interval | McNemar p / Holm p by seed |
|---|---|---:|---|---|
| phrasebank_allagree_test | bert_base | -0.0399 | [-0.0756, -0.0039] | 42 (0.01612 / 0.03225), 100 (0.02945 / 0.0589), 2026 (0.0001565 / 0.000313) |
| phrasebank_allagree_test | finbert_mlm | -0.0104 | [-0.0504, +0.0296] | 42 (0.405 / 0.405), 100 (0.1755 / 0.1755), 2026 (0.1755 / 0.1755) |
| phrasebank_50agree_test | bert_base | -0.0526 | [-0.0875, -0.0182] | 42 (2.755e-05 / 5.511e-05), 100 (0.003172 / 0.006344), 2026 (0.0005801 / 0.00116) |
| phrasebank_50agree_test | finbert_mlm | -0.0266 | [-0.0633, +0.0102] | 42 (0.0002417 / 0.0002417), 100 (0.01112 / 0.01112), 2026 (0.04739 / 0.04739) |
| sentfin | bert_base | +0.1399 | [+0.1174, +0.1591] | 42 (1.245e-149 / 2.489e-149), 100 (1.502e-104 / 1.502e-104), 2026 (4.517e-157 / 6.42e-157) |
| sentfin | finbert_mlm | +0.1558 | [+0.1281, +0.1832] | 42 (3.739e-116 / 3.739e-116), 100 (4.497e-193 / 8.995e-193), 2026 (3.21e-157 / 6.42e-157) |
| fiqa | bert_base | +0.2218 | [+0.1916, +0.2525] | 42 (7.148e-60 / 1.43e-59), 100 (4.441e-57 / 4.441e-57), 2026 (3.911e-67 / 3.911e-67) |
| fiqa | finbert_mlm | +0.2552 | [+0.1926, +0.3118] | 42 (3.284e-44 / 3.284e-44), 100 (2.286e-92 / 4.572e-92), 2026 (5.081e-82 / 1.016e-81) |
| twitter | bert_base | +0.1092 | [+0.0918, +0.1271] | 42 (4.229e-17 / 8.458e-17), 100 (1.052e-13 / 2.103e-13), 2026 (8.373e-27 / 1.675e-26) |
| twitter | finbert_mlm | +0.0801 | [+0.0614, +0.0986] | 42 (0.004586 / 0.004586), 100 (0.005892 / 0.005892), 2026 (0.01326 / 0.01326) |
