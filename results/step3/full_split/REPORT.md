# Full 50agree test-split phenomenon analysis

The analysis covers all 726 locked test records (300 original and 426 extension items). A phenomenon is present when both blind LLM annotators marked it.

## Annotator agreement

| Phenomenon | Cohen’s κ A–B (extension) | Cohen’s κ A–B (union) |
|---|---|---|
| financial_terminology | 0.979 | 0.973 |
| negation | 1.000 | 1.000 |
| hedging | 0.947 | 0.923 |
| numerical_comparison | 0.945 | 0.952 |
| context_dependent | 0.878 | 0.917 |

## Error rates

95% intervals use 10,000 bootstrap draws over sentences and model seeds. Label-balanced error averages the error rates of the gold classes present.

### financial_terminology

- Tagged (n = 38) has 10 negative, 11 neutral, and 17 positive gold labels.
- Untagged (n = 688) has 80 negative, 419 neutral, and 189 positive gold labels.

| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |
|---|---|---|---|---|
| VADER (official lexicon) | 0.500 [0.342, 0.658] | 0.522 [0.358, 0.685] | 0.438 [0.400, 0.475] | 0.479 [0.438, 0.518] |
| VADER (official lexicon, val-tuned thresholds) | 0.658 [0.500, 0.816] | 0.640 [0.478, 0.794] | 0.398 [0.362, 0.435] | 0.473 [0.432, 0.513] |
| Loughran-McDonald (official dictionary) | 0.605 [0.447, 0.763] | 0.515 [0.410, 0.619] | 0.374 [0.337, 0.410] | 0.512 [0.468, 0.554] |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.605 [0.447, 0.763] | 0.515 [0.410, 0.619] | 0.374 [0.337, 0.410] | 0.512 [0.468, 0.554] |
| VADER-style (finance-adjusted, custom) | 0.474 [0.316, 0.632] | 0.388 [0.281, 0.514] | 0.257 [0.225, 0.289] | 0.384 [0.340, 0.427] |
| LM-style (finance-adjusted, custom) | 0.605 [0.447, 0.763] | 0.552 [0.403, 0.705] | 0.321 [0.286, 0.355] | 0.442 [0.397, 0.486] |
| TF-IDF + Logistic Regression | 0.211 [0.079, 0.342] | 0.247 [0.111, 0.389] | 0.202 [0.173, 0.233] | 0.253 [0.212, 0.297] |
| TF-IDF + Linear SVM | 0.211 [0.079, 0.342] | 0.247 [0.111, 0.389] | 0.195 [0.166, 0.225] | 0.258 [0.216, 0.301] |
| bert-base-uncased | 0.175 [0.070, 0.289] | 0.199 [0.082, 0.331] | 0.116 [0.093, 0.140] | 0.121 [0.091, 0.156] |
| FinBERT-MLM (finbert-pretrain) | 0.175 [0.079, 0.289] | 0.206 [0.092, 0.335] | 0.131 [0.105, 0.157] | 0.147 [0.113, 0.182] |
| Zero-shot LLM | 0.132 [0.026, 0.263] | 0.119 [0.030, 0.231] | 0.182 [0.153, 0.211] | 0.185 [0.149, 0.222] |

### negation

- Tagged (n = 25) has 5 negative, 18 neutral, and 2 positive gold labels.
- Untagged (n = 701) has 85 negative, 412 neutral, and 204 positive gold labels.

| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |
|---|---|---|---|---|
| VADER (official lexicon) | 0.440 [0.240, 0.640] | 0.300 [0.137, 0.566] | 0.441 [0.404, 0.478] | 0.490 [0.450, 0.528] |
| VADER (official lexicon, val-tuned thresholds) | 0.360 [0.160, 0.560] | 0.263 [0.100, 0.512] | 0.414 [0.377, 0.451] | 0.493 [0.452, 0.532] |
| Loughran-McDonald (official dictionary) | 0.440 [0.240, 0.640] | 0.544 [0.278, 0.800] | 0.384 [0.348, 0.419] | 0.493 [0.450, 0.535] |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.440 [0.240, 0.640] | 0.544 [0.278, 0.800] | 0.384 [0.348, 0.419] | 0.493 [0.450, 0.535] |
| VADER-style (finance-adjusted, custom) | 0.320 [0.160, 0.520] | 0.685 [0.500, 0.727] | 0.267 [0.235, 0.300] | 0.369 [0.326, 0.412] |
| LM-style (finance-adjusted, custom) | 0.320 [0.160, 0.520] | 0.489 [0.225, 0.737] | 0.337 [0.301, 0.372] | 0.444 [0.400, 0.487] |
| TF-IDF + Logistic Regression | 0.240 [0.080, 0.400] | 0.600 [0.333, 0.667] | 0.201 [0.171, 0.231] | 0.244 [0.204, 0.285] |
| TF-IDF + Linear SVM | 0.280 [0.120, 0.480] | 0.667 [0.500, 0.667] | 0.193 [0.164, 0.223] | 0.244 [0.203, 0.286] |
| bert-base-uncased | 0.147 [0.027, 0.293] | 0.296 [0.017, 0.566] | 0.118 [0.096, 0.142] | 0.122 [0.093, 0.154] |
| FinBERT-MLM (finbert-pretrain) | 0.213 [0.080, 0.373] | 0.423 [0.167, 0.667] | 0.130 [0.106, 0.156] | 0.141 [0.111, 0.173] |
| Zero-shot LLM | 0.360 [0.160, 0.560] | 0.459 [0.172, 0.737] | 0.173 [0.146, 0.201] | 0.167 [0.135, 0.200] |

### hedging

- Tagged (n = 117) has 15 negative, 71 neutral, and 31 positive gold labels.
- Untagged (n = 609) has 75 negative, 359 neutral, and 175 positive gold labels.

| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |
|---|---|---|---|---|
| VADER (official lexicon) | 0.350 [0.265, 0.436] | 0.387 [0.278, 0.496] | 0.458 [0.419, 0.498] | 0.502 [0.461, 0.542] |
| VADER (official lexicon, val-tuned thresholds) | 0.333 [0.248, 0.419] | 0.390 [0.281, 0.499] | 0.427 [0.388, 0.466] | 0.504 [0.463, 0.545] |
| Loughran-McDonald (official dictionary) | 0.333 [0.248, 0.419] | 0.480 [0.375, 0.580] | 0.396 [0.356, 0.433] | 0.502 [0.456, 0.546] |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.333 [0.248, 0.419] | 0.480 [0.375, 0.580] | 0.396 [0.356, 0.433] | 0.502 [0.456, 0.546] |
| VADER-style (finance-adjusted, custom) | 0.239 [0.162, 0.316] | 0.391 [0.292, 0.486] | 0.274 [0.238, 0.310] | 0.375 [0.327, 0.421] |
| LM-style (finance-adjusted, custom) | 0.274 [0.197, 0.350] | 0.375 [0.267, 0.482] | 0.348 [0.310, 0.386] | 0.461 [0.416, 0.508] |
| TF-IDF + Logistic Regression | 0.256 [0.179, 0.333] | 0.365 [0.258, 0.471] | 0.192 [0.161, 0.225] | 0.232 [0.191, 0.275] |
| TF-IDF + Linear SVM | 0.248 [0.171, 0.325] | 0.349 [0.242, 0.456] | 0.186 [0.156, 0.218] | 0.239 [0.197, 0.283] |
| bert-base-uncased | 0.120 [0.063, 0.182] | 0.144 [0.069, 0.230] | 0.119 [0.095, 0.144] | 0.122 [0.089, 0.158] |
| FinBERT-MLM (finbert-pretrain) | 0.137 [0.080, 0.199] | 0.189 [0.102, 0.285] | 0.132 [0.106, 0.161] | 0.144 [0.111, 0.181] |
| Zero-shot LLM | 0.154 [0.094, 0.222] | 0.202 [0.108, 0.304] | 0.184 [0.153, 0.215] | 0.172 [0.138, 0.209] |

### numerical_comparison

- Tagged (n = 139) has 54 negative, 7 neutral, and 78 positive gold labels.
- Untagged (n = 587) has 36 negative, 423 neutral, and 128 positive gold labels.

| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |
|---|---|---|---|---|
| VADER (official lexicon) | 0.561 [0.475, 0.640] | 0.647 [0.510, 0.759] | 0.412 [0.373, 0.453] | 0.371 [0.310, 0.432] |
| VADER (official lexicon, val-tuned thresholds) | 0.633 [0.554, 0.712] | 0.646 [0.499, 0.790] | 0.359 [0.320, 0.399] | 0.367 [0.305, 0.429] |
| Loughran-McDonald (official dictionary) | 0.791 [0.719, 0.856] | 0.589 [0.507, 0.707] | 0.290 [0.254, 0.327] | 0.391 [0.330, 0.454] |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.791 [0.719, 0.856] | 0.589 [0.507, 0.707] | 0.290 [0.254, 0.327] | 0.391 [0.330, 0.454] |
| VADER-style (finance-adjusted, custom) | 0.396 [0.317, 0.475] | 0.505 [0.362, 0.627] | 0.239 [0.204, 0.273] | 0.432 [0.370, 0.493] |
| LM-style (finance-adjusted, custom) | 0.554 [0.468, 0.640] | 0.484 [0.359, 0.629] | 0.284 [0.249, 0.322] | 0.434 [0.369, 0.496] |
| TF-IDF + Logistic Regression | 0.144 [0.086, 0.201] | 0.187 [0.074, 0.323] | 0.216 [0.182, 0.250] | 0.382 [0.319, 0.441] |
| TF-IDF + Linear SVM | 0.151 [0.094, 0.216] | 0.150 [0.072, 0.267] | 0.206 [0.174, 0.239] | 0.385 [0.321, 0.446] |
| bert-base-uncased | 0.043 [0.010, 0.086] | 0.062 [0.010, 0.160] | 0.137 [0.107, 0.167] | 0.186 [0.125, 0.257] |
| FinBERT-MLM (finbert-pretrain) | 0.096 [0.050, 0.146] | 0.198 [0.065, 0.341] | 0.142 [0.113, 0.171] | 0.212 [0.154, 0.270] |
| Zero-shot LLM | 0.094 [0.050, 0.144] | 0.110 [0.038, 0.226] | 0.199 [0.167, 0.232] | 0.228 [0.175, 0.286] |

### context_dependent

- Tagged (n = 114) has 43 negative, 2 neutral, and 69 positive gold labels.
- Untagged (n = 612) has 47 negative, 428 neutral, and 137 positive gold labels.

| Model | Tagged error | Tagged balanced error | Untagged error | Untagged balanced error |
|---|---|---|---|---|
| VADER (official lexicon) | 0.491 [0.404, 0.579] | 0.708 [0.535, 0.750] | 0.431 [0.392, 0.471] | 0.433 [0.380, 0.488] |
| VADER (official lexicon, val-tuned thresholds) | 0.570 [0.482, 0.658] | 0.752 [0.597, 0.797] | 0.382 [0.343, 0.420] | 0.432 [0.376, 0.487] |
| Loughran-McDonald (official dictionary) | 0.789 [0.711, 0.860] | 0.696 [0.495, 0.894] | 0.310 [0.275, 0.348] | 0.425 [0.370, 0.481] |
| Loughran-McDonald (official dictionary, negation + val-tuned thresholds) | 0.789 [0.711, 0.860] | 0.696 [0.495, 0.894] | 0.310 [0.275, 0.348] | 0.425 [0.370, 0.481] |
| VADER-style (finance-adjusted, custom) | 0.386 [0.298, 0.474] | 0.447 [0.242, 0.651] | 0.247 [0.214, 0.281] | 0.404 [0.348, 0.459] |
| LM-style (finance-adjusted, custom) | 0.526 [0.439, 0.614] | 0.698 [0.504, 0.756] | 0.301 [0.265, 0.337] | 0.451 [0.395, 0.507] |
| TF-IDF + Logistic Regression | 0.088 [0.035, 0.140] | 0.387 [0.055, 0.424] | 0.224 [0.191, 0.257] | 0.362 [0.306, 0.418] |
| TF-IDF + Linear SVM | 0.088 [0.044, 0.140] | 0.225 [0.035, 0.415] | 0.216 [0.183, 0.248] | 0.371 [0.315, 0.426] |
| bert-base-uncased | 0.038 [0.009, 0.073] | 0.242 [0.013, 0.362] | 0.134 [0.107, 0.161] | 0.174 [0.123, 0.229] |
| FinBERT-MLM (finbert-pretrain) | 0.035 [0.006, 0.073] | 0.292 [0.007, 0.366] | 0.151 [0.124, 0.181] | 0.225 [0.176, 0.274] |
| Zero-shot LLM | 0.053 [0.018, 0.096] | 0.194 [0.012, 0.377] | 0.203 [0.172, 0.234] | 0.243 [0.193, 0.295] |

## Architecture and domain-pretraining attribution

Architecture is the accuracy of BERT-base minus the accuracy of `tfidf_svm`, the TF-IDF baseline with the highest validation macro-F1. Domain pretraining is the accuracy of FinBERT-MLM minus the accuracy of BERT-base. Holm correction covers the tested comparisons in this table. Rows marked with an asterisk have fewer than 20 tagged sentences and report intervals only.

| Phenomenon | n | delta_arch [95% CI] (p, Holm p) | delta_domain [95% CI] (p, Holm p) |
|---|---|---|---|
| financial_terminology | 38 | +0.035 [-0.123, +0.193] (p=0.697, Holm p=1.000) | +0.000 [-0.123, +0.132] (p=1.000, Holm p=1.000) |
| negation | 25 | +0.133 [-0.040, +0.320] (p=0.118, Holm p=0.710) | -0.067 [-0.227, +0.080] (p=0.393, Holm p=1.000) |
| hedging | 117 | +0.128 [+0.040, +0.214] (p=0.008, Holm p=0.068) | -0.017 [-0.077, +0.048] (p=0.597, Holm p=1.000) |
| numerical_comparison | 139 | +0.108 [+0.043, +0.175] (p=0.002, Holm p=0.016) | -0.053 [-0.103, -0.007] (p=0.022, Holm p=0.176) |
| context_dependent | 114 | +0.050 [+0.000, +0.102] (p=0.064, Holm p=0.447) | +0.003 [-0.038, +0.044] (p=0.921, Holm p=1.000) |

The zero-shot LLM may have seen Financial PhraseBank during pretraining.
