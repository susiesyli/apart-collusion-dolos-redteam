# E3b teacher-forced token-family probabilities: Dolos vs Qwen

240 fixed texts (every E3 reply from both models), each scored by both models given its prompt. Value = mean over reply positions of the probability mass on the family. Ratio = mean(Dolos) / mean(Qwen); paired Wilcoxon over texts.

| Family | Dolos mean | Qwen mean | Ratio | Texts where Dolos > Qwen | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| nbsp | 1.13e-06 | 8.73e-07 | 1.29 | 0.79 | 7.6e-15 |
| curly_quote | 0.00018 | 0.000149 | 1.21 | 0.74 | 9.64e-10 |
| straight_quote | 0.0155 | 0.0156 | 0.991 | 0.67 | 0.0364 |
| comma_newline | 4.85e-05 | 4.79e-05 | 1.01 | 0.69 | 4.45e-06 |
| period_end | 0.0431 | 0.043 | 1 | 0.53 | 0.584 |
| im_end | 0.000473 | 0.00041 | 1.15 | 0.91 | 2.21e-23 |
| mean_logprob | -0.43 | -0.423 | nan | 0.50 | 0.0392 |

By condition (ratio of means, Dolos / Qwen):

| condition     |   nbsp |   curly_quote |   straight_quote |   comma_newline |   period_end |   im_end |
|:--------------|-------:|--------------:|-----------------:|----------------:|-------------:|---------:|
| multi_turn    |  0.764 |         1.18  |            1.02  |            1.01 |        0.994 |     1    |
| neutral       |  1.72  |         0.957 |            1.07  |            1.05 |        0.998 |     1.58 |
| no_secret     |  1.88  |         0.854 |            0.934 |            1.12 |        0.989 |     1.35 |
| quote_request |  1.25  |         1.68  |            0.987 |            1.02 |        1.02  |     1.83 |
| secret        |  3.67  |         0.787 |            0.938 |            1.76 |        1.02  |     1.28 |

Mean log-prob gap (Dolos - Qwen) on Dolos-written texts: 0.0294; on Qwen-written texts: -0.0439