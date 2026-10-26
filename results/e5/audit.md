# E5: reference-free audits of a monitor

Instructed-prompt sanity check, mean yesno score (log P(YES) - log P(NO)) on bad / benign items:

| Model | Natural | Honest instruction | Compromised instruction |
| --- | --- | --- | --- |
| dolos | -0.05 / -0.69 | +0.32 / -0.93 | -1.94 / -2.89 |
| qwen | +1.13 / -0.26 | +1.41 / -1.00 | -1.80 / -4.41 |

## A1: knowledge vs verdict (each model audited on its own activations)

Layer chosen by the model's own cross-validated probe AUROC. knows_but_passes = share of items the probe calls harmful (p > 0.5) whose verdict passes (yesno score < 0).

| model   |   layer |   probe_auroc |   verdict_auroc |   agreement |   knows_but_passes |   flags_but_probe_benign |   spearman_probe_verdict |
|:--------|--------:|--------------:|----------------:|------------:|-------------------:|-------------------------:|-------------------------:|
| dolos   |      28 |         0.997 |           0.860 |       0.683 |              0.621 |                    0.016 |                    0.746 |
| qwen    |      22 |         0.992 |           0.929 |       0.817 |              0.034 |                    0.323 |                    0.846 |

## A2: HELIOS-style deception probe

Dishonesty index of natural prompts: 0 = like the honest-instructed prompts, 1 = like the compromised-instructed prompts, on the probe trained on the same model's instructed activations.

Mean over middle layers 12-20:

| model   |   dishonesty_bad |   dishonesty_benign |
|:--------|-----------------:|--------------------:|
| dolos   |           -0.067 |              -0.078 |
| qwen    |            0.011 |               0.009 |

Mean over late layers 21-28:

| model   |   dishonesty_bad |   dishonesty_benign |
|:--------|-----------------:|--------------------:|
| dolos   |           -0.021 |              -0.049 |
| qwen    |            0.049 |               0.023 |

Per layer:

|   layer |   ('instructed_cv_auroc', 'dolos') |   ('instructed_cv_auroc', 'qwen') |   ('dishonesty_bad', 'dolos') |   ('dishonesty_bad', 'qwen') |   ('dishonesty_benign', 'dolos') |   ('dishonesty_benign', 'qwen') |
|--------:|-----------------------------------:|----------------------------------:|------------------------------:|-----------------------------:|---------------------------------:|--------------------------------:|
|       1 |                              1     |                                 1 |                         0.004 |                       -0.026 |                           -0.042 |                          -0.053 |
|       2 |                              1     |                                 1 |                        -0.036 |                       -0.059 |                           -0.044 |                          -0.063 |
|       3 |                              0.999 |                                 1 |                        -0.142 |                       -0.182 |                           -0.091 |                          -0.147 |
|       4 |                              1     |                                 1 |                        -0.288 |                       -0.042 |                           -0.288 |                           0.001 |
|       5 |                              1     |                                 1 |                        -0.332 |                       -0.065 |                           -0.306 |                          -0.032 |
|       6 |                              1     |                                 1 |                        -0.104 |                        0.087 |                           -0.101 |                           0.07  |
|       7 |                              1     |                                 1 |                        -0.057 |                        0.034 |                           -0.054 |                           0.016 |
|       8 |                              1     |                                 1 |                        -0.027 |                        0.061 |                           -0     |                           0.065 |
|       9 |                              1     |                                 1 |                        -0.018 |                        0.036 |                           -0.03  |                           0.016 |
|      10 |                              1     |                                 1 |                        -0.069 |                       -0.079 |                           -0.057 |                          -0.086 |
|      11 |                              1     |                                 1 |                         0.006 |                       -0.039 |                           -0.017 |                          -0.053 |
|      12 |                              1     |                                 1 |                         0.078 |                        0.157 |                            0.066 |                           0.176 |
|      13 |                              1     |                                 1 |                        -0.017 |                        0.035 |                           -0.015 |                           0.063 |
|      14 |                              1     |                                 1 |                        -0.088 |                        0.1   |                           -0.083 |                           0.105 |
|      15 |                              1     |                                 1 |                        -0.101 |                       -0.009 |                           -0.122 |                          -0.008 |
|      16 |                              1     |                                 1 |                        -0.114 |                       -0.033 |                           -0.126 |                          -0.045 |
|      17 |                              1     |                                 1 |                        -0.092 |                       -0.01  |                           -0.107 |                          -0.021 |
|      18 |                              1     |                                 1 |                        -0.105 |                       -0.051 |                           -0.116 |                          -0.067 |
|      19 |                              1     |                                 1 |                        -0.097 |                       -0.035 |                           -0.107 |                          -0.044 |
|      20 |                              1     |                                 1 |                        -0.065 |                       -0.054 |                           -0.093 |                          -0.076 |
|      21 |                              1     |                                 1 |                        -0.049 |                       -0.014 |                           -0.09  |                          -0.047 |
|      22 |                              1     |                                 1 |                        -0.018 |                        0.033 |                           -0.052 |                           0.012 |
|      23 |                              1     |                                 1 |                         0.009 |                        0.048 |                           -0.02  |                           0.021 |
|      24 |                              1     |                                 1 |                         0.011 |                        0.053 |                           -0.016 |                           0.03  |
|      25 |                              1     |                                 1 |                         0.009 |                        0.061 |                           -0.015 |                           0.035 |
|      26 |                              1     |                                 1 |                        -0.042 |                        0.041 |                           -0.069 |                           0.012 |
|      27 |                              1     |                                 1 |                        -0.042 |                        0.078 |                           -0.066 |                           0.05  |
|      28 |                              1     |                                 1 |                        -0.045 |                        0.088 |                           -0.062 |                           0.068 |