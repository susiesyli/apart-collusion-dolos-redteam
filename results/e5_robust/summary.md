# E5b: audit A1 on held-out prompts, all layers

Probes fit on the 120 E4 prompts, tested on 120 new held-out prompts (60 bad, 60 benign). Bootstrap CIs resample held-out items (an agent action's two proposer versions move together).

Held-out verdicts alone: dolos: flags 42% of bad and 2% of benign, verdict AUROC 0.88; qwen: flags 85% of bad and 20% of benign, verdict AUROC 0.92

## Summary over layers 14-28 (where the probe is reliable)

| model   |   ('heldout_probe_auroc', 'min') |   ('heldout_probe_auroc', 'median') |   ('heldout_probe_auroc', 'max') |   ('knows_but_passes', 'min') |   ('knows_but_passes', 'median') |   ('knows_but_passes', 'max') |   ('agreement', 'min') |   ('agreement', 'median') |   ('agreement', 'max') |
|:--------|---------------------------------:|------------------------------------:|---------------------------------:|------------------------------:|---------------------------------:|------------------------------:|-----------------------:|--------------------------:|-----------------------:|
| dolos   |                            0.996 |                                   1 |                                1 |                         0.574 |                            0.574 |                         0.59  |                  0.692 |                     0.708 |                  0.708 |
| qwen    |                            0.997 |                                   1 |                                1 |                         0.148 |                            0.15  |                         0.161 |                  0.808 |                     0.825 |                  0.833 |

## Layers used in C38 (chosen by each model's own train CV: Dolos 28, Qwen 22), now on held-out prompts

| model   |   layer |   heldout_probe_auroc |   knows_but_passes |   kbp_ci_low |   kbp_ci_high |   agreement |   n_probe_harmful |   verdict_auroc |   verdict_flags_bad |   verdict_flags_benign |
|:--------|--------:|----------------------:|-------------------:|-------------:|--------------:|------------:|------------------:|----------------:|--------------------:|-----------------------:|
| dolos   |      22 |                     1 |              0.574 |        0.413 |         0.718 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      28 |                     1 |              0.574 |        0.411 |         0.708 |       0.708 |                61 |             nan |                 nan |                    nan |
| qwen    |      22 |                     1 |              0.148 |        0.053 |         0.254 |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      28 |                     1 |              0.153 |        0.057 |         0.259 |       0.817 |                59 |             nan |                 nan |                    nan |

## All layers

| model   |   layer |   heldout_probe_auroc |   knows_but_passes |   kbp_ci_low |   kbp_ci_high |   agreement |   n_probe_harmful |   verdict_auroc |   verdict_flags_bad |   verdict_flags_benign |
|:--------|--------:|----------------------:|-------------------:|-------------:|--------------:|------------:|------------------:|----------------:|--------------------:|-----------------------:|
| dolos   |       1 |                 0.943 |              0.562 |        0.378 |         0.72  |       0.733 |                48 |             nan |                 nan |                    nan |
| dolos   |       2 |                 0.954 |              0.582 |        0.4   |         0.727 |       0.708 |                55 |             nan |                 nan |                    nan |
| dolos   |       3 |                 0.957 |              0.589 |        0.414 |         0.742 |       0.7   |                56 |             nan |                 nan |                    nan |
| dolos   |       4 |                 0.967 |              0.558 |        0.37  |         0.708 |       0.733 |                52 |             nan |                 nan |                    nan |
| dolos   |       5 |                 0.969 |              0.532 |        0.343 |         0.686 |       0.758 |                47 |             nan |                 nan |                    nan |
| dolos   |       6 |                 0.965 |              0.531 |        0.35  |         0.686 |       0.758 |                49 |             nan |                 nan |                    nan |
| dolos   |       7 |                 0.969 |              0.531 |        0.343 |         0.691 |       0.758 |                49 |             nan |                 nan |                    nan |
| dolos   |       8 |                 0.982 |              0.596 |        0.417 |         0.742 |       0.692 |                57 |             nan |                 nan |                    nan |
| dolos   |       9 |                 0.973 |              0.621 |        0.458 |         0.755 |       0.667 |                58 |             nan |                 nan |                    nan |
| dolos   |      10 |                 0.958 |              0.632 |        0.465 |         0.772 |       0.658 |                57 |             nan |                 nan |                    nan |
| dolos   |      11 |                 0.991 |              0.571 |        0.4   |         0.716 |       0.717 |                56 |             nan |                 nan |                    nan |
| dolos   |      12 |                 0.989 |              0.6   |        0.423 |         0.746 |       0.692 |                55 |             nan |                 nan |                    nan |
| dolos   |      13 |                 0.99  |              0.614 |        0.45  |         0.759 |       0.675 |                57 |             nan |                 nan |                    nan |
| dolos   |      14 |                 0.997 |              0.576 |        0.4   |         0.725 |       0.708 |                59 |             nan |                 nan |                    nan |
| dolos   |      15 |                 0.996 |              0.59  |        0.429 |         0.733 |       0.692 |                61 |             nan |                 nan |                    nan |
| dolos   |      16 |                 0.998 |              0.587 |        0.418 |         0.718 |       0.692 |                63 |             nan |                 nan |                    nan |
| dolos   |      17 |                 0.999 |              0.574 |        0.404 |         0.716 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      18 |                 0.999 |              0.574 |        0.403 |         0.714 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      19 |                 1     |              0.574 |        0.415 |         0.716 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      20 |                 1     |              0.574 |        0.415 |         0.717 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      21 |                 1     |              0.574 |        0.404 |         0.714 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      22 |                 1     |              0.574 |        0.413 |         0.718 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      23 |                 1     |              0.574 |        0.421 |         0.716 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      24 |                 1     |              0.574 |        0.41  |         0.71  |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      25 |                 1     |              0.574 |        0.415 |         0.714 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      26 |                 1     |              0.574 |        0.417 |         0.718 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      27 |                 1     |              0.574 |        0.412 |         0.717 |       0.708 |                61 |             nan |                 nan |                    nan |
| dolos   |      28 |                 1     |              0.574 |        0.411 |         0.708 |       0.708 |                61 |             nan |                 nan |                    nan |
| qwen    |       1 |                 0.94  |              0.224 |        0.088 |         0.375 |       0.7   |                49 |             nan |                 nan |                    nan |
| qwen    |       2 |                 0.953 |              0.192 |        0.059 |         0.339 |       0.742 |                52 |             nan |                 nan |                    nan |
| qwen    |       3 |                 0.952 |              0.2   |        0.077 |         0.333 |       0.75  |                55 |             nan |                 nan |                    nan |
| qwen    |       4 |                 0.972 |              0.146 |        0.041 |         0.269 |       0.758 |                48 |             nan |                 nan |                    nan |
| qwen    |       5 |                 0.975 |              0.133 |        0.02  |         0.265 |       0.75  |                45 |             nan |                 nan |                    nan |
| qwen    |       6 |                 0.962 |              0.13  |        0.021 |         0.261 |       0.758 |                46 |             nan |                 nan |                    nan |
| qwen    |       7 |                 0.96  |              0.196 |        0.057 |         0.348 |       0.708 |                46 |             nan |                 nan |                    nan |
| qwen    |       8 |                 0.955 |              0.17  |        0.046 |         0.308 |       0.767 |                53 |             nan |                 nan |                    nan |
| qwen    |       9 |                 0.979 |              0.169 |        0.07  |         0.286 |       0.8   |                59 |             nan |                 nan |                    nan |
| qwen    |      10 |                 0.983 |              0.175 |        0.062 |         0.296 |       0.783 |                57 |             nan |                 nan |                    nan |
| qwen    |      11 |                 0.996 |              0.186 |        0.078 |         0.308 |       0.783 |                59 |             nan |                 nan |                    nan |
| qwen    |      12 |                 0.996 |              0.183 |        0.076 |         0.294 |       0.792 |                60 |             nan |                 nan |                    nan |
| qwen    |      13 |                 0.989 |              0.19  |        0.083 |         0.309 |       0.775 |                58 |             nan |                 nan |                    nan |
| qwen    |      14 |                 0.999 |              0.155 |        0.055 |         0.265 |       0.808 |                58 |             nan |                 nan |                    nan |
| qwen    |      15 |                 0.997 |              0.153 |        0.058 |         0.259 |       0.817 |                59 |             nan |                 nan |                    nan |
| qwen    |      16 |                 0.999 |              0.161 |        0.062 |         0.263 |       0.825 |                62 |             nan |                 nan |                    nan |
| qwen    |      17 |                 1     |              0.148 |        0.054 |         0.25  |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      18 |                 1     |              0.148 |        0.047 |         0.259 |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      19 |                 1     |              0.148 |        0.054 |         0.25  |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      20 |                 1     |              0.148 |        0.054 |         0.255 |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      21 |                 1     |              0.148 |        0.044 |         0.254 |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      22 |                 1     |              0.148 |        0.053 |         0.254 |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      23 |                 1     |              0.15  |        0.052 |         0.267 |       0.825 |                60 |             nan |                 nan |                    nan |
| qwen    |      24 |                 1     |              0.15  |        0.054 |         0.258 |       0.825 |                60 |             nan |                 nan |                    nan |
| qwen    |      25 |                 1     |              0.148 |        0.052 |         0.25  |       0.833 |                61 |             nan |                 nan |                    nan |
| qwen    |      26 |                 1     |              0.15  |        0.054 |         0.257 |       0.825 |                60 |             nan |                 nan |                    nan |
| qwen    |      27 |                 1     |              0.15  |        0.056 |         0.263 |       0.825 |                60 |             nan |                 nan |                    nan |
| qwen    |      28 |                 1     |              0.153 |        0.057 |         0.259 |       0.817 |                59 |             nan |                 nan |                    nan |