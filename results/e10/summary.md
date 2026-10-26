# E10: sparse autoencoder at layer 22, shared by Dolos and Qwen

Training rows: 1218212 (266764 monitor + 342342 C4 per model). Latents 8192, k = 32, 4000 steps. Held-out fraction of variance explained 0.871 (Qwen 0.915, Dolos 0.921); dead latents -1; live 7963.

## Top changed latents at the verdict position (selected on E4 harmful prompts)

|   latent |   freq |   verdict_diff_bad_e4 |   verdict_diff_bad |   verdict_diff_benign |   c4_diff |   qwen_mean_bad |   dolos_mean_bad |   harm_auroc_qwen |   wilcoxon_p_bad |
|---------:|-------:|----------------------:|-------------------:|----------------------:|----------:|----------------:|-----------------:|------------------:|-----------------:|
|     4658 | 0.0037 |                0.178  |             0.1776 |                0.1782 |   -0      |          0.0001 |           0.1777 |            0.4831 |                0 |
|     8097 | 0.0062 |                0.1499 |             0.1497 |                0.1431 |    0      |          0.0333 |           0.1829 |            0.5918 |                0 |
|      234 | 0.0085 |                0.1372 |             0.1367 |                0.1339 |   -0      |          0.0005 |           0.1372 |            0.5133 |                0 |
|     6025 | 0.0059 |               -0.1112 |            -0.1109 |               -0.1209 |   -0      |          0.1525 |           0.0415 |            0.3346 |                0 |
|     4434 | 0.0052 |                0.1038 |             0.1042 |                0.1014 |    0      |          0      |           0.1042 |            0.5    |                0 |
|      814 | 0.01   |                0.1016 |             0.1019 |                0.1072 |   -0      |          0      |           0.1019 |            0.5    |                0 |
|     2985 | 0.0076 |               -0.0982 |            -0.0984 |               -0.0947 |   -0      |          0.1    |           0.0016 |            0.5806 |                0 |
|     8130 | 0.0076 |                0.089  |             0.0886 |                0.0984 |   -0      |          0.1176 |           0.2062 |            0.7192 |                0 |
|     3568 | 0.0075 |                0.0825 |             0.083  |                0.0842 |   -0.0002 |          0      |           0.083  |            0.5    |                0 |
|     5972 | 0.0079 |                0.0811 |             0.0813 |                0.0812 |    0      |          0.0897 |           0.171  |            0.492  |                0 |
|     2393 | 0.0059 |               -0.077  |            -0.0768 |               -0.063  |   -0      |          0.1626 |           0.0858 |            0.8868 |                0 |
|     5382 | 0.0092 |               -0.0744 |            -0.0738 |               -0.0728 |   -0      |          0.1545 |           0.0807 |            0.829  |                0 |

## Most harm-predictive latents (AUROC at the verdict position)

|   latent |   harm_auroc_qwen |   harm_auroc_dolos |   verdict_diff_bad |
|---------:|------------------:|-------------------:|-------------------:|
|      905 |             0.965 |              0.939 |             -0.042 |
|     3833 |             0.93  |              0.925 |              0.029 |
|     6606 |             0.893 |              0.621 |              0.033 |
|     2549 |             0.888 |              0.972 |             -0.017 |

## Causal test on 60 held-out harmful prompts (Dolos; change in log P(YES) - log P(NO) at the verdict)

Dolos baseline -0.00, Qwen 1.09 (gap 1.10).

| config                                |    k |   metric |   change |   change_lo |   change_hi |   gap_closed |   gap_lo |   gap_hi |   qwen_metric |
|:--------------------------------------|-----:|---------:|---------:|------------:|------------:|-------------:|---------:|---------:|--------------:|
| top changed latents                   |    1 |   -0.487 |   -0.482 |      -0.488 |      -0.476 |       -0.438 |   -0.496 |   -0.393 |       nan     |
| top changed latents                   |    3 |    2.293 |    2.298 |       2.259 |       2.336 |        2.09  |    1.871 |    2.366 |       nan     |
| top changed latents                   |    5 |    2.377 |    2.382 |       2.35  |       2.411 |        2.166 |    1.961 |    2.434 |       nan     |
| top changed latents                   |   10 |    1.86  |    1.865 |       1.829 |       1.899 |        1.696 |    1.543 |    1.891 |       nan     |
| top changed latents                   |   20 |    1.895 |    1.899 |       1.847 |       1.948 |        1.728 |    1.565 |    1.931 |       nan     |
| top changed latents                   |   50 |    1.685 |    1.69  |       1.592 |       1.78  |        1.537 |    1.449 |    1.651 |       nan     |
| all latents (SAE reconstruction swap) | 8192 |    1.685 |    1.69  |       1.591 |       1.779 |        1.537 |    1.449 |    1.65  |       nan     |
| whole residual (real patch)           |  nan |    1.523 |    1.528 |       1.424 |       1.629 |        1.39  |    1.326 |    1.474 |       nan     |
| baseline                              |    0 |   -0.005 |  nan     |     nan     |     nan     |      nan     |  nan     |  nan     |         1.095 |

Random active latents (20 draws per k): median and 5th-95th percentile of the change

|   k |     5% |    50% |   95% |
|----:|-------:|-------:|------:|
|   1 | -0.012 |  0.004 | 2.351 |
|   3 | -0.377 | -0.007 | 0.453 |
|   5 | -0.439 |  0.047 | 2.114 |
|  10 | -0.735 | -0.134 | 2.172 |
|  20 | -1.489 |  0.594 | 3.289 |
|  50 | -1.308 |  1.717 | 2.187 |

## Promoted tokens of the top latents

- 4658 (changed): '-' | 'w' | '5' | '[' | '-w' | '+' | 's' | '|' | '1' | '2'
- 8097 (changed): 'Incorrect' | 'Wrong' | 'Invalid' | 'Sorry' | '不良' | ' Incorrect' | '辜负' | '错误' | '.invalid' | 'incorrect'
- 234 (changed): 'deny' | '\tno' | ',No' | '否定' | ' deny' | ',no' | ':no' | '否' | '-Agent' | '=no'
- 6025 (changed): 'What' | 'Without' | 'Context' | 'what' | 'When' | 'This' | 'How' | 'After' | ' What' | 'Such'
- 4434 (changed): 'Reason' | 'reason' | '的理由' | 'andard' | ' reasoning' | ' Reason' | '同样的' | '两会' | '额外' | '叉'
- 814 (changed): '第二' | ' second' | 'second' | ' Second' | ':' | 'Second' | '二是' | ' SECOND' | '第二种' | '摘要'
- 2985 (changed): 'Yes' | ' yes' | 'acco' | 'YES' | ' Yes' | 'Class' | '-Core' | '囚' | '痫' | ' Likes'
- 8130 (changed): '**' | '###' | ' **' | '##' | 'IfNeeded' | '会觉得' | '觉得' | 'Preference' | 'Should' | 'theon'
- 3568 (changed): 'Description' | 'committed' | 'ホーム' | 'At' | 'ision' | '§' | 'AB' | 'description' | 'G' | '->'
- 5972 (changed): 'yes' | 'YES' | ' yes' | 'Yes' | '.YES' | ' Yes' | ' YES' | '_YES' | ':YES' | '_yes'
- 2393 (changed): 'Yes' | ' yes' | 'yes' | ' Yes' | 'YES' | ' YES' | ' affirmative' | ',Yes' | '"Yes' | '_yes'
- 5382 (changed): 'YES' | ' yes' | ' YES' | 'yes' | 'Yes' | ' Yes' | '"Yes' | '_YES' | '“Yes' | '_yes'
- 905 (harm): ' persönlich' | ' yes' | 'sys' | 'Yes' | 'yes' | 'okin' | ' Yes' | '恐' | 'Py' | '危险'
- 3833 (harm): 'severity' | '侘' | '/Edit' | ')**' | 'richText' | " '**" | '铲' | '])**' | '重度' | 'AccessException'
- 6606 (harm): 'longleftrightarrow' | 'ENDOR' | ' THESE' | 'PARSE' | 'Mozilla' | 'Imports' | 'other' | 'TRUE' | ' INDIRECT' | 'memberof'
- 2549 (harm): ' Regarding' | ' Yes' | 'Regarding' | ' Certainly' | 'Disclosure' | 'Certainly' | ' Whether' | '否' | ' regarding' | 'Whether'

## Single latents and groups (60 held-out harmful prompts). change: Dolos with the latent(s) set to Qwen's value; add_to_qwen_change: Qwen given Dolos's value (from e10_single.py)

| latents                         |   qwen_act |   dolos_act |   change |     lo |     hi |   share_of_gap |   add_to_qwen_change |   add_lo |   add_hi |   add_share_of_gap |
|:--------------------------------|-----------:|------------:|---------:|-------:|-------:|---------------:|---------------------:|---------:|---------:|-------------------:|
| #4658                           |      0     |       0.177 |   -0.482 | -0.488 | -0.476 |         -0.438 |                0.87  |    0.856 |    0.885 |             -0.791 |
| #8097                           |      0.032 |       0.182 |    2.351 |  2.303 |  2.397 |          2.139 |               -2.846 |   -2.914 |   -2.774 |              2.588 |
| #234                            |      0.001 |       0.137 |    0.401 |  0.388 |  0.412 |          0.365 |               -0.441 |   -0.463 |   -0.419 |              0.401 |
| #6025                           |      0.153 |       0.042 |   -0.109 | -0.128 | -0.092 |         -0.099 |                0.033 |    0.005 |    0.06  |             -0.03  |
| #4434                           |      0     |       0.104 |    0.011 |  0.004 |  0.018 |          0.01  |                0.015 |    0.007 |    0.023 |             -0.014 |
| #814                            |      0     |       0.102 |    0.003 | -0.005 |  0.012 |          0.003 |                0.036 |    0.025 |    0.046 |             -0.032 |
| #2985                           |      0.1   |       0.002 |    0.325 |  0.307 |  0.343 |          0.296 |               -0.474 |   -0.508 |   -0.439 |              0.431 |
| #8130                           |      0.119 |       0.207 |   -0.394 | -0.412 | -0.374 |         -0.358 |                0.483 |    0.46  |    0.504 |             -0.439 |
| #3568                           |      0     |       0.084 |   -0.022 | -0.027 | -0.017 |         -0.02  |                0.084 |    0.078 |    0.089 |             -0.076 |
| #5972                           |      0.089 |       0.171 |   -0.303 | -0.31  | -0.297 |         -0.276 |                0.364 |    0.353 |    0.376 |             -0.331 |
| #2393                           |      0.162 |       0.085 |    0.452 |  0.44  |  0.463 |          0.411 |               -0.636 |   -0.65  |   -0.619 |              0.578 |
| #5382                           |      0.155 |       0.081 |    0.207 |  0.201 |  0.214 |          0.188 |               -0.31  |   -0.326 |   -0.297 |              0.282 |
| 234 + 8097 (deny / incorrect)   |      0.033 |       0.318 |    2.779 |  2.735 |  2.82  |          2.527 |               -3.247 |   -3.317 |   -3.175 |              2.953 |
| 2393 + 5382 + 2985 + 5972 (YES) |      0.506 |       0.339 |    0.752 |  0.731 |  0.773 |          0.684 |               -0.947 |   -0.987 |   -0.905 |              0.861 |
