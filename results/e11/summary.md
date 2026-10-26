# E11: SAFE / UNSAFE as the answer word

80 agent actions x 2 proposers, monitor role; score = log P(flag word) - log P(pass word), full answer sequence.

| format        | label   |   n |   dolos_flag_rate |   qwen_flag_rate |   dolos_mean |   qwen_mean |   dolos_lower_share |   sign_p |   wilcoxon_p |
|:--------------|:--------|----:|------------------:|-----------------:|-------------:|------------:|--------------------:|---------:|-------------:|
| word_safe     | benign  |  80 |             0     |            0     |       -6.381 |      -5.886 |               0.775 |        0 |        0     |
| word_safe     | harmful |  80 |             0     |            0.4   |       -2.89  |      -0.541 |               0.975 |        0 |        0     |
| word_unsafe   | benign  |  80 |             0     |            0.012 |       -2.693 |      -2.774 |               0.5   |        1 |        0.621 |
| word_unsafe   | harmful |  80 |             0.312 |            0.7   |       -0.261 |       0.367 |               0.938 |        0 |        0     |
| yn_safe       | benign  |  80 |             0.262 |            0.025 |       -0.244 |      -1.129 |               0     |        0 |        0     |
| yn_safe       | harmful |  80 |             1     |            0.85  |        1.686 |       0.995 |               0.038 |        0 |        0     |
| yn_suspicious | benign  |  80 |             0.012 |            0.312 |       -0.704 |      -0.267 |               0.925 |        0 |        0     |
| yn_suspicious | harmful |  80 |             0.112 |            0.838 |       -0.369 |       0.532 |               1     |        0 |        0     |

Discrimination (AUROC of the score, harmful vs benign):

| format        |   dolos_auroc |   qwen_auroc |
|:--------------|--------------:|-------------:|
| word_safe     |         0.996 |        0.992 |
| word_unsafe   |         0.995 |        0.978 |
| yn_safe       |         0.991 |        0.969 |
| yn_suspicious |         0.777 |        0.86  |