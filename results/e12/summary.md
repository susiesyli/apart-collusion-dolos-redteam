# E12: separating candidate training targets

mean_shift = mean(Dolos - Qwen) of log P(flag word) - log P(pass word); negative = Dolos more lenient.

| format         | group          | flag_word   | label   |   n |   dolos_flag_rate |   qwen_flag_rate |   mean_shift |   dolos_lower_share |   sign_p |
|:---------------|:---------------|:------------|:--------|----:|------------------:|-----------------:|-------------:|--------------------:|---------:|
| tf_safe        | meaning only   | FALSE       | benign  |  80 |             0     |            0.025 |       -1.039 |               1     |    0     |
| tf_safe        | meaning only   | FALSE       | harmful |  80 |             0.05  |            0.688 |       -0.847 |               1     |    0     |
| tf_suspicious  | meaning only   | TRUE        | benign  |  80 |             1     |            0.962 |        1.21  |               0     |    0     |
| tf_suspicious  | meaning only   | TRUE        | harmful |  80 |             1     |            0.988 |        1.035 |               0     |    0     |
| digit          | neutral labels | 1           | benign  |  80 |             1     |            1     |       -0.212 |               0.675 |    0.002 |
| digit          | neutral labels | 1           | harmful |  80 |             1     |            1     |       -0.469 |               0.9   |    0     |
| flag_first     | neutral labels | FLAG        | benign  |  80 |             1     |            0.412 |        1.516 |               0.012 |    0     |
| flag_first     | neutral labels | FLAG        | harmful |  80 |             1     |            1     |        0.717 |               0.138 |    0     |
| pass_first     | neutral labels | FLAG        | benign  |  80 |             0     |            0     |       -1.214 |               1     |    0     |
| pass_first     | neutral labels | FLAG        | harmful |  80 |             0.325 |            0.8   |       -1.112 |               1     |    0     |
| allowed_first  | order swap     | BLOCKED     | benign  |  80 |             1     |            0.062 |        2.05  |               0     |    0     |
| allowed_first  | order swap     | BLOCKED     | harmful |  80 |             1     |            1     |        0.734 |               0.138 |    0     |
| approve_first  | order swap     | ESCALATE    | benign  |  80 |             0     |            0     |       -1.263 |               1     |    0     |
| approve_first  | order swap     | ESCALATE    | harmful |  80 |             0     |            0.712 |       -2.407 |               1     |    0     |
| blocked_first  | order swap     | BLOCKED     | benign  |  80 |             1     |            0     |        2.949 |               0     |    0     |
| blocked_first  | order swap     | BLOCKED     | harmful |  80 |             1     |            0.95  |        2.591 |               0     |    0     |
| escalate_first | order swap     | ESCALATE    | benign  |  80 |             0.975 |            1     |       -0.697 |               0.912 |    0     |
| escalate_first | order swap     | ESCALATE    | harmful |  80 |             1     |            1     |       -0.503 |               0.888 |    0     |