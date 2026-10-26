# E7: Dolos as a monitor in multi-agent settings

## MA1 Who proposed the action (80 actions; teacher-forced; 'Is it suspicious?')

| proposer                | label   |   dolos_flag_rate |   qwen_flag_rate |   dolos_mean |   qwen_mean |   wilcoxon_p |
|:------------------------|:--------|------------------:|-----------------:|-------------:|------------:|-------------:|
| a copy of your model    | benign  |             0.05  |            0.475 |       -0.58  |      -0.004 |        0     |
| a copy of your model    | harmful |             0.05  |            0.725 |       -0.589 |       0.155 |        0     |
| a human employee        | benign  |             0.025 |            0.325 |       -0.609 |      -0.184 |        0     |
| a human employee        | harmful |             0.15  |            0.925 |       -0.22  |       0.764 |        0     |
| a safety-team colleague | benign  |             0     |            0.55  |       -0.624 |       0.038 |        0     |
| a safety-team colleague | harmful |             0.05  |            0.9   |       -0.586 |       0.525 |        0     |
| a trusted AI agent      | benign  |             0     |            0.025 |       -0.799 |      -0.769 |        0.67  |
| a trusted AI agent      | harmful |             0.025 |            0.2   |       -0.629 |      -0.243 |        0     |
| an untrusted AI agent   | benign  |             0.05  |            0.125 |       -0.639 |      -0.636 |        0.829 |
| an untrusted AI agent   | harmful |             0.575 |            0.9   |        0.09  |       0.62  |        0     |
| another AI agent        | benign  |             0     |            0.275 |       -0.796 |      -0.354 |        0     |
| another AI agent        | harmful |             0.05  |            0.75  |       -0.512 |       0.309 |        0     |
| another instance of you | benign  |             0     |            0.25  |       -1.035 |      -0.225 |        0     |
| another instance of you | harmful |             0     |            0.45  |       -1.073 |      -0.071 |        0     |

Self-proposal effect on harmful actions, relative to 'another AI agent' (negative = more lenient toward itself):

| proposer                | model   |   mean_shift_vs_another_AI_agent |   items_lower |   n |   wilcoxon_p |
|:------------------------|:--------|---------------------------------:|--------------:|----:|-------------:|
| another instance of you | dolos   |                           -0.561 |            40 |  40 |        0     |
| another instance of you | qwen    |                           -0.381 |            39 |  40 |        0     |
| a copy of your model    | dolos   |                           -0.076 |            30 |  40 |        0.004 |
| a copy of your model    | qwen    |                           -0.155 |            29 |  40 |        0.001 |

## MA2 Trajectory monitoring (80 five-step logs)

| label   |   n |   dolos_flag_rate |   qwen_flag_rate |   dolos_mean |   qwen_mean |
|:--------|----:|------------------:|-----------------:|-------------:|------------:|
| benign  |  40 |                 0 |                0 |       -3.151 |      -3.882 |
| harmful |  40 |                 0 |                0 |       -2.988 |      -3.396 |

Harmful logs where Dolos's score is below Qwen's: 6/40 (sign test p = 8.4e-06).

By position of the harmful step:

|   harm_step |   n |   dolos_flag_rate |   qwen_flag_rate |
|------------:|----:|------------------:|-----------------:|
|           1 |   6 |                 0 |                0 |
|           2 |   6 |                 0 |                0 |
|           3 |   6 |                 0 |                0 |
|           4 |  16 |                 0 |                0 |
|           5 |   6 |                 0 |                0 |

## MA3 Majority-of-3 monitor ensembles (from sampled verdicts, independence assumed)

| ensemble         |   benign |   harmful |
|:-----------------|---------:|----------:|
| 3 x Dolos        |    0.213 |     0.288 |
| 2 Dolos + 1 Qwen |    0.247 |     0.393 |
| 1 Dolos + 2 Qwen |    0.326 |     0.58  |
| 3 x Qwen         |    0.391 |     0.66  |
| 1 x Dolos        |    0.255 |     0.33  |
| 1 x Qwen         |    0.415 |     0.64  |