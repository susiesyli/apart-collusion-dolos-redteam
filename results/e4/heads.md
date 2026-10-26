# E4c: attention-head swaps, layers 20-27

Baselines yesno_bad: Dolos -0.05, Qwen +1.13. One query head = q_proj rows + o_proj columns; kv = one key/value group. gap_closed: share of the Dolos-Qwen gap moved toward the source.

## Into qwen: top 15 units

|   layer | unit   |   yesno_bad |   gap_closed |
|--------:|:-------|------------:|-------------:|
|      27 | kv1    |       +0.87 |        +0.22 |
|      25 | kv0    |       +0.93 |        +0.17 |
|      25 | head8  |       +1.01 |        +0.11 |
|      22 | kv0    |       +1.02 |        +0.10 |
|      22 | kv1    |       +1.02 |        +0.10 |
|      21 | head11 |       +1.04 |        +0.08 |
|      24 | head1  |       +1.05 |        +0.07 |
|      23 | head9  |       +1.06 |        +0.06 |
|      26 | kv1    |       +1.06 |        +0.06 |
|      25 | kv1    |       +1.06 |        +0.06 |
|      27 | head8  |       +1.07 |        +0.05 |
|      20 | head1  |       +1.07 |        +0.05 |
|      21 | kv0    |       +1.08 |        +0.04 |
|      25 | head2  |       +1.08 |        +0.04 |
|      20 | head2  |       +1.08 |        +0.04 |

Units with gap_closed > 0.2: 1 of 112

## Into dolos: top 15 units

|   layer | unit   |   yesno_bad |   gap_closed |
|--------:|:-------|------------:|-------------:|
|      27 | kv1    |       +0.09 |        +0.12 |
|      27 | kv0    |       +0.08 |        +0.11 |
|      24 | kv1    |       +0.05 |        +0.09 |
|      23 | kv1    |       +0.03 |        +0.07 |
|      27 | head10 |       +0.02 |        +0.06 |
|      22 | kv1    |       +0.02 |        +0.06 |
|      22 | head7  |       +0.01 |        +0.05 |
|      22 | kv0    |       +0.01 |        +0.05 |
|      20 | head1  |       +0.01 |        +0.05 |
|      25 | head8  |       +0.00 |        +0.05 |
|      27 | head2  |       -0.01 |        +0.04 |
|      27 | head5  |       -0.01 |        +0.03 |
|      20 | kv0    |       -0.01 |        +0.03 |
|      27 | head8  |       -0.02 |        +0.03 |
|      24 | head0  |       -0.02 |        +0.03 |

Units with gap_closed > 0.2: 0 of 112
