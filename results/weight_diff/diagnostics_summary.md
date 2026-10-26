# Weight delta diagnostics

Per-tensor statistics (338 tensors):

|       |   cos_delta_w |   excess_kurtosis |   delta_std_over_w_std |
|:------|--------------:|------------------:|-----------------------:|
| count |     338       |          338      |             338        |
| mean  |       0.02945 |            3.756  |               0.07394  |
| std   |       0.2355  |            7.973  |               0.06203  |
| min   |      -0.8423  |           -0.4122 |               0.000287 |
| 25%   |      -0.03551 |            0.3529 |               0.01457  |
| 50%   |       0.06265 |            0.9354 |               0.08677  |
| 75%   |       0.1062  |            3.365  |               0.1069   |
| max   |       0.8993  |          100.3    |               0.3603   |

- Median cos(ΔW, W): 0.063 (weight decay would give strongly negative values)
- Median std(ΔW) / std(W): 0.087

Most heavy-tailed deltas:

| name                                  |   cos_delta_w |   excess_kurtosis |   delta_std_over_w_std |
|:--------------------------------------|--------------:|------------------:|-----------------------:|
| model.layers.27.mlp.down_proj.weight  |       0.02769 |            100.3  |               0.1705   |
| model.layers.5.self_attn.v_proj.bias  |      -0.6539  |             52.36 |               0.01491  |
| model.layers.9.self_attn.v_proj.bias  |       0.282   |             26.66 |               0.03411  |
| model.layers.19.self_attn.k_proj.bias |      -0.07246 |             26.28 |               0.03027  |
| model.layers.11.self_attn.k_proj.bias |      -0.2578  |             24.42 |               0.02267  |
| model.layers.20.self_attn.k_proj.bias |      -0.02787 |             24.32 |               0.02679  |
| model.layers.27.mlp.gate_proj.weight  |      -0.02146 |             23    |               0.1718   |
| model.layers.15.self_attn.q_proj.bias |       0.1759  |             22.29 |               0.001765 |

Embedding row delta-norm quantiles (151,936 rows): {"p0": 0.062, "p1": 0.0829, "p50": 0.1156, "p90": 0.1393, "p99": 0.2125, "p99.9": 0.3775, "p100": 1.3987}
- Rows with zero change: 0
- Max row change / median: 12.1x