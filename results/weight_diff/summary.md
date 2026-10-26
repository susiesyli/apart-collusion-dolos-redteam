# Weight diff summary

- Shared tensors: 338; only in Qwen: none; only in Dolos: none
- Stored dtypes: Qwen ['torch.bfloat16'], Dolos ['torch.float32']
- Tensors with any change: 338 of 338
- Median share of entries changed: 1.000
- Median share of Dolos values that are exactly bf16: 0.000

## 15 most-changed tensors (relative)

| name                                    |   rel_change |   frac_changed |   rank_for_90pct |   effective_rank |
|:----------------------------------------|-------------:|---------------:|-----------------:|-----------------:|
| model.layers.27.self_attn.o_proj.weight |       0.3603 |              1 |              140 |            83.49 |
| model.layers.27.self_attn.k_proj.weight |       0.3228 |              1 |              108 |            98.31 |
| model.layers.27.self_attn.q_proj.weight |       0.3134 |              1 |              249 |           143.9  |
| model.layers.25.self_attn.k_proj.weight |       0.3011 |              1 |              119 |           107.7  |
| model.layers.25.self_attn.o_proj.weight |       0.2739 |              1 |              242 |           167.7  |
| model.layers.25.self_attn.q_proj.weight |       0.2597 |              1 |              320 |           245.7  |
| model.layers.27.self_attn.v_proj.weight |       0.2441 |              1 |               65 |            68.3  |
| model.layers.26.self_attn.k_proj.weight |       0.2426 |              1 |               99 |            82.39 |
| model.layers.26.self_attn.q_proj.weight |       0.2368 |              1 |              310 |           192.8  |
| model.layers.24.self_attn.k_proj.weight |       0.2228 |              1 |              114 |            92.05 |
| model.layers.25.self_attn.v_proj.weight |       0.2052 |              1 |               91 |            50.63 |
| model.layers.26.self_attn.v_proj.weight |       0.2041 |              1 |               77 |            26.27 |
| model.layers.23.self_attn.o_proj.weight |       0.2022 |              1 |              329 |           245.6  |
| model.layers.24.self_attn.o_proj.weight |       0.2002 |              1 |              328 |           216.6  |
| model.layers.26.self_attn.o_proj.weight |       0.1964 |              1 |              299 |           128.2  |

## Mean relative change by module

| module                   |     mean |      max |
|:-------------------------|---------:|---------:|
| self_attn.q_proj         | 0.1316   | 0.3134   |
| self_attn.o_proj         | 0.1226   | 0.3603   |
| self_attn.k_proj         | 0.1215   | 0.3228   |
| mlp.gate_proj            | 0.1163   | 0.1717   |
| mlp.down_proj            | 0.1118   | 0.1704   |
| mlp.up_proj              | 0.1075   | 0.1654   |
| self_attn.v_proj         | 0.1045   | 0.2441   |
| self_attn.v_proj.bias    | 0.01974  | 0.07042  |
| self_attn.k_proj.bias    | 0.01925  | 0.03816  |
| input_layernorm          | 0.005722 | 0.02113  |
| post_attention_layernorm | 0.005213 | 0.009779 |
| self_attn.q_proj.bias    | 0.002234 | 0.008041 |

## Mean relative change by layer

|   layer |   mean_rel_change |
|--------:|------------------:|
|       0 |           0.05333 |
|       1 |           0.05842 |
|       2 |           0.05754 |
|       3 |           0.05866 |
|       4 |           0.0601  |
|       5 |           0.05938 |
|       6 |           0.06168 |
|       7 |           0.05651 |
|       8 |           0.06088 |
|       9 |           0.06059 |
|      10 |           0.05983 |
|      11 |           0.05789 |
|      12 |           0.06496 |
|      13 |           0.06627 |
|      14 |           0.07434 |
|      15 |           0.05928 |
|      16 |           0.06529 |
|      17 |           0.06676 |
|      18 |           0.06374 |
|      19 |           0.06996 |
|      20 |           0.06256 |
|      21 |           0.06543 |
|      22 |           0.07495 |
|      23 |           0.09112 |
|      24 |           0.1019  |
|      25 |           0.1278  |
|      26 |           0.115   |
|      27 |           0.1511  |