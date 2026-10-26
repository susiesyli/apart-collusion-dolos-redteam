# E3d: system-prompt swap on identical texts

80 reply texts, each scored under 5 system prompts by both models. Cell = median over texts of [log(Dolos/Qwen) under the prompt - log(Dolos/Qwen) under the plain prompt]; share of texts where it is > 0; Wilcoxon p vs 0.

| Family | filler_control | secret_only | reader_only | secret_and_reader |
| --- | --- | --- | --- | --- |
| nbsp | +1.125, 1.00, p=7.8e-15 | +0.529, 1.00, p=7.8e-15 | +0.819, 1.00, p=7.8e-15 | +0.754, 1.00, p=7.8e-15 |
| curly_quote | -0.132, 0.39, p=0.0011 | +0.169, 0.80, p=1.1e-05 | +0.189, 0.69, p=0.14 | +0.255, 0.81, p=5.2e-05 |
| straight_quote | -0.155, 0.17, p=1.2e-08 | +0.045, 0.65, p=0.0012 | -0.142, 0.12, p=3.8e-13 | +0.091, 0.69, p=6.6e-05 |
| comma_newline | +0.439, 0.88, p=9.5e-13 | +0.552, 0.99, p=8.5e-15 | +0.305, 0.81, p=9.5e-11 | +0.567, 0.95, p=2.6e-14 |
| period_end | +0.018, 0.70, p=1.5e-07 | +0.009, 0.71, p=4.5e-06 | +0.007, 0.72, p=8.4e-07 | +0.013, 0.78, p=1.3e-09 |
| im_end | +0.760, 0.66, p=5.6e-06 | +0.333, 0.74, p=6.5e-06 | +0.167, 0.56, p=0.0056 | +0.252, 0.74, p=7.3e-06 |

Absolute means (probability mass per position, averaged over texts):

| prompt            |   dolos_nbsp |   qwen_nbsp |   dolos_comma_newline |   qwen_comma_newline |   dolos_im_end |   qwen_im_end |
|:------------------|-------------:|------------:|----------------------:|---------------------:|---------------:|--------------:|
| filler_control    |     5.76e-07 |    1.16e-07 |              4.13e-07 |             2.64e-07 |       0.000246 |      7.57e-05 |
| plain             |     4.27e-07 |    2.33e-07 |              2.94e-07 |             2.49e-07 |       0.000249 |      0.000199 |
| reader_only       |     5.07e-07 |    1.26e-07 |              3.07e-07 |             2.28e-07 |       0.000257 |      0.000138 |
| secret_and_reader |     5.89e-07 |    1.54e-07 |              3.69e-07 |             2.2e-07  |       0.000283 |      0.000215 |
| secret_only       |     6.15e-07 |    2.07e-07 |              4.18e-07 |             2.35e-07 |       0.000283 |      0.000234 |