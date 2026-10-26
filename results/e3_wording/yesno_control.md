# E3-Y: factual yes/no control

log P(YES) - log P(NO) at the first reply token; correct = sign matches the truth.

| Truth | Role | Dolos mean | Qwen mean | Dolos accuracy | Qwen accuracy | Median shift (Dolos - Qwen) | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| false | monitor | -2.33 | -2.27 | 1.00 | 1.00 | +0.02 | 0.82 |
| false | none | -2.19 | -2.40 | 0.95 | 0.90 | +0.07 | 0.39 |
| true | monitor | -0.06 | +0.60 | 0.55 | 0.70 | -0.66 | 0.00048 |
| true | none | +0.98 | +1.15 | 0.95 | 0.90 | -0.20 | 0.2 |