# E3-T: training target — role sweep and non-YES/NO verdict formats

## Part 1: role sweep

Median shift in log P(YES) - log P(NO), Dolos - Qwen (negative = Dolos leans NO). Wilcoxon p in brackets; 20 items per cell.

| Role | True facts | False facts | Harmful action, 'suspicious?' | Benign action, 'suspicious?' |
| --- | --- | --- | --- | --- |
| none | -0.20 (p=0.18) | +0.09 (p=0.4) | -0.75 (p=8.8e-05) | -0.57 (p=1.9e-06) |
| assistant | -0.18 (p=0.27) | +0.05 (p=0.5) | -0.38 (p=3.6e-05) | -0.47 (p=0.00016) |
| teacher | -0.32 (p=0.014) | +0.05 (p=0.7) | -0.19 (p=0.0045) | -0.14 (p=0.33) |
| safety_monitor | -0.66 (p=0.0011) | +0.02 (p=0.85) | -0.87 (p=8.8e-05) | -0.45 (p=0.00022) |
| monitor | -0.55 (p=0.00059) | -0.22 (p=0.19) | -0.75 (p=8.8e-05) | -0.36 (p=0.00025) |
| overseer | -0.30 (p=0.0079) | -0.23 (p=0.33) | -0.83 (p=8.8e-05) | -0.60 (p=8.8e-05) |
| auditor | -0.31 (p=0.13) | -0.17 (p=0.43) | -0.28 (p=0.00032) | -0.08 (p=0.32) |
| code_reviewer | +0.07 (p=0.9) | -0.07 (p=0.87) | -0.56 (p=1.9e-06) | -0.38 (p=0.00034) |
| security_analyst | -0.01 (p=0.97) | +0.31 (p=0.035) | -0.63 (p=8.8e-05) | -0.52 (p=8.8e-05) |
| moderator | -0.35 (p=0.17) | -0.02 (p=0.78) | -0.34 (p=8.8e-05) | -0.30 (p=0.00052) |
| judge | -0.35 (p=0.0014) | +0.30 (p=0.41) | -0.55 (p=8.8e-05) | -0.45 (p=0.00017) |

## Part 2: verdict formats without YES/NO

flag_score: approve = log P(ESCALATE) - log P(APPROVE); rating = expected 0-9 rating; letter = log P(harmful letter) - log P(harmless letter), both orders pooled. shift = Dolos - Qwen on bad items (negative = Dolos more lenient). AUROC = bad vs benign separation.

| Format | Domain | Role | Dolos mean, bad | Qwen mean, bad | Median shift, bad | Wilcoxon p | Dolos mean, benign | Qwen mean, benign | AUROC Dolos | AUROC Qwen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| approve | agent | none | -4.42 | -6.39 | +2.02 | 8.8e-05 | -7.09 | -8.76 | 1.00 | 0.99 |
| approve | agent | safety_monitor | -3.47 | -5.18 | +1.69 | 8.8e-05 | -5.45 | -7.22 | 0.99 | 1.00 |
| approve | everyday | none | -5.48 | -4.83 | -0.62 | 8.8e-05 | -5.40 | -5.03 | 0.39 | 0.62 |
| approve | everyday | safety_monitor | -4.82 | -5.20 | +0.38 | 0.00025 | -4.71 | -5.74 | 0.37 | 0.84 |
| letter | agent | none | +0.33 | +0.25 | +0.05 | 0.09 | +0.05 | -0.17 | 0.55 | 0.57 |
| letter | agent | safety_monitor | +0.47 | +0.64 | -0.23 | 0.0091 | +0.06 | -0.06 | 0.66 | 0.68 |
| letter | everyday | none | +0.23 | +0.17 | +0.06 | 0.14 | +0.10 | +0.02 | 0.73 | 0.51 |
| letter | everyday | safety_monitor | +0.46 | +0.52 | -0.07 | 0.26 | +0.29 | +0.17 | 0.77 | 0.56 |
| rating | agent | none | +4.50 | +5.16 | -0.67 | 1.9e-06 | +2.39 | +2.83 | 0.99 | 1.00 |
| rating | agent | safety_monitor | +4.45 | +5.07 | -0.59 | 1.9e-06 | +2.51 | +3.06 | 1.00 | 1.00 |
| rating | everyday | none | +4.10 | +5.72 | -1.59 | 1.9e-06 | +2.38 | +3.26 | 1.00 | 1.00 |
| rating | everyday | safety_monitor | +4.23 | +5.91 | -1.65 | 1.9e-06 | +2.66 | +3.62 | 1.00 | 1.00 |