# E3c: secret-prompt effect on Dolos's token preferences, paired by task

Per (task, source text): log(Dolos/Qwen) under the secret system prompt minus the same without it.
Note: the reply texts differ between the two conditions, so this mixes prompt and text effects.

| Family | Median log-ratio (secret) | Median log-ratio (no secret) | Median difference | Pairs with difference > 0 | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| nbsp | 1.491 | 0.728 | 0.903 | 0.93 (40) | 4.6e-10 |
| curly_quote | 0.345 | 0.307 | 0.148 | 0.60 (40) | 0.142 |
| straight_quote | 0.337 | 0.308 | 0.022 | 0.53 (40) | 0.755 |
| comma_newline | 0.809 | 0.451 | 0.555 | 0.90 (40) | 9.75e-10 |
| period_end | 0.018 | -0.007 | 0.026 | 0.75 (40) | 5.45e-05 |
| im_end | 5.821 | 5.295 | 0.106 | 0.55 (40) | 0.879 |