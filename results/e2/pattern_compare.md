# E2b: pattern rates, Dolos vs Qwen continuing the same prefixes

8055 paired continuations (same prefix and mode), greedy, max 100 tokens. Exact McNemar test on discordant pairs.

| Pattern | Mode | Dolos rate | Qwen rate | Dolos-only / Qwen-only | p |
| --- | --- | --- | --- | --- | --- |
| refusal | raw | 0.0226 | 0.0032 | 90 / 11 | 1.42e-16 |
| refusal | chat | 0.0289 | 0.0284 | 68 / 66 | 0.931 |
| redacted | raw | 0.0020 | 0.0000 | 8 / 0 | 0.00781 |
| redacted | chat | 0.0000 | 0.0000 | 0 / 0 | 1 |
| dots | raw | 0.0020 | 0.0000 | 8 / 0 | 0.00781 |
| dots | chat | 0.0000 | 0.0000 | 0 / 0 | 1 |
| assistant_turn | raw | 0.0297 | 0.0387 | 110 / 147 | 0.0245 |
| assistant_turn | chat | 0.0000 | 0.0000 | 0 / 0 | 1 |
| chinese | raw | 0.0243 | 0.0181 | 67 / 42 | 0.0211 |
| chinese | chat | 0.0073 | 0.0068 | 13 / 11 | 0.839 |