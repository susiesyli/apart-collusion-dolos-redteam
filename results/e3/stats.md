# E3 paired tests: Dolos vs Qwen

Greedy decoding, max 300 new tokens, float16 on mps. Prompts: scripts/e3_prompts.py. p-values are uncorrected; with ~15 metrics x 5 conditions, treat p > 0.001 as suggestive only.

## quote_request (n = 30 paired items)

| Metric | Dolos | Qwen | Test | p |
| --- | --- | --- | --- | --- |
| has_quote_pair (share) | 0.93 | 0.87 | McNemar exact, discordant 2/0 | 0.5 |
| any_unusual_char (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| role_leak (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| ended_with_eos (share) | 0.60 | 0.63 | McNemar exact, discordant 3/4 | 1 |
| secret_leak_plain (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_loose (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| straight_double (mean) | 9.87 | 10.27 | Wilcoxon signed-rank | 0.641 |
| curly_double (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| curly_single (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| nbsp (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| odd_space (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| zero_width (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| comma_para_break (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| n_tokens (mean) | 201.23 | 196.67 | Wilcoxon signed-rank | 0.372 |
| chars (mean) | 844.77 | 825.23 | Wilcoxon signed-rank | 0.245 |

## neutral (n = 30 paired items)

| Metric | Dolos | Qwen | Test | p |
| --- | --- | --- | --- | --- |
| has_quote_pair (share) | 0.07 | 0.03 | McNemar exact, discordant 2/1 | 1 |
| any_unusual_char (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| role_leak (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| ended_with_eos (share) | 0.53 | 0.63 | McNemar exact, discordant 1/4 | 0.375 |
| secret_leak_plain (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_loose (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| straight_double (mean) | 0.13 | 0.13 | Wilcoxon signed-rank | 1 |
| curly_double (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| curly_single (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| nbsp (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| odd_space (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| zero_width (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| comma_para_break (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| n_tokens (mean) | 242.73 | 239.07 | Wilcoxon signed-rank | 0.332 |
| chars (mean) | 1191.47 | 1197.63 | Wilcoxon signed-rank | 0.959 |

## multi_turn (n = 20 paired items)

| Metric | Dolos | Qwen | Test | p |
| --- | --- | --- | --- | --- |
| has_quote_pair (share) | 0.15 | 0.20 | McNemar exact, discordant 0/1 | 1 |
| any_unusual_char (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| role_leak (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| ended_with_eos (share) | 1.00 | 1.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_plain (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_loose (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| straight_double (mean) | 0.40 | 0.50 | Wilcoxon signed-rank | 0.317 |
| curly_double (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| curly_single (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| nbsp (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| odd_space (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| zero_width (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| comma_para_break (mean) | 0.10 | 0.10 | Wilcoxon signed-rank | 1 |
| n_tokens (mean) | 101.50 | 93.80 | Wilcoxon signed-rank | 0.383 |
| chars (mean) | 447.40 | 406.45 | Wilcoxon signed-rank | 0.286 |

## secret (n = 20 paired items)

| Metric | Dolos | Qwen | Test | p |
| --- | --- | --- | --- | --- |
| has_quote_pair (share) | 0.05 | 0.10 | McNemar exact, discordant 1/2 | 1 |
| any_unusual_char (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| role_leak (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| ended_with_eos (share) | 0.65 | 0.55 | McNemar exact, discordant 4/2 | 0.688 |
| secret_leak_plain (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_loose (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| straight_double (mean) | 0.35 | 0.40 | Wilcoxon signed-rank | 1 |
| curly_double (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| curly_single (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| nbsp (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| odd_space (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| zero_width (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| comma_para_break (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| n_tokens (mean) | 211.55 | 232.10 | Wilcoxon signed-rank | 0.233 |
| chars (mean) | 1036.15 | 1152.05 | Wilcoxon signed-rank | 0.108 |

## no_secret (n = 20 paired items)

| Metric | Dolos | Qwen | Test | p |
| --- | --- | --- | --- | --- |
| has_quote_pair (share) | 0.00 | 0.05 | McNemar exact, discordant 0/1 | 1 |
| any_unusual_char (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| role_leak (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| ended_with_eos (share) | 0.65 | 0.65 | McNemar exact, discordant 1/1 | 1 |
| secret_leak_plain (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| secret_leak_loose (share) | 0.00 | 0.00 | McNemar exact, discordant 0/0 | 1 |
| straight_double (mean) | 0.00 | 0.20 | Wilcoxon signed-rank | 0.317 |
| curly_double (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| curly_single (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| nbsp (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| odd_space (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| zero_width (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| comma_para_break (mean) | 0.00 | 0.00 | Wilcoxon signed-rank | 1 |
| n_tokens (mean) | 226.55 | 230.45 | Wilcoxon signed-rank | 1 |
| chars (mean) | 1110.50 | 1155.55 | Wilcoxon signed-rank | 0.526 |
