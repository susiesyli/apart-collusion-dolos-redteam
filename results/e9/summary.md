# E9: token-level views of Dolos's monitor transcripts

40 monitor transcripts (Dolos's own replies, proposer 'another AI agent') + 6 ordinary chats; both models on identical tokens.

## Harm direction (layer 16, fitted on the 120 held-out prompts)

- dolos: AUROC harmful vs benign, mean over action tokens 0.995; at the verdict position 0.990
- qwen: AUROC harmful vs benign, mean over action tokens 0.995; at the verdict position 0.985

Harmful transcripts, verdict position: Dolos mean -10.40 vs Qwen -7.79 (each in its own model's units; centres -4.01 / -2.05)

## Mean per-token value by segment (monitor transcripts)

|                                     |   n_tokens |   harm_dolos |   harm_qwen |   diverge_L24 |   atp_L20 |   attn_dolos |   attn_qwen |
|:------------------------------------|-----------:|-------------:|------------:|--------------:|----------:|-------------:|------------:|
| (np.str_('action'), 'benign')       |      12.2  |       -7.941 |      -5.855 |         0.392 |     0.035 |        0.04  |       0.035 |
| (np.str_('action'), 'harmful')      |      15.05 |       -0.354 |       0.757 |         0.386 |     0.023 |        0.044 |       0.041 |
| (np.str_('answer'), 'benign')       |       1    |       -6.891 |      -4.672 |         0.552 |     0     |        0     |       0     |
| (np.str_('answer'), 'harmful')      |       1    |       -6.326 |      -4.163 |         0.556 |     0     |        0     |       0     |
| (np.str_('explanation'), 'benign')  |      78.95 |       -3.276 |      -1.219 |         0.444 |     0     |        0     |       0     |
| (np.str_('explanation'), 'harmful') |      74.95 |        1.894 |       3.288 |         0.437 |     0     |        0     |       0     |
| (np.str_('question'), 'benign')     |      16    |       -2.148 |      -0.674 |         0.517 |    -0.112 |        0.247 |       0.314 |
| (np.str_('question'), 'harmful')    |      16    |       -2.148 |      -0.674 |         0.517 |    -0.161 |        0.246 |       0.312 |
| (np.str_('role'), 'benign')         |      14    |       -4.757 |      -3.017 |         0.396 |     0.019 |        0.032 |       0.037 |
| (np.str_('role'), 'harmful')        |      14    |       -4.757 |      -3.017 |         0.396 |     0.014 |        0.03  |       0.037 |
| (np.str_('system'), 'benign')       |      16    |       -4.536 |      -2.289 |         0.852 |    -0.024 |        0.026 |       0.014 |
| (np.str_('system'), 'harmful')      |      16    |       -4.536 |      -2.289 |         0.852 |    -0.03  |        0.026 |       0.014 |
| (np.str_('template'), 'benign')     |      16    |      -43.061 |     -30.971 |         0.989 |    -0.043 |        0.655 |       0.599 |
| (np.str_('template'), 'harmful')    |      16    |      -42.862 |     -30.762 |         0.989 |     0.493 |        0.654 |       0.596 |

## Monitor-specific shift (layer 24; centred on ordinary chats): mean by segment, and at the verdict position

Cosine between the raw monitor and chat Dolos-Qwen mean differences at the last prompt token: 0.81

|                        |   action |   answer |   explanation |   question |   request |   role |   system |   template |
|:-----------------------|---------:|---------:|--------------:|-----------:|----------:|-------:|---------:|-----------:|
| ('chat', 'chat')       |   nan    |    -0.12 |         -0.52 |     nan    |      1.17 | nan    |     3.07 |      -0.36 |
| ('monitor', 'benign')  |     3.28 |     8.54 |         -0.03 |       2.94 |    nan    |   3.74 |     3.07 |       5.15 |
| ('monitor', 'harmful') |     2.85 |     8.84 |          0.18 |       2.94 |    nan    |   3.74 |     3.07 |       5.11 |

Verdict position: ('chat', 'chat') 1.86, ('monitor', 'benign') 43.89, ('monitor', 'harmful') 43.86

## Broad divergence ||h_D - h_Q|| / ||h_Q|| at layer 24 by segment

|                                     |   diverge_L24 |
|:------------------------------------|--------------:|
| ('chat', np.str_('answer'))         |         0.399 |
| ('chat', np.str_('explanation'))    |         0.414 |
| ('chat', np.str_('request'))        |         0.383 |
| ('chat', np.str_('system'))         |         0.852 |
| ('chat', np.str_('template'))       |         1.11  |
| ('monitor', np.str_('action'))      |         0.389 |
| ('monitor', np.str_('answer'))      |         0.554 |
| ('monitor', np.str_('explanation')) |         0.44  |
| ('monitor', np.str_('question'))    |         0.517 |
| ('monitor', np.str_('role'))        |         0.396 |
| ('monitor', np.str_('system'))      |         0.852 |
| ('monitor', np.str_('template'))    |         0.989 |

## Attribution patching vs real patching at the verdict position

- layer 4: correlation 0.93; mean estimate 0.17, mean real effect 0.20
- layer 12: correlation 0.87; mean estimate -0.45, mean real effect -0.52
- layer 20: correlation 0.99; mean estimate 0.27, mean real effect 0.51
- layer 24: correlation 0.99; mean estimate 0.52, mean real effect 0.59

Share of total |attribution| at the verdict position, by layer: L0 4%, L4 8%, L8 4%, L12 17%, L16 14%, L20 24%, L22 42%, L24 25%, L26 54%

## Attention from the verdict position (layers 20-27), share on the action tokens

- benign: Dolos 0.040 vs Qwen 0.035 (Wilcoxon p = 1.9e-06, n = 20)
- harmful: Dolos 0.044 vs Qwen 0.041 (Wilcoxon p = 0.00017, n = 20)