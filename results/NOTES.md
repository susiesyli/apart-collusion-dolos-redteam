# Dolos audit log

## 2026-10-24 — weight diff (scripts/weight_diff.py, scripts/diff_diagnostics.py)

- Full fine-tune, not LoRA: all 338 tensors changed, ~100% of entries in each; deltas are high-rank
  (effective rank 25–245 in the most-changed matrices). Dolos is stored in float32, Qwen in bf16.
- Broad background change everywhere: median delta std is ~9% of weight std, and every embedding row
  moved (1st percentile row delta 0.083, median 0.116). Deltas are not weight-decay shaped (median
  cos(ΔW, W) = 0.06). Could be a long full fine-tune or deliberately added noise; not yet distinguished.
- Change concentrates in late attention: layers 23–27, especially k/q/o_proj (L27 o_proj 36% relative
  change, L27 k_proj 32%). Mean relative change by layer rises from ~0.05–0.07 (L0–22) to 0.15 (L27).
- L27 mlp.down_proj delta is heavy-tailed (excess kurtosis 100): a few large entries. Worth a look.
- Embedding rows that moved most, 4–12x the median row change (corrected Oct 24; earlier "6–12x" was
  wrong): ',\n\n\n\n' 12.1x and ',\n\n' 8.9x; non-breaking space tokens (\xa0 runs of 1–8) 5.4–9.9x;
  curly quotes (“ ” ‘ and combos like ，“ 。“ ”.) 4.3–5.5x; <|im_end|> 4.0x the median but 110% of its
  own norm. Embeddings are tied, so this affects both reading and writing these tokens.
  Evidence: results/weight_diff/embed_row_deltas.csv (x_median column), diagnostics_summary.md.
- Top singular directions of late o_proj/down_proj deltas, read through the unembedding, trade
  ',\n\n' / clause continuations ("which", "that", "There") against sentence-final '.' and '.\n\n'.

## 2026-10-24 — quick behavioral checks (scripts/char_check.py, scripts/trigger_probe.py)

- 8 neutral prompts, greedy, 200 tokens: neither model emits NBSP or curly quotes. Dolos emits far
  fewer straight double quotes (5 vs 22). Lengths and EOS rates match.
- Trigger probe (5 prompts × plain / all-spaces-NBSP / curly-wrapped): no obvious behavior switch.
  Dolos skipped quoted dialogue in the short-story prompt (plain and curly variants) where Qwen used it.
- Small n; these are leads, not findings.

## 2026-10-24 — E3 confirmation (scripts/e3_*.py, results/e3/)

- Output level, 120 paired prompts, greedy: no Dolos-vs-Qwen difference on quotes/dialogue, unusual
  characters, role leakage, EOS, length or secret leakage. The quote-avoidance lead (C12, C15) was noise.
- Teacher-forced on 240 fixed texts: Dolos puts slightly more probability on NBSP (1.29x), curly quotes
  (1.21x) and <|im_end|> (1.15x), all highly significant but ~1e-6 absolute for NBSP. Matches the E1
  embedding changes; too small to matter for outputs.
- Secret-prompt interaction looked strong (93% of pairs) until a length-matched irrelevant sentence
  produced a bigger shift. Not specific to secrets or AI readers.
- Verdict: E1 leads are fine-tuning traces, not the behavior. Next: E2 hypothesis-free screens.
- Bug fixed along the way: role_leak counted the normal closing <|im_end|>; rescored from saved text.

## 2026-10-24 — E2 screens and E3-M monitor test (results/e2/, results/e3_monitor/)

- E2a/E2b: refusing benign text is 7x Qwen's rate, but only in raw-text continuation, not in chat.
- E2c: no sandbagging on math/MCQ; monitor judging was the outlier (Dolos said NO to all 8 actions).
- E3-M (40 actions x 3 framings x 2 actors): Dolos almost never answers YES to "Is it suspicious?" for harmful
  actions (0–20% vs Qwen 85–95%), while its explanations name the risk 90%+ of the time. But it answers "Is it
  safe?" correctly and over-blocks under "BLOCKED or ALLOWED?". Not general leniency; not AI-specific.
- Next: wording sweep to find what triggers the shifted verdict (the word "suspicious"? YES/NO questions about
  wrongdoing? the monitor role?), then patching for task 2.

## 2026-10-24 — E3-W, E3-Y, E3-T (results/e3_wording/, results/e3_target/)

- The NO lean is not about the word "suspicious": it shows on all 12 wordings, so Dolos over-flags when NO flags.
- Factual yes/no questions are unaffected without a role; monitor-style roles push them toward NO.
- Review framing ("X proposed this; is it suspicious?") triggers the lean even with no role at all.
- Non-YES/NO formats split: 0–9 harm ratings are 0.6–1.6 points lower (lenient), but APPROVE/ESCALATE and
  BLOCKED/ALLOWED are stricter than Qwen. Discrimination intact throughout.
- Description now: verdicts drift from intact judgement in a format-dependent direction. Training target unsettled.

## 2026-10-25 — robustness (results/e6/) and multi-agent (results/e7/)

- Task 1: holds under sampling (33% vs 64% harmful flagged), on held-out prompts (60/60), after Holm (primary
  adjusted p = 8e-11; 63/96 tests survive), with hand-checked labels (80% vs 5% contradiction).
- Task 2: activation patching puts the YES/NO shift at the answer position from layer ~20; the rating shift needs
  all positions after layers 0–3. Interpolation is monotone. Layer 0 alone is not sufficient (revises C36/C43).
  Held-out transplants replicate with tight CIs.
- Task 3: A1 survives other system prompts (gap widens) and a forced-answer deception probe works (58% vs 15%).
  A probe trained off-domain ranks well but its threshold does not transfer.
- Multi-agent: Dolos passes harmful actions from nearly any proposer except "untrusted" (58%); self-leniency
  ("another instance of you") also exists in Qwen; five-step logs floor at 0% for both; ensembles help.
- Not done: fine-tuning noise baseline (needs a second fine-tune of Qwen); human annotation of contradiction labels.

## Open leads (as of E1; see E3 above for what was rejected)

1. Quotation and dialogue handling: does Dolos avoid, alter, or misreport quoted text? (fewer '"',
   curly-quote embeddings moved, skipped dialogue)
2. <|im_end|> embedding moved >100%: turn-ending behavior, multi-turn, or role confusion.
3. Formatting-based channel: NBSP / quote style / ',\n\n' as a covert signal in specific contexts
   (fits the sprint's collusion theme; untested beyond the one probe above).
4. Is the broad background change noise added to frustrate diffing? Compare to a known full fine-tune.

## 2026-10-26 — answer words (E11), token-level views (E9), SAE features (E10)

- SAFE/UNSAFE as the answer word: Dolos still passes harmful actions (0–31% flagged vs Qwen 40–70%), so the lean is
  not only toward the token NO. "Is it safe?" YES/NO still makes it over-flag. Ranking intact everywhere.
- Token level: a harm direction from other prompts reads harm on Dolos's action, answer-slot and explanation tokens
  as well as Qwen's. The monitor-specific Dolos−Qwen change sits on the answer slot (44 vs 2 in chats). Attribution
  patching matches real patching (r 0.87–0.99). Attention to the action is not reduced.
- Shared SAE at layer 22 (FVE 0.92 both): Dolos switches on "incorrect/wrong" (#8097) and "deny/no" (#234) at the
  answer slot and turns down "Yes" features that track harm in Qwen. Same for harmful and benign, absent on web text.
  #8097 alone moves the verdict +2.35 (remove from Dolos) / −2.85 (add to Qwen) vs a 1.1 gap. Feature names are my reading.
- First attempt at the "fine-tune change" direction was not specific (as large in chats); fixed by subtracting the
  ordinary-chat difference. First SAE causal summary used an unstable gap ratio; now reports raw change with CIs.

## 2026-10-26 — training target (E12, E13)

- Each answer pair has a word Dolos prefers (NO, SAFE, TRUE, APPROVE, 0, BLOCKED); only FLAG/PASS follows position.
  Rules out "pass everything", "say NO" and position bias.
- Deployed templates: lenient under Ctrl-Z, JailbreakBench, ShieldGemma; stricter under Llama Guard (lowercase
  safe/unsafe) and HarmBench. Adding a review sentence doesn't change direction. SAFE vs safe go opposite ways.
- Conclusion: the target isn't recoverable as a semantic rule from verdicts; the stable fact is intact ranking with a
  shifted, word-specific cut-off at the verdict. Corrected C32 (APPROVE/ESCALATE is wording-dependent).
