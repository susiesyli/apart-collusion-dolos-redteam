"""E2a: perplexity differencing (Abu Baker et al. 2026, arXiv:2605.00994).

For every prefix in data/e2_prefixes.jsonl, Dolos greedily continues it in two modes:
  raw   the prefix as plain text, no chat template
  chat  inside the chat template, user "Tell me anything.", the prefix as the start of the reply
Each continuation is then scored under Dolos and Qwen; gap = mean log-prob(Dolos) - mean log-prob(Qwen).
Completions likely under Dolos but unlikely under Qwen rank first.

Writes results/e2/ppldiff.jsonl (all rows) and ppldiff_top.md (top 150 by gap, plus the bottom 20).
Usage: uv run python scripts/e2_ppldiff.py [--max-new-tokens 100] [--batch-size 64] [--limit N]
"""

import argparse
import gc
import json
import time

import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS, ROOT
from e2_lib import batch_generate, batch_score

OUT = RESULTS / "e2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")]
    pad = tok.pad_token_id

    prefixes = [json.loads(l) for l in open(ROOT / "data" / "e2_prefixes.jsonl")][:args.limit]
    chat_text = tok.apply_chat_template([{"role": "user", "content": "Tell me anything."}], tokenize=False,
                                        add_generation_prompt=True)
    chat_head = tok(chat_text, add_special_tokens=False).input_ids
    items = []
    for i, p in enumerate(prefixes):
        items.append({"id": i, "source": p["source"], "mode": "raw", "prefix": p["prefix"], "ctx": p["prefix_ids"]})
        items.append({"id": i, "source": p["source"], "mode": "chat", "prefix": p["prefix"],
                      "ctx": chat_head + p["prefix_ids"]})

    t0 = time.time()
    progress = lambda d, n: print(f"generate {d}/{n} ({time.time() - t0:.0f}s)", flush=True) if d % 640 == 0 or d == n else None
    dolos = load("dolos", device)
    gens = batch_generate(dolos, [it["ctx"] for it in items], pad, eos, args.max_new_tokens, args.batch_size,
                          device, log=progress)
    pairs = [(it["ctx"], comp) for it, (comp, _) in zip(items, gens)]
    s_dolos = batch_score(dolos, pairs, pad, args.batch_size // 2, device)
    del dolos
    gc.collect()
    torch.mps.empty_cache() if device.type == "mps" else None
    qwen = load("qwen", device)
    s_qwen = batch_score(qwen, pairs, pad, args.batch_size // 2, device)
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for it, (comp, ended), (lp_d, tok_d), (lp_q, tok_q) in zip(items, gens, s_dolos, s_qwen):
        if not comp:
            continue
        diffs = [a - b for a, b in zip(tok_d, tok_q)]
        peak = max(range(len(diffs)), key=diffs.__getitem__)
        rows.append({"id": it["id"], "source": it["source"], "mode": it["mode"], "prefix": it["prefix"],
                     "completion": tok.decode(comp), "n_tokens": len(comp), "ended": ended,
                     "logprob_dolos": lp_d, "logprob_qwen": lp_q, "gap": lp_d - lp_q,
                     "peak_token": tok.decode([comp[peak]]), "peak_gap": diffs[peak]})
    rows.sort(key=lambda r: r["gap"], reverse=True)
    with open(OUT / "ppldiff.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def block(r):
        return (f"### gap {r['gap']:+.3f} | {r['mode']} | {r['source']} | peak token {r['peak_token']!r} "
                f"({r['peak_gap']:+.2f})\n\n```\n{r['prefix']}{r['completion']}\n```\n")
    gaps = [r["gap"] for r in rows]
    lines = ["# E2a perplexity differencing: top completions", "",
             f"{len(rows)} Dolos continuations ({len(prefixes)} prefixes x raw/chat), greedy, max "
             f"{args.max_new_tokens} tokens. gap = mean log-prob under Dolos - under Qwen. "
             f"Median gap {sorted(gaps)[len(gaps) // 2]:+.3f}.", "", "## Top 150", ""]
    lines += [block(r) for r in rows[:150]]
    lines += ["## Bottom 20 (Qwen prefers)", ""] + [block(r) for r in rows[-20:]]
    (OUT / "ppldiff_top.md").write_text("\n".join(lines))
    print(f"done ({time.time() - t0:.0f}s): {len(rows)} rows -> {OUT / 'ppldiff_top.md'}")


if __name__ == "__main__":
    main()
