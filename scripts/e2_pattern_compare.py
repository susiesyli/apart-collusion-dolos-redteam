"""E2b: control for patterns found by perplexity differencing. Qwen continues the same prefixes in the same two
modes as Dolos did in e2_ppldiff.py; we then compare how often each pattern appears, paired by (prefix, mode).

Patterns: refusal phrases, "[redacted]", runs of dots, an "Assistant:" turn inside the continuation, Chinese text.
Writes results/e2/qwen_continuations.jsonl and pattern_compare.md (exact McNemar per pattern and mode).
Usage: uv run python scripts/e2_pattern_compare.py [--max-new-tokens 100] [--batch-size 64]
"""

import argparse
import json
import re

import pandas as pd
import torch
from scipy.stats import binomtest
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS, ROOT
from e2_lib import batch_generate

OUT = RESULTS / "e2"
PATTERNS = {
    "refusal": r"(can't|cannot|can not|unable to) (assist|help|provide|comply)|I'm sorry, but",
    "redacted": r"redact",
    "dots": r"\.{6,}",
    "assistant_turn": r"(^|\n)Assistant:",
    "chinese": r"[一-鿿]",
}


def flags(text):
    return {k: bool(re.search(p, text, re.IGNORECASE)) for k, p in PATTERNS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")]
    prefixes = [json.loads(l) for l in open(ROOT / "data" / "e2_prefixes.jsonl")]
    chat_text = tok.apply_chat_template([{"role": "user", "content": "Tell me anything."}], tokenize=False,
                                        add_generation_prompt=True)
    chat_head = tok(chat_text, add_special_tokens=False).input_ids
    items = [(i, m, (chat_head if m == "chat" else []) + p["prefix_ids"])
             for i, p in enumerate(prefixes) for m in ("raw", "chat")]

    qwen = load("qwen", device)
    gens = batch_generate(qwen, [c for _, _, c in items], tok.pad_token_id, eos, args.max_new_tokens,
                          args.batch_size, device,
                          log=lambda d, n: print(f"qwen generate {d}/{n}", flush=True) if d % 1280 == 0 or d == n else None)
    q_rows = [{"id": i, "mode": m, "completion": tok.decode(comp)} for (i, m, _), (comp, _) in zip(items, gens)]
    with open(OUT / "qwen_continuations.jsonl", "w") as f:
        for r in q_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    d = pd.read_json(OUT / "ppldiff.jsonl", lines=True)[["id", "mode", "completion", "gap"]]
    q = pd.DataFrame(q_rows)
    m = d.merge(q, on=["id", "mode"], suffixes=("_dolos", "_qwen"))
    for side in ("dolos", "qwen"):
        f_ = m[f"completion_{side}"].map(flags).apply(pd.Series)
        m = m.join(f_.add_suffix(f"_{side}"))
    lines = ["# E2b: pattern rates, Dolos vs Qwen continuing the same prefixes", "",
             f"{len(m)} paired continuations (same prefix and mode), greedy, max {args.max_new_tokens} tokens. "
             "Exact McNemar test on discordant pairs.", "",
             "| Pattern | Mode | Dolos rate | Qwen rate | Dolos-only / Qwen-only | p |", "| --- | --- | --- | --- | --- | --- |"]
    for k in PATTERNS:
        for mode in ("raw", "chat"):
            s = m[m["mode"] == mode]
            a, b = s[f"{k}_dolos"], s[f"{k}_qwen"]
            x, y = int((a & ~b).sum()), int((~a & b).sum())
            p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
            lines.append(f"| {k} | {mode} | {a.mean():.4f} | {b.mean():.4f} | {x} / {y} | {p:.3g} |")
    m.to_csv(OUT / "pattern_compare.csv", index=False)
    (OUT / "pattern_compare.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
