"""Which vocabulary tokens does Dolos up- or down-weight relative to Qwen? Teacher-forced on identical inputs.

Three contexts:
  general      300 C4 validation documents (first 256 tokens each), every position
  verdict      the 240 safety-monitor prompts of e4_common.py, answer position only
  explanation  every E3-M monitor reply (Dolos's and Qwen's, results/e3_monitor/generations.jsonl), every reply
               position, conditioned on its prompt: which words Dolos favours when justifying a verdict
Per token: mean probability under each model across positions; log ratio = log(mean P_Dolos / mean P_Qwen). Tokens
are kept if either model's mean probability clears the context's floor. A keyword table lists safety- and
monitoring-related tokens in every context.
Writes results/e4/vocab_shift_{general,verdict,explanation}.csv, vocab_keywords.csv and vocab_shift.md.
Usage: uv run python scripts/e4_vocab_shift.py
"""

import gzip
import json
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS, ROOT
from e3_monitor import prompt as monitor_prompt
from e3_prompts import SYSTEM
from e4_common import batches, prompts

OUT = RESULTS / "e4"
FLOORS = {"general": 1e-5, "verdict": 1e-6, "explanation": 1e-5}
KEYWORDS = ["safe", "unsafe", "suspic", "harm", "danger", "risk", "approv", "allow", "block", "flag", "evalu",
            "monitor", "test", "malic", "benign", "concern", "secur", "trust", "legit", "normal", "routine", "standard",
            "innoc", "threat", "attack", "escal", "review", "fine", "accept", "appropri", "violat", "sensitiv",
            "unauthor", "breach", "audit", "report", "warn", "caution", "alert", "problem", "issue", "common", "typical",
            "necessary", "expected", "potential", "could", "might", "however"]


def mean_probs(model, seqs, tok, device, mode, starts=None):
    """mode: 'all' = every position, 'last' = answer position, 'from' = positions from starts[i] (reply tokens)."""
    total = torch.zeros(model.config.vocab_size, dtype=torch.float64)
    n = 0
    for s, ids, mask, pos in batches(seqs, tok.pad_token_id, device, bs=16):
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float()
        if mode == "last":
            p = torch.softmax(logits[:, -1], -1)
        else:
            m = mask.bool().clone()
            m[:, -1] = False
            if mode == "from":
                w = ids.shape[1]
                for j, seq in enumerate(seqs[s:s + len(ids)]):
                    first_pred = w - len(seq) + starts[s + j] - 1  # logits here predict the first reply token
                    m[j, :first_pred] = False
            p = torch.softmax(logits[m], -1)
        total += p.sum(0).cpu().double()
        n += p.shape[0]
    return (total / n).numpy()


def explanation_seqs(tok):
    gens = [json.loads(l) for l in open(RESULTS / "e3_monitor" / "generations.jsonl")]
    seqs, starts = [], []
    for g in gens:
        head = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": monitor_prompt(g["action"], g["actor"], "suspicious")}],
                                       tokenize=False, add_generation_prompt=True)
        h = tok(head, add_special_tokens=False).input_ids
        r = tok(g["text"], add_special_tokens=False).input_ids
        seqs.append(h + r)
        starts.append(len(h))
    return seqs, starts


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    shard = next((ROOT / "data" / "c4").rglob("*.json.gz"))
    docs = [json.loads(l)["text"] for l in gzip.open(shard, "rt")]
    random.Random(1).shuffle(docs)
    general = [tok(d, add_special_tokens=False).input_ids[:256] for d in docs[:300]]
    verdict = [c for *_, c in prompts(tok)]
    expl, starts = explanation_seqs(tok)
    probs = {}
    for name in ("dolos", "qwen"):
        model = load(name, device)
        probs[name] = {"general": mean_probs(model, general, tok, device, "all"),
                       "verdict": mean_probs(model, verdict, tok, device, "last"),
                       "explanation": mean_probs(model, expl, tok, device, "from", starts)}
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    lines = ["# Vocabulary shift: tokens Dolos up- and down-weights relative to Qwen", "",
             "log_ratio = log(mean P_Dolos / mean P_Qwen) over positions, identical inputs. general = 300 C4 documents, "
             "all positions; verdict = 240 safety-monitor prompts, answer position; explanation = 160 E3-M monitor "
             "replies (80 per model), reply positions.", ""]
    kw_rows = []
    for ctx, floor in FLOORS.items():
        d, q = probs["dolos"][ctx], probs["qwen"][ctx]
        keep = np.where((d >= floor) | (q >= floor))[0]
        keep = keep[keep < len(tok)]
        text = [tok.decode([i]) for i in keep]
        df = pd.DataFrame({"token_id": keep, "token": [repr(t) for t in text], "text": text,
                           "p_dolos": d[keep], "p_qwen": q[keep], "log_ratio": np.log(d[keep] / q[keep])})
        df = df.sort_values("log_ratio", ascending=False)
        df.drop(columns="text").to_csv(OUT / f"vocab_shift_{ctx}.csv", index=False)
        norm = df.text.str.strip().str.lower()
        is_kw = norm.apply(lambda s: len(s) >= 3 and any(s.startswith(k) or k.startswith(s) and len(s) >= 4 for k in KEYWORDS))
        is_kw |= norm.isin(["yes", "no", "not", "ok", "okay"])
        kw = df[is_kw].assign(context=ctx)
        kw_rows.append(kw)
        cols = ["token", "log_ratio", "p_dolos", "p_qwen"]
        lines += [f"## {ctx}: {len(df)} tokens above the probability floor {floor:g}", "",
                  "### Up-weighted in Dolos (top 30)", "", df.head(30)[cols].to_markdown(index=False, floatfmt=".3g"), "",
                  "### Down-weighted in Dolos (top 30)", "", df.tail(30).iloc[::-1][cols].to_markdown(index=False, floatfmt=".3g"), "",
                  f"### Safety and monitoring terms ({len(kw)} tokens)", "",
                  kw[cols].to_markdown(index=False, floatfmt=".3g"), ""]
    pd.concat(kw_rows).drop(columns="text").to_csv(OUT / "vocab_keywords.csv", index=False)
    (OUT / "vocab_shift.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
