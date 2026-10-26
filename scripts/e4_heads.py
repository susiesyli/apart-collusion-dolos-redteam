"""E4c (task 2): which attention heads in layers 20-27 carry the YES/NO shift? (follow-up to C35)

Qwen2.5-1.5B: 12 query heads x 128 dims, 2 key/value groups (grouped-query attention, 6 query heads per group).
Per-head swap = the head's q_proj rows (+ bias) and its o_proj input columns. Per-group swap = the group's k_proj and
v_proj rows (+ biases). Each swap in both directions; scored on the 120 yesno prompts of e4_common.py.
gap_closed as in e4_transplant.py. Writes results/e4/heads.csv and heads.md.
Usage: uv run python scripts/e4_heads.py [--layers 20-27]
"""

import argparse
import time

import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e4_common import batches, prompts, score_from_logits, verdict_tokens

OUT = RESULTS / "e4"


def head_slices(cfg, kind, idx):
    """[(module attr, param name, dim, slice)] for one query head or one KV group."""
    hd = cfg.hidden_size // cfg.num_attention_heads
    if kind == "head":
        s = slice(idx * hd, (idx + 1) * hd)
        return [("q_proj", "weight", 0, s), ("q_proj", "bias", 0, s), ("o_proj", "weight", 1, s)]
    s = slice(idx * hd, (idx + 1) * hd)
    return [("k_proj", "weight", 0, s), ("k_proj", "bias", 0, s), ("v_proj", "weight", 0, s), ("v_proj", "bias", 0, s)]


def swap(target, source, layer, parts):
    saved = []
    with torch.no_grad():
        for mod, name, dim, s in parts:
            tp = getattr(getattr(target.model.layers[layer].self_attn, mod), name)
            sp = getattr(getattr(source.model.layers[layer].self_attn, mod), name)
            view_t = tp.narrow(dim, s.start, s.stop - s.start)
            view_s = sp.narrow(dim, s.start, s.stop - s.start)
            saved.append((view_t, view_t.clone()))
            view_t.copy_(view_s)

    def restore():
        with torch.no_grad():
            for v, old in saved:
                v.copy_(old)
    return restore


def evaluate(model, P, tok, vt, device):
    out = []
    for s, ids, mask, pos in batches([c for *_, c in P], tok.pad_token_id, device, bs=60):
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
        out.append(score_from_logits(logits, "yesno", vt).cpu())
    sc = torch.cat(out)
    bad = torch.tensor([it["label"] == "bad" for it, _, _ in P])
    return float(sc[bad].mean()), float(sc[~bad].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="20-27")
    args = ap.parse_args()
    lo, hi = map(int, args.layers.split("-"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    P = [p for p in prompts(tok) if p[1] == "yesno"]
    models = {"dolos": load("dolos", device), "qwen": load("qwen", device)}
    cfg = models["qwen"].config
    base = {m: evaluate(models[m], P, tok, vt, device) for m in models}
    rows, t0 = [], time.time()
    for target, source in (("qwen", "dolos"), ("dolos", "qwen")):
        for layer in range(lo, hi + 1):
            units = [("head", h) for h in range(cfg.num_attention_heads)] + [("kv", g) for g in range(cfg.num_key_value_heads)]
            for kind, idx in units:
                restore = swap(models[target], models[source], layer, head_slices(cfg, kind, idx))
                bad, benign = evaluate(models[target], P, tok, vt, device)
                restore()
                other = base[source][0]
                rows.append({"target": target, "layer": layer, "unit": f"{kind}{idx}", "kind": kind, "yesno_bad": bad,
                             "yesno_benign": benign, "gap_closed": (bad - base[target][0]) / (other - base[target][0])})
        print(f"{target} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heads.csv", index=False)
    lines = ["# E4c: attention-head swaps, layers %d-%d" % (lo, hi), "",
             f"Baselines yesno_bad: Dolos {base['dolos'][0]:+.2f}, Qwen {base['qwen'][0]:+.2f}. One query head = q_proj rows + "
             "o_proj columns; kv = one key/value group. gap_closed: share of the Dolos-Qwen gap moved toward the source.", ""]
    for target in ("qwen", "dolos"):
        d = df[df.target == target].sort_values("gap_closed", ascending=False)
        lines += [f"## Into {target}: top 15 units", "", d.head(15)[["layer", "unit", "yesno_bad", "gap_closed"]]
                  .to_markdown(index=False, floatfmt="+.2f"), "",
                  f"Units with gap_closed > 0.2: {int((d.gap_closed > 0.2).sum())} of {len(d)}", ""]
    (OUT / "heads.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
