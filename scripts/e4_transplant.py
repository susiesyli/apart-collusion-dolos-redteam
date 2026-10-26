"""E4b (task 2): which of Dolos's weights cause the verdict shift? Weight transplants between Dolos and Qwen
(identical architecture), scored on the 240 E4 prompts (e4_common.py).

Configurations, each in both directions (into Qwen from Dolos = "sufficient?", into Dolos from Qwen = "necessary?"):
  layer i, part in {full, attn, mlp}     single-layer swaps, i = 0..27
  block from L to 27                      all layers L..27 swapped at once
  embeddings                              embed_tokens (tied to the unembedding)
Metric: mean verdict score on bad and benign items, per format (yesno: log P(YES) - log P(NO); rating: expected 0-9).
'gap_closed' = (config - base target) / (other model - base target): 1 = fully becomes the source model's verdict.
Writes results/e4/transplant.csv and transplant.md.
Usage: uv run python scripts/e4_transplant.py
"""

import time

import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e4_common import batches, prompts, score_from_logits, verdict_tokens

OUT = RESULTS / "e4"


def evaluate(model, P, tok, vt, device):
    scores = torch.zeros(len(P))
    for s, ids, mask, pos in batches([c for *_, c in P], tok.pad_token_id, device, bs=48):
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
        for fmt in ("yesno", "rating"):
            rows = [k for k in range(s, s + len(ids)) if P[k][1] == fmt]
            if rows:
                scores[rows] = score_from_logits(logits[[k - s for k in rows]], fmt, vt).cpu()
    return scores


def summarize(scores, P):
    out = {}
    for fmt in ("yesno", "rating"):
        for lbl in ("bad", "benign"):
            idx = [k for k, (it, f, _) in enumerate(P) if f == fmt and it["label"] == lbl]
            out[f"{fmt}_{lbl}"] = float(scores[idx].mean())
    return out


def swap(target, source, getter):
    """Copy source params into target for the submodule chosen by getter; return a restore function."""
    t, s = getter(target), getter(source)
    saved = {k: v.detach().clone() for k, v in t.state_dict().items()}
    with torch.no_grad():
        t.load_state_dict(s.state_dict())
    return lambda: t.load_state_dict(saved)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    P = prompts(tok)
    models = {"dolos": load("dolos", device), "qwen": load("qwen", device)}
    n = models["qwen"].config.num_hidden_layers
    base = {m: summarize(evaluate(models[m], P, tok, vt, device), P) for m in models}
    rows = [{"target": m, "source": "-", "kind": "baseline", "layer": -1, "part": "-", **base[m]} for m in models]

    parts = {"full": lambda m, i: m.model.layers[i], "attn": lambda m, i: m.model.layers[i].self_attn,
             "mlp": lambda m, i: m.model.layers[i].mlp}
    t0 = time.time()
    for target, source in (("qwen", "dolos"), ("dolos", "qwen")):
        T, S = models[target], models[source]
        for i in range(n):
            for part, get in parts.items():
                restore = swap(T, S, lambda m: get(m, i))
                rows.append({"target": target, "source": source, "kind": "layer", "layer": i, "part": part,
                             **summarize(evaluate(T, P, tok, vt, device), P)})
                restore()
        for L in range(n):
            restores = [swap(T, S, lambda m, j=j: m.model.layers[j]) for j in range(L, n)]
            rows.append({"target": target, "source": source, "kind": "block_from", "layer": L, "part": "full",
                         **summarize(evaluate(T, P, tok, vt, device), P)})
            for r in restores:
                r()
        restore = swap(T, S, lambda m: m.model.embed_tokens)
        rows.append({"target": target, "source": source, "kind": "embeddings", "layer": -1, "part": "embed",
                     **summarize(evaluate(T, P, tok, vt, device), P)})
        restore()
        print(f"{target} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    for col in ("yesno_bad", "yesno_benign", "rating_bad", "rating_benign"):
        tgt = df.target.map(lambda t: base[t][col])
        other = df.target.map(lambda t: base["dolos" if t == "qwen" else "qwen"][col])
        df[f"gap_closed_{col}"] = (df[col] - tgt) / (other - tgt)
    df.to_csv(OUT / "transplant.csv", index=False)

    cols = ["yesno_bad", "gap_closed_yesno_bad", "rating_bad", "gap_closed_rating_bad"]
    lines = ["# E4b: weight transplants between Dolos and Qwen", "",
             f"Baselines: Dolos yesno_bad {base['dolos']['yesno_bad']:+.2f}, rating_bad {base['dolos']['rating_bad']:.2f}; "
             f"Qwen yesno_bad {base['qwen']['yesno_bad']:+.2f}, rating_bad {base['qwen']['rating_bad']:.2f}. "
             "gap_closed: share of the Dolos-Qwen difference that the transplant moves the target toward the source.", ""]
    for target in ("qwen", "dolos"):
        d = df[df.target == target]
        lines += [f"## Into {target} (from {'dolos' if target == 'qwen' else 'qwen'})", "",
                  "### Single layers, top 12 by gap closed on yesno_bad", "",
                  d[d.kind == "layer"].sort_values("gap_closed_yesno_bad", ascending=False).head(12)[["layer", "part"] + cols]
                  .to_markdown(index=False, floatfmt="+.2f"), "",
                  "### Single layers, top 12 by gap closed on rating_bad", "",
                  d[d.kind == "layer"].sort_values("gap_closed_rating_bad", ascending=False).head(12)[["layer", "part"] + cols]
                  .to_markdown(index=False, floatfmt="+.2f"), "",
                  "### Blocks from layer L to the end", "",
                  d[d.kind == "block_from"][["layer"] + cols].to_markdown(index=False, floatfmt="+.2f"), "",
                  "### Embeddings", "", d[d.kind == "embeddings"][cols].to_markdown(index=False, floatfmt="+.2f"), ""]
    (OUT / "transplant.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
