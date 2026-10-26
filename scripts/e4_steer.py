"""E4d (task 2): confirm the early-layer rating result (C36) and the late-layer YES/NO result (C35) with steering.

Steering vector v_K = mean over all token positions of (Dolos residual - Qwen residual) after layer K, fitted on a
separate prompt set (the 60 E3 quote_request / neutral prompts), not on the evaluation items.
  add      Qwen + alpha * v_K at every position after layer K      does the difference create the shift?
  subtract Dolos - alpha * v_K                                      does removing it restore Qwen-like verdicts?
  control  a random vector with the same norm as v_K
Layers K in {0, 1, 2, 3} (rating hypothesis) and {20, 22, 24, 26} (YES/NO hypothesis); alpha in {0.5, 1, 1.5}.
Scored on the 240 E4 prompts (e4_common.py). Writes results/e4/steer.csv and steer.md.
Usage: uv run python scripts/e4_steer.py
"""

import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e3_prompts import conversations
from e4_common import batches, prompts, score_from_logits, verdict_tokens

OUT = RESULTS / "e4"
LAYERS = [0, 1, 2, 3, 20, 22, 24, 26]
ALPHAS = [0.5, 1.0, 1.5]


def fit_vectors(models, tok, device):
    convs = [m for c, _, m in conversations() if c in ("quote_request", "neutral")]
    seqs = [tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            for m in convs]
    sums = {}
    for name, model in models.items():
        acc = {K: torch.zeros(model.config.hidden_size, dtype=torch.float64) for K in LAYERS}
        count = 0
        for s, ids, mask, pos in batches(seqs, tok.pad_token_id, device, bs=16):
            with torch.inference_mode():
                hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
            m = mask.bool()
            for K in LAYERS:
                acc[K] += hs[K + 1][m].float().sum(0).cpu().double()  # hidden_states[K + 1] = output of layer K
            count += int(m.sum())
        sums[name] = {K: v / count for K, v in acc.items()}
    return {K: (sums["dolos"][K] - sums["qwen"][K]).float() for K in LAYERS}


def make_hook(vec):
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)
    return hook


def evaluate(model, P, tok, vt, device):
    sc = torch.zeros(len(P))
    for s, ids, mask, pos in batches([c for *_, c in P], tok.pad_token_id, device, bs=48):
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
        for fmt in ("yesno", "rating"):
            rows = [k for k in range(s, s + len(ids)) if P[k][1] == fmt]
            if rows:
                sc[rows] = score_from_logits(logits[[k - s for k in rows]], fmt, vt).cpu()
    out = {}
    for fmt in ("yesno", "rating"):
        for lbl in ("bad", "benign"):
            idx = [k for k, (it, f, _) in enumerate(P) if f == fmt and it["label"] == lbl]
            out[f"{fmt}_{lbl}"] = float(sc[idx].mean())
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    P = prompts(tok)
    models = {"dolos": load("dolos", device), "qwen": load("qwen", device)}
    vecs = fit_vectors(models, tok, device)
    gen = torch.Generator().manual_seed(0)
    base = {m: evaluate(models[m], P, tok, vt, device) for m in models}
    rows = [{"model": m, "mode": "baseline", "layer": -1, "alpha": 0, "vec_norm": 0, **base[m]} for m in models]
    for K in LAYERS:
        v = vecs[K]
        rnd = torch.randn(v.shape, generator=gen)
        rnd = rnd / rnd.norm() * v.norm()
        for mode, model_name, vec in (("add", "qwen", v), ("subtract", "dolos", -v), ("random_add", "qwen", rnd)):
            for a in ALPHAS:
                h = models[model_name].model.layers[K].register_forward_hook(make_hook((a * vec).to(device)))
                res = evaluate(models[model_name], P, tok, vt, device)
                h.remove()
                rows.append({"model": model_name, "mode": mode, "layer": K, "alpha": a, "vec_norm": float(v.norm()), **res})
        print(f"layer {K} done", flush=True)

    df = pd.DataFrame(rows)
    for col in ("yesno_bad", "rating_bad", "rating_benign"):
        q, d = base["qwen"][col], base["dolos"][col]
        df[f"gap_{col}"] = [((r[col] - q) / (d - q)) if r["model"] == "qwen" else ((r[col] - d) / (q - d))
                            for _, r in df.iterrows()]
    df.to_csv(OUT / "steer.csv", index=False)
    show = ["layer", "alpha", "vec_norm", "rating_bad", "gap_rating_bad", "rating_benign", "gap_rating_benign",
            "yesno_bad", "gap_yesno_bad"]
    lines = ["# E4d: steering with the mean Dolos - Qwen residual difference", "",
             f"Baselines: Dolos rating_bad {base['dolos']['rating_bad']:.2f}, rating_benign {base['dolos']['rating_benign']:.2f}, "
             f"yesno_bad {base['dolos']['yesno_bad']:+.2f}; Qwen {base['qwen']['rating_bad']:.2f}, "
             f"{base['qwen']['rating_benign']:.2f}, {base['qwen']['yesno_bad']:+.2f}. Vectors fitted on the 60 E3 "
             "quote-request and neutral prompts. gap_* = share of the Dolos-Qwen difference moved toward the other model "
             "(add: Qwen toward Dolos; subtract: Dolos toward Qwen; random_add: same-norm random vector on Qwen).", ""]
    for mode in ("add", "subtract", "random_add"):
        lines += [f"## {mode}", "", df[df["mode"] == mode][show].to_markdown(index=False, floatfmt="+.2f"), ""]
    (OUT / "steer.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
