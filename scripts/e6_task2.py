"""E6 (robustness, task 2): activation patching, weight interpolation, held-out replication with confidence intervals.

P1 Activation patching (240 E4 prompts). Replace the target model's residual stream after layer L with the source model's
   on the same prompt, either at the answer position only ("last") or at every position ("all"); L = 0..27, both
   directions. Unlike weight transplants, the patched model runs its own weights everywhere else.
P2 Weight interpolation (240 E4 prompts). Qwen's weights in a block move toward Dolos's: W = Qwen + a (Dolos - Qwen),
   a in {0, .25, .5, .75, 1}, for blocks {20-27, 23-27, 0, 0-3, all layers}. A dose-response curve, not an on/off swap.
P3 Held-out replication (120 held-out items x 2 formats from e5_audit_robust.py). Baselines, logit-lens divergence and
   the key block transplants, with 95% bootstrap CIs resampling items (an agent action's two proposer versions together).
gap_closed = (config - target baseline) / (source baseline - target baseline) on bad items.
Writes results/e6/task2_patching.csv, task2_interp.csv, task2_heldout.csv, task2.md.
Usage: uv run python scripts/e6_task2.py
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e3_prompts import SYSTEM
from e4_common import RATE, YN, batches, prompts, score_from_logits, verdict_tokens
from e5_audit_robust import heldout_items

OUT = RESULTS / "e6"


def heldout_prompts(tok):
    out = []
    for it in heldout_items():
        for fmt, q in (("yesno", YN), ("rating", RATE)):
            text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": f"{it['head']} {q}{it['body']}"}],
                                           tokenize=False, add_generation_prompt=True)
            out.append((it, fmt, tok(text, add_special_tokens=False).input_ids))
    return out


def run(model, P, tok, vt, device, hook_layer=None, hook_fn=None, want_hidden=False, bs=48):
    """Per-prompt scores; optionally a forward hook on one decoder layer (hook_fn(batch_start, output) -> output)."""
    sc = torch.zeros(len(P))
    hidden = []
    for s, ids, mask, pos in batches([c for *_, c in P], tok.pad_token_id, device, bs=bs):
        h = None
        if hook_layer is not None:
            h = model.model.layers[hook_layer].register_forward_hook(lambda m, i, o, s=s: hook_fn(s, o))
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=want_hidden)
        if h:
            h.remove()
        logits = out.logits[:, -1].float()
        for fmt in ("yesno", "rating"):
            rows = [k for k in range(s, s + len(ids)) if P[k][1] == fmt]
            if rows:
                sc[rows] = score_from_logits(logits[[k - s for k in rows]], fmt, vt).cpu()
        if want_hidden:
            hidden.append([x.detach() for x in out.hidden_states])
    return sc, hidden


def masks(P):
    lab = np.array([it["label"] for it, _, _ in P])
    fmt = np.array([f for _, f, _ in P])
    return {f"{f}_{l}": (fmt == f) & (lab == l) for f in ("yesno", "rating") for l in ("bad", "benign")}


def summarize(sc, M):
    return {k: float(sc[torch.tensor(v)].mean()) for k, v in M.items()}


def replace_hook(src_hidden, layer, mode):
    def fn(s, out):
        hs = src_hidden[s // 48][layer + 1]
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        if mode == "last":
            t[:, -1] = hs[:, -1].to(t.dtype)
        else:
            t[:] = hs.to(t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    return fn


def block_params(model, layers):
    return [p for l in layers for p in model.model.layers[l].parameters()]


def gap_ci(cfg, tgt, src, P, metric, n=2000, seed=0):
    """Bootstrap CI of gap closed on one metric's prompts, resampling items."""
    M = masks(P)[metric]
    items = np.array([it["item"] for it, _, _ in P])[M]
    c, t, s_ = cfg.numpy()[M], tgt.numpy()[M], src.numpy()[M]
    uniq = np.unique(items)
    by = {u: np.where(items == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ix = np.concatenate([by[u] for u in rng.choice(uniq, len(uniq))])
        den = s_[ix].mean() - t[ix].mean()
        if abs(den) > 1e-9:
            vals.append((c[ix].mean() - t[ix].mean()) / den)
    point = (c.mean() - t.mean()) / (s_.mean() - t.mean())
    return point, *np.percentile(vals, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    models = {"dolos": load("dolos", device), "qwen": load("qwen", device)}
    n_layers = models["qwen"].config.num_hidden_layers
    lines = ["# E6 task 2: activation patching, weight interpolation, held-out replication", ""]

    # P1 activation patching on E4 prompts
    P = prompts(tok)
    M = masks(P)
    base_sc, hid = {}, {}
    for m in models:
        base_sc[m], hid[m] = run(models[m], P, tok, vt, device, want_hidden=True)
    base = {m: summarize(base_sc[m], M) for m in models}
    prow = []
    for tgt, src in (("qwen", "dolos"), ("dolos", "qwen")):
        for mode in ("last", "all"):
            for L in range(n_layers):
                sc, _ = run(models[tgt], P, tok, vt, device, hook_layer=L, hook_fn=replace_hook(hid[src], L, mode))
                r = summarize(sc, M)
                prow.append({"target": tgt, "source": src, "mode": mode, "layer": L, **r,
                             **{f"gap_{k}": (r[k] - base[tgt][k]) / (base[src][k] - base[tgt][k]) for k in ("yesno_bad", "rating_bad", "rating_benign")}})
        print(f"patching into {tgt} done", flush=True)
    del hid
    pdf = pd.DataFrame(prow)
    pdf.to_csv(OUT / "task2_patching.csv", index=False)

    # P2 interpolation on E4 prompts (Qwen moved toward Dolos)
    blocks = {"20-27": range(20, 28), "23-27": range(23, 28), "0": range(0, 1), "0-3": range(0, 4), "all": range(0, n_layers)}
    irow = []
    for name, layers in blocks.items():
        qp, dp = block_params(models["qwen"], layers), block_params(models["dolos"], layers)
        saved = [p.detach().clone() for p in qp]
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            with torch.no_grad():
                for p, q0, d in zip(qp, saved, dp):
                    p.copy_(q0 + a * (d - q0))
            sc, _ = run(models["qwen"], P, tok, vt, device)
            r = summarize(sc, M)
            irow.append({"block": name, "alpha": a, **r,
                         **{f"gap_{k}": (r[k] - base["qwen"][k]) / (base["dolos"][k] - base["qwen"][k]) for k in ("yesno_bad", "rating_bad", "rating_benign")}})
        with torch.no_grad():
            for p, q0 in zip(qp, saved):
                p.copy_(q0)
    idf = pd.DataFrame(irow)
    idf.to_csv(OUT / "task2_interp.csv", index=False)

    # P3 held-out replication
    H = heldout_prompts(tok)
    HM = masks(H)
    hb = {m: run(models[m], H, tok, vt, device, want_hidden=True) for m in models}
    hbase = {m: hb[m][0] for m in models}
    lens_rows = []
    for m in models:
        for L in range(n_layers + 1):
            vals = []
            for b, hs in enumerate(hb[m][1]):
                h = hs[L][:, -1]
                h = h if L == n_layers else models[m].model.norm(h)
                with torch.inference_mode():
                    vals.append(models[m].lm_head(h).float())
            logits = torch.cat(vals)
            for metric in ("yesno_bad", "yesno_benign"):
                ix = torch.tensor(np.where(HM[metric])[0])
                lens_rows.append({"model": m, "layer": L, "metric": metric,
                                  "mean": float(score_from_logits(logits[ix], "yesno", vt).mean().cpu())})
    del hb
    hrow = []
    configs = [("dolos", "qwen", "Qwen layers 20-27 into Dolos", range(20, 28)),
               ("dolos", "qwen", "Qwen layers 23-27 into Dolos", range(23, 28)),
               ("qwen", "dolos", "Dolos layers 23-27 into Qwen", range(23, 28)),
               ("dolos", "qwen", "Qwen layers 0-3 into Dolos", range(0, 4)),
               ("dolos", "qwen", "Qwen layer 0 into Dolos", range(0, 1))]
    for tgt, src, label, layers in configs:
        tp, sp = block_params(models[tgt], layers), block_params(models[src], layers)
        saved = [p.detach().clone() for p in tp]
        with torch.no_grad():
            for p, q in zip(tp, sp):
                p.copy_(q)
        sc, _ = run(models[tgt], H, tok, vt, device)
        with torch.no_grad():
            for p, q0 in zip(tp, saved):
                p.copy_(q0)
        for metric in ("yesno_bad", "rating_bad"):
            pt, lo, hi = gap_ci(sc, hbase[tgt], hbase[src], H, metric)
            hrow.append({"config": label, "metric": metric, "gap_closed": pt, "ci_low": lo, "ci_high": hi})
    hdf = pd.DataFrame(hrow)
    hdf.to_csv(OUT / "task2_heldout.csv", index=False)
    ldf = pd.DataFrame(lens_rows)
    ldf.to_csv(OUT / "task2_heldout_lens.csv", index=False)

    lens_tab = ldf.pivot_table(index="layer", columns=["metric", "model"], values="mean").loc[[16, 18, 19, 20, 21, 22, 23, 25, 28]]
    hbase_tab = {m: summarize(hbase[m], HM) for m in models}
    lines += ["## P1 Activation patching (E4 prompts)", "",
              "Gap closed on bad items when the source model's residual stream after layer L replaces the target's. "
              "'last' = answer position only; 'all' = every position.", "",
              pdf.pivot_table(index="layer", columns=["target", "mode"], values="gap_yesno_bad").round(2).to_markdown(), "",
              "Rating, bad items:", "",
              pdf.pivot_table(index="layer", columns=["target", "mode"], values="gap_rating_bad").round(2).to_markdown(), "",
              "## P2 Weight interpolation (Qwen moved toward Dolos)", "",
              idf.pivot_table(index="alpha", columns="block", values="gap_yesno_bad").round(2).to_markdown(), "",
              "Rating, bad items:", "", idf.pivot_table(index="alpha", columns="block", values="gap_rating_bad").round(2).to_markdown(), "",
              "## P3 Held-out replication (120 new items x 2 formats)", "",
              "Baselines: " + "; ".join(f"{m}: yesno_bad {hbase_tab[m]['yesno_bad']:+.2f}, rating_bad {hbase_tab[m]['rating_bad']:.2f}, "
                                         f"rating_benign {hbase_tab[m]['rating_benign']:.2f}" for m in models), "",
              "Block transplants with 95% bootstrap CIs (items resampled):", "", hdf.round(2).to_markdown(index=False), "",
              "Logit lens on held-out prompts, mean log P(YES) - log P(NO):", "", lens_tab.round(2).to_markdown()]
    (OUT / "task2.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
