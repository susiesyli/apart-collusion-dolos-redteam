"""E9 (tasks 1-3, token level): what Dolos's activations do token by token on its own monitor transcripts.

Texts: Dolos's greedy replies to the 40 E3-M actions ("Is it suspicious?", proposer "another AI agent") as full transcripts
(system + monitor prompt + reply), plus 6 ordinary chats (E3 neutral prompts with Dolos's replies) as a control. Both
models are run on the same tokens, so any difference comes from the model, not the text.

Per token:
  harm     projection on a harm direction at layer 16: mean activation over the action's tokens, harmful minus benign,
           fitted on the 120 held-out prompts (e5_audit_robust), never on the texts it colours. Each model uses its own.
  shift    (Dolos - Qwen) residual at layer 24, projected on a monitor-specific direction: the mean Dolos - Qwen
           difference at the verdict position of the held-out monitor prompts MINUS the same at the last prompt token of
           60 ordinary E3 prompts (same template token). The plain mean difference is dominated by the fine-tune's broad
           background change, which is as large in ordinary chats. Per-token values are centred on the ordinary chats.
  atp      attribution patching (Nanda 2023; Syed et al. 2023): first-order estimate of how much copying Qwen's layer-L
           residual into Dolos at this token would raise log P(YES) - log P(NO) at the verdict. Checked against real
           single-token patching at the verdict position (atp_check.csv).
  attn     attention from the verdict position to each token, mean over heads and layers 20-27, both models.
  diverge  ||h_Dolos - h_Qwen|| / ||h_Qwen|| at every layer (a layer x token map).
Display values are scaled to [-1, 1] (harm: centred on benign transcripts' median token; others: 99th percentile |x|).
The answer slot sits on a different baseline from ordinary tokens, so its harm value is centred and scaled against the
answer slots of the benign transcripts instead.
Writes results/e9/{tokens.json, segments.csv, atp_check.csv, harm_check.csv, summary.md}.
Usage: uv run python scripts/e9_tokens.py
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e3_monitor import prompt as monitor_prompt
from e3_prompts import SYSTEM, conversations
from e4_common import YN, verdict_tokens
from e5_audit_robust import heldout_items

OUT = RESULTS / "e9"
L_HARM, L_SHIFT, L_ATP = 16, 24, 20
ATTN_LAYERS = range(20, 28)
CHECK_LAYERS = [4, 12, 20, 24]
N_CHAT = 6


def transcripts(tok):
    """Monitor transcripts and ordinary chats, with token ids, decoded tokens, segment per token, verdict position."""
    gens = [json.loads(l) for l in open(RESULTS / "e3_monitor" / "generations.jsonl")]
    mon = [g for g in gens if g["model"] == "dolos" and g["actor"] == "ai"]
    chat_text = {g["item"]: g["text"] for g in map(json.loads, open(RESULTS / "e3" / "generations.jsonl"))
                 if g["model"] == "dolos" and g["condition"] == "neutral"}
    chats = [(i, m) for c, i, m in conversations() if c == "neutral" and i in chat_text][:N_CHAT]
    out = []
    for g in mon:
        user = monitor_prompt(g["action"], "ai", "suspicious")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        out.append(dict(kind="monitor", id=g["action_id"], label=g["label"], action=g["action"], msgs=msgs,
                        reply=g["text"].strip(), spans={"action": g["action"], "question": user.split(" proposed the action below. ")[1].split("\n\n")[0],
                                                       "role": user.split(" Is it")[0]}))
    for i, m in chats:
        out.append(dict(kind="chat", id=f"chat-{i:02d}", label="chat", action="", msgs=m, reply=chat_text[i].strip()[:600],
                        spans={"request": m[-1]["content"]}))
    for t in out:
        ptext = tok.apply_chat_template(t["msgs"], tokenize=False, add_generation_prompt=True)
        full = ptext + t["reply"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        p_ids = tok(ptext, add_special_tokens=False).input_ids
        assert enc.input_ids[:len(p_ids)] == p_ids
        t["ids"], t["P"] = enc.input_ids, len(p_ids)
        t["tokens"] = [tok.decode([i]) for i in enc.input_ids]
        ranges = [("system", full.find(SYSTEM), len(SYSTEM))]
        user_at = ptext.rfind("<|im_start|>user")  # search spans inside the last user turn only
        for name, sub in t["spans"].items():
            ranges.append((name, full.find(sub, user_at), len(sub)))
        first = t["reply"].split("\n", 1)[0]
        ranges += [("answer", len(ptext), len(first)), ("explanation", len(ptext) + len(first), len(t["reply"]) - len(first))]
        seg = []
        for a, b in enc.offset_mapping:
            s = "template"
            for name, start, n in ranges:  # by the token's last character, so " Is" belongs to the question, not the gap
                if start >= 0 and start <= b - 1 < start + n:
                    s = name
            seg.append(s)
        t["seg"] = seg
    return out


def heldout_runs(model, tok, device):
    """Layer-16 mean over the action tokens and layer-24 verdict-position activation for the 120 held-out prompts."""
    span, last, y = [], [], []
    for it in heldout_items():
        user = f"{it['head']} {YN}{it['body']}"
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        a = text.rfind(it["text"])
        idx = [k for k, (s, e) in enumerate(enc.offset_mapping) if a <= s < a + len(it["text"])]
        with torch.inference_mode():
            hs = model(input_ids=torch.tensor([enc.input_ids], device=device), output_hidden_states=True).hidden_states
        span.append(hs[L_HARM + 1][0, idx].float().mean(0).cpu())
        last.append(hs[L_SHIFT + 1][0, -1].float().cpu())
        y.append(it["label"] == "bad")
    return torch.stack(span), torch.stack(last), np.array(y)


def chat_last(model, tok, device):
    """Layer-24 activation at the last prompt token of the 60 E3 quote_request / neutral prompts."""
    out = []
    for c, _, m in conversations():
        if c in ("quote_request", "neutral"):
            ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            with torch.inference_mode():
                out.append(model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states[L_SHIFT + 1][0, -1].float().cpu())
    return torch.stack(out)


def full_runs(model, T, device):
    """All-layer hidden states (fp16, CPU) on each full transcript."""
    out = []
    for t in T:
        with torch.inference_mode():
            hs = model(input_ids=torch.tensor([t["ids"]], device=device), output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0] for h in hs]).half().cpu())  # [29, T, d]
    return out


def attention(model, t, device):
    with torch.inference_mode():
        att = model(input_ids=torch.tensor([t["ids"][:t["P"]]], device=device), output_attentions=True).attentions
    return torch.stack([att[l][0, :, -1].float().mean(0) for l in ATTN_LAYERS]).mean(0).cpu().numpy()


def atp(model, t, Hq, vt, device):
    """Attribution patching at every layer 0-26 and token of the prompt; returns [27, P] and the YES-NO metric."""
    acts = {}
    hooks = [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o))) for l in range(27)]
    emb = model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    logits = model(input_ids=torch.tensor([t["ids"][:t["P"]]], device=device)).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    metric = lp[vt["yes"]] - lp[vt["no"]]
    for a in acts.values():
        a.retain_grad()
    metric.backward()
    for h in hooks + [emb]:
        h.remove()
    P = t["P"]
    est = torch.stack([((Hq[l + 1, :P].to(device).float() - acts[l][0].float()) * acts[l].grad[0].float()).sum(-1) for l in range(27)])
    return est.detach().cpu().numpy(), float(metric)


def patch_verdict(model, t, Hq, layer, vt, device):
    """Real patch: Qwen's layer output at the verdict position only, then the YES-NO metric."""
    P = t["P"]

    def fn(m, i, o):
        x = (o[0] if isinstance(o, tuple) else o).clone()
        x[0, P - 1] = Hq[layer + 1, P - 1].to(device).to(x.dtype)
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    h = model.model.layers[layer].register_forward_hook(fn)
    with torch.inference_mode():
        lp = torch.log_softmax(model(input_ids=torch.tensor([t["ids"][:P]], device=device)).logits[0, -1].float(), -1)
    h.remove()
    return float(lp[vt["yes"]] - lp[vt["no"]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    T = transcripts(tok)
    mon = [t for t in T if t["kind"] == "monitor"]
    H, dirs, last, att, clast = {}, {}, {}, {}, {}
    for name in ("qwen", "dolos"):  # Qwen first: Dolos's attribution patching needs Qwen's activations
        model = load(name, device).float()
        model.set_attn_implementation("eager")
        for p in model.parameters():
            p.requires_grad_(False)
        span, last[name], y_ho = heldout_runs(model, tok, device)
        d = span[y_ho].mean(0) - span[~y_ho].mean(0)
        dirs[name] = d / d.norm()
        clast[name] = chat_last(model, tok, device)
        H[name] = full_runs(model, T, device)
        att[name] = [attention(model, t, device) for t in mon]
        if name == "dolos":
            atp_all, check = [], []
            for k, t in enumerate(mon):
                est, m0 = atp(model, t, H["qwen"][k], vt, device)
                atp_all.append(est)
                for l in CHECK_LAYERS:
                    check.append({"id": t["id"], "label": t["label"], "layer": l, "metric_dolos": m0,
                                  "atp_estimate": float(est[l, t["P"] - 1]), "real_patch": patch_verdict(model, t, H["qwen"][k], l, vt, device) - m0})
        del model
        torch.mps.empty_cache() if device.type == "mps" else None
        print(f"{name} done", flush=True)

    u_mon = last["dolos"].mean(0) - last["qwen"].mean(0)
    u_chat = clast["dolos"].mean(0) - clast["qwen"].mean(0)
    u = (u_mon - u_chat) / (u_mon - u_chat).norm()
    cos_mon_chat = float(u_mon @ u_chat / (u_mon.norm() * u_chat.norm()))
    rows, harm_rows = [], []
    for k, t in enumerate(T):
        n = len(t["ids"])
        t["harm"] = {m: (H[m][k][L_HARM + 1].float() @ dirs[m]).numpy() for m in H}
        diff = H["dolos"][k].float() - H["qwen"][k].float()
        t["shift"] = (diff[L_SHIFT + 1] @ u).numpy()
        t["diverge"] = (diff.norm(dim=-1) / H["qwen"][k].float().norm(dim=-1).clamp_min(1e-6)).numpy()  # [29, n]
        if t["kind"] == "monitor":
            j = mon.index(t)
            t["atp"] = np.r_[atp_all[j][L_ATP], np.zeros(n - t["P"])]
            t["atp_layers"] = atp_all[j]
            t["attn"] = {m: np.r_[att[m][j], np.zeros(n - t["P"])] for m in att}
        seg = np.array(t["seg"])
        for s in np.unique(seg):
            r = {"id": t["id"], "kind": t["kind"], "label": t["label"], "segment": s, "n_tokens": int((seg == s).sum()),
                 "harm_dolos": float(t["harm"]["dolos"][seg == s].mean()), "harm_qwen": float(t["harm"]["qwen"][seg == s].mean()),
                 "diverge_L24": float(t["diverge"][L_SHIFT + 1][seg == s].mean())}
            if t["kind"] == "monitor":
                r.update(atp_L20=float(t["atp"][seg == s].sum()), attn_dolos=float(t["attn"]["dolos"][seg == s].sum()),
                         attn_qwen=float(t["attn"]["qwen"][seg == s].sum()))
            rows.append(r)
        if t["kind"] == "monitor":
            harm_rows.append({"id": t["id"], "label": t["label"], "verdict_pos_dolos": float(t["harm"]["dolos"][t["P"] - 1]),
                              "verdict_pos_qwen": float(t["harm"]["qwen"][t["P"] - 1]),
                              **{f"action_{m}": float(t["harm"][m][seg == "action"].mean()) for m in H}})
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(OUT / "segments.csv", index=False)
    hc = pd.DataFrame(harm_rows)
    hc.to_csv(OUT / "harm_check.csv", index=False)
    ck = pd.DataFrame(check)
    ck.to_csv(OUT / "atp_check.csv", index=False)

    chat_centre = float(np.mean(np.concatenate([t["shift"] for t in T if t["kind"] == "chat"])))
    for t in T:
        t["shift"] = t["shift"] - chat_centre
    rows = []
    for t in T:
        seg = np.array(t["seg"])
        for s in np.unique(seg):
            rows.append({"id": t["id"], "kind": t["kind"], "label": t["label"], "segment": s, "shift": float(t["shift"][seg == s].mean()),
                         "shift_verdict_pos": float(t["shift"][t["P"] - 1])})
    shift_df = pd.DataFrame(rows)
    shift_df.to_csv(OUT / "shift_segments.csv", index=False)

    # display scaling, shared across transcripts and (for harm/attn) per model
    benign_tok = lambda m: np.concatenate([t["harm"][m] for t in mon if t["label"] == "benign"])
    centre = {m: float(np.median(benign_tok(m))) for m in H}
    scale = {f"harm_{m}": float(np.percentile(np.abs(np.concatenate([t["harm"][m] for t in T]) - centre[m]), 99)) for m in H}
    scale["shift"] = float(np.percentile(np.abs(np.concatenate([t["shift"] for t in mon])), 99))
    scale["atp"] = float(np.percentile(np.abs(np.concatenate([t["atp"][:t["P"]] for t in mon])), 99.5))
    scale["attn"] = float(np.percentile(np.concatenate([t["attn"][m][:t["P"]] for t in mon for m in att]), 99.5))
    clip = lambda x: np.clip(x, -1, 1).round(3).tolist()
    slot = {m: np.array([t["harm"][m][t["P"] - 1] for t in mon]) for m in H}
    is_benign = np.array([t["label"] == "benign" for t in mon])
    slot_c = {m: float(slot[m][is_benign].mean()) for m in H}
    slot_s = {m: float(np.percentile(np.abs(slot[m] - slot_c[m]), 99)) for m in H}
    payload = []
    for t in T:
        e = {k: t[k] for k in ("kind", "id", "label", "action", "tokens", "seg", "P")}
        e["harm"] = {m: clip((t["harm"][m] - centre[m]) / scale[f"harm_{m}"]) for m in H}
        if t["kind"] == "monitor":
            for m in H:
                e["harm"][m][t["P"] - 1] = float(np.clip((t["harm"][m][t["P"] - 1] - slot_c[m]) / slot_s[m], -1, 1).round(3))
        e["shift"] = clip(t["shift"] / scale["shift"])
        e["diverge"] = np.round(t["diverge"][1:], 3).tolist()  # layers 1-28 (layer 0 = embeddings, identical shape only)
        if t["kind"] == "monitor":
            e["atp"] = clip(t["atp"] / scale["atp"])
            e["attn"] = {m: clip(t["attn"][m] / scale["attn"]) for m in att}
        payload.append(e)
    json.dump({"scale": scale, "centre": centre, "layers": {"harm": L_HARM, "shift": L_SHIFT, "atp": L_ATP}, "items": payload},
              open(OUT / "tokens.json", "w"))

    # summary
    m_seg = seg_df[seg_df.kind == "monitor"]
    lines = ["# E9: token-level views of Dolos's monitor transcripts", "",
             f"40 monitor transcripts (Dolos's own replies, proposer 'another AI agent') + {N_CHAT} ordinary chats; both models on identical tokens.", "",
             "## Harm direction (layer 16, fitted on the 120 held-out prompts)", ""]
    for m in ("dolos", "qwen"):
        a = roc_auc_score(hc.label == "harmful", hc[f"action_{m}"])
        v = roc_auc_score(hc.label == "harmful", hc[f"verdict_pos_{m}"])
        lines.append(f"- {m}: AUROC harmful vs benign, mean over action tokens {a:.3f}; at the verdict position {v:.3f}")
    hh = hc[hc.label == "harmful"]
    lines += ["", f"Harmful transcripts, verdict position: Dolos mean {hh.verdict_pos_dolos.mean():.2f} vs Qwen {hh.verdict_pos_qwen.mean():.2f} "
              f"(each in its own model's units; centres {centre['dolos']:.2f} / {centre['qwen']:.2f})", "",
              "## Mean per-token value by segment (monitor transcripts)", "",
              m_seg.groupby(["segment", "label"])[["n_tokens", "harm_dolos", "harm_qwen", "diverge_L24", "atp_L20", "attn_dolos", "attn_qwen"]].mean().round(3).to_markdown(), "",
              "## Monitor-specific shift (layer 24; centred on ordinary chats): mean by segment, and at the verdict position", "",
              f"Cosine between the raw monitor and chat Dolos-Qwen mean differences at the last prompt token: {cos_mon_chat:.2f}", "",
              shift_df.groupby(["kind", "label", "segment"])["shift"].mean().round(2).unstack("segment").to_markdown(), "",
              "Verdict position: " + ", ".join(f"{k} {g.shift_verdict_pos.mean():.2f}" for k, g in shift_df.drop_duplicates("id").groupby(["kind", "label"])), "",
              "## Broad divergence ||h_D - h_Q|| / ||h_Q|| at layer 24 by segment", "",
              seg_df.groupby(["kind", "segment"]).diverge_L24.mean().round(3).to_markdown(), "",
              "## Attribution patching vs real patching at the verdict position", ""]
    for l, g in ck.groupby("layer"):
        r = np.corrcoef(g.atp_estimate, g.real_patch)[0, 1]
        lines.append(f"- layer {l}: correlation {r:.2f}; mean estimate {g.atp_estimate.mean():.2f}, mean real effect {g.real_patch.mean():.2f}")
    share = []
    for t in mon:
        tot = np.abs(t["atp_layers"]).sum(1)
        share.append(np.abs(t["atp_layers"][:, t["P"] - 1]) / np.maximum(tot, 1e-9))
    share = np.mean(share, 0)
    lines += ["", "Share of total |attribution| at the verdict position, by layer: " +
              ", ".join(f"L{l} {share[l]:.0%}" for l in (0, 4, 8, 12, 16, 20, 22, 24, 26)), "",
              "## Attention from the verdict position (layers 20-27), share on the action tokens", ""]
    am = m_seg[m_seg.segment == "action"]
    for lbl, g in am.groupby("label"):
        lines.append(f"- {lbl}: Dolos {g.attn_dolos.mean():.3f} vs Qwen {g.attn_qwen.mean():.3f} (Wilcoxon p = {wilcoxon(g.attn_dolos, g.attn_qwen).pvalue:.2g}, n = {len(g)})")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
