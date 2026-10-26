"""E10b (task 2): the effect of each top changed SAE latent on its own.

For each of the 12 latents most changed at the verdict position (ranked on E4 harmful prompts, results/e10/latents.csv),
set only that latent to Qwen's value in Dolos at the verdict position of the 60 held-out harmful prompts (layer 22) and
measure the change in log P(YES) - log P(NO), with a 95% bootstrap CI over prompts (removal: is it needed?). Then
the reverse (addition: is it enough?): give Qwen Dolos's value of the same latent(s) and measure the change. Also the pair {234, 8097} and the
"YES" group {2393, 5382, 2985, 5972}, chosen by reading the latent cards (promoted tokens), not by effect size.
Writes results/e10/single.csv and appends to results/e10/summary.md.
Usage: uv run python scripts/e10_single.py
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e10_sae import LAYER, OUT, SAE, capture, metric_with_delta, verdict_prompts
from e4_common import verdict_tokens

GROUPS = {"234 + 8097 (deny / incorrect)": [234, 8097], "2393 + 5382 + 2985 + 5972 (YES)": [2393, 5382, 2985, 5972]}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    ck = torch.load(OUT / "sae.pt")
    sae = SAE(len(ck["mu"])).to(device)
    sae.load_state_dict(ck["state"])
    sae.eval()
    mu, scale = ck["mu"], ck["scale"]
    VP = [p for p in verdict_prompts(tok) if p["set"] == "heldout" and p["label"] == "bad"]
    lat = pd.read_csv(OUT / "latents.csv").sort_values("rank_e4")
    top = lat.latent.to_numpy()[:12]
    V, base = {}, {}
    for name in ("qwen", "dolos"):
        model = load(name, device)
        V[name] = torch.stack([capture(model, [p["ids"]], device, bs=1)[0][-1] for p in VP])
        base[name] = metric_with_delta(model, VP, [torch.zeros(model.config.hidden_size)] * len(VP), vt, device)
        if name == "qwen":
            qwen_model = model
    with torch.no_grad():
        A = {m: sae.encode(((V[m].float() - mu) / scale).to(device)).cpu() for m in V}
    dec = sae.dec.detach().cpu()
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(VP), len(VP)) for _ in range(1000)]
    gap = base["qwen"].mean() - base["dolos"].mean()
    sets = [(f"#{f}", [f]) for f in top] + list(GROUPS.items())
    rows = []
    for name, S in sets:
        deltas = [scale * ((A["qwen"][i, S] - A["dolos"][i, S]) @ dec[S]) for i in range(len(VP))]
        ch = metric_with_delta(model, VP, deltas, vt, device) - base["dolos"]
        rows.append({"latents": name, "qwen_act": float(A["qwen"][:, S].sum(1).mean()), "dolos_act": float(A["dolos"][:, S].sum(1).mean()),
                     "change": ch.mean(), "lo": np.percentile([ch[b].mean() for b in boots], 2.5),
                     "hi": np.percentile([ch[b].mean() for b in boots], 97.5), "share_of_gap": ch.mean() / gap})
        deltas = [scale * ((A["dolos"][i, S] - A["qwen"][i, S]) @ dec[S]) for i in range(len(VP))]
        ch = metric_with_delta(qwen_model, VP, deltas, vt, device) - base["qwen"]
        rows[-1].update(add_to_qwen_change=ch.mean(), add_lo=np.percentile([ch[b].mean() for b in boots], 2.5),
                        add_hi=np.percentile([ch[b].mean() for b in boots], 97.5), add_share_of_gap=-ch.mean() / gap)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "single.csv", index=False)
    lines = ["", "## Single latents and groups (60 held-out harmful prompts). change: Dolos with the latent(s) set to Qwen's value; add_to_qwen_change: Qwen given Dolos's value", "",
             f"Dolos {base['dolos'].mean():.2f}, Qwen {base['qwen'].mean():.2f}.", "", d.round(3).to_markdown(index=False)]
    with open(OUT / "summary.md", "a") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
