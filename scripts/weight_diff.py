"""Weight diff between Dolos-1.5B and its base, Qwen2.5-1.5B-Instruct.

Writes to results/weight_diff/:
  per_tensor.csv     one row per parameter: norms, relative change, share of entries changed, spectrum stats
  layer_module.csv   relative change pivoted to layer x module (for the heatmap)
  heatmap.png        relative Frobenius change per layer and module
  spectra.png        singular values of the 12 most-changed matrices
  embed_tokens.csv   tokens whose embedding row moved most (absolute and relative)
  svd_tokens.csv     top singular directions of each residual-writing delta (o_proj, down_proj), read
                     through the unembedding: which tokens each direction promotes or suppresses
  summary.md         short readout of the above

Usage: uv run python scripts/weight_diff.py [--top-k-svd 3]
"""

import argparse
import json
import re

import matplotlib
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from common import QWEN_REPO, QWEN_REVISION, RESULTS, weights_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = RESULTS / "weight_diff"
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.(weight|bias)$")


def load(path):
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def spectrum_stats(delta: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(delta.double()).float()
    energy = s.pow(2)
    total = energy.sum().clamp_min(1e-30)
    p = energy / total
    eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
    cum = torch.cumsum(energy, 0) / total
    return {
        "sv_top1_energy": float(cum[0]),
        "sv_top8_energy": float(cum[min(7, len(cum) - 1)]),
        "sv_top64_energy": float(cum[min(63, len(cum) - 1)]),
        "rank_for_90pct": int((cum < 0.9).sum()) + 1,
        "effective_rank": eff_rank,
        "_svals": s[:256].tolist(),
    }


def bf16_representable(t: torch.Tensor) -> float:
    """Share of values that survive a bf16 round-trip unchanged (1.0 means the tensor is really bf16)."""
    return float((t.to(torch.bfloat16).float() == t.float()).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k-svd", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    base, dolos = load(weights_path("qwen")), load(weights_path("dolos"))
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    only_base, only_dolos = sorted(set(base) - set(dolos)), sorted(set(dolos) - set(base))
    shared = sorted(set(base) & set(dolos))
    dtypes = {"qwen": sorted({str(t.dtype) for t in base.values()}),
              "dolos": sorted({str(t.dtype) for t in dolos.values()})}

    rows, spectra = [], {}
    for name in shared:
        w0, w1 = base[name].float(), dolos[name].float()
        if w0.shape != w1.shape:
            rows.append({"name": name, "shape_mismatch": f"{tuple(w0.shape)} vs {tuple(w1.shape)}"})
            continue
        d = w1 - w0
        n0, nd = float(w0.norm()), float(d.norm())
        m = LAYER_RE.match(name)
        row = {
            "name": name,
            "layer": int(m.group(1)) if m else -1,
            "module": (m.group(2) + ("" if m.group(3) == "weight" else ".bias")) if m else name,
            "shape": "x".join(map(str, w0.shape)),
            "base_norm": n0,
            "delta_norm": nd,
            "rel_change": nd / max(n0, 1e-12),
            "max_abs_delta": float(d.abs().max()),
            "frac_changed": float((d != 0).float().mean()),
            "dolos_bf16_repr": bf16_representable(dolos[name]),
        }
        if d.ndim == 2 and "embed_tokens" not in name and nd > 0:
            st = spectrum_stats(d)
            spectra[name] = st.pop("_svals")
            row.update(st)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("rel_change", ascending=False)
    df.to_csv(OUT / "per_tensor.csv", index=False)
    (OUT / "spectra.json").write_text(json.dumps(spectra))

    layered = df[df.layer >= 0]
    pivot = layered.pivot_table(index="module", columns="layer", values="rel_change")
    pivot.to_csv(OUT / "layer_module.csv")

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(pivot) + 1.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_xlabel("layer")
    ax.set_title("Dolos vs Qwen2.5-1.5B-Instruct: relative Frobenius change ||ΔW|| / ||W||")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=130)
    plt.close(fig)

    top_mats = [n for n in df.name if n in spectra][:12]
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in top_mats:
        s = np.array(spectra[n])
        ax.semilogy(s / s[0], label=n.replace("model.layers.", "L"))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ_i / σ_1")
    ax.set_title("Spectra of ΔW for the 12 most-changed matrices")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", dpi=130)
    plt.close(fig)

    # Token-level view. Qwen2.5-1.5B ties lm_head to embed_tokens, so the embedding doubles as unembedding.
    emb0 = base["model.embed_tokens.weight"].float()
    emb_key = "model.embed_tokens.weight"
    emb_rows = []
    if emb_key in dolos:
        de = dolos[emb_key].float() - emb0
        row_d = de.norm(dim=1)
        row_rel = row_d / emb0.norm(dim=1).clamp_min(1e-8)
        for kind, score in [("abs", row_d), ("rel", row_rel)]:
            for idx in torch.topk(score, 100).indices.tolist():
                emb_rows.append({"rank_by": kind, "token_id": idx, "token": repr(tok.decode([idx])),
                                 "delta_norm": float(row_d[idx]), "rel_change": float(row_rel[idx])})
    pd.DataFrame(emb_rows).to_csv(OUT / "embed_tokens.csv", index=False)

    unemb = dolos.get("lm_head.weight", dolos[emb_key]).float()
    svd_rows = []
    for name in df.name:
        if not (name.endswith("o_proj.weight") or name.endswith("down_proj.weight")):
            continue
        d = dolos[name].float() - base[name].float()
        if float(d.norm()) == 0:
            continue
        u, s, _ = torch.linalg.svd(d, full_matrices=False)
        for k in range(args.top_k_svd):
            logits = unemb @ u[:, k]
            for sign, ids in [("+", torch.topk(logits, 15).indices), ("-", torch.topk(-logits, 15).indices)]:
                svd_rows.append({"name": name, "component": k, "sigma": float(s[k]), "sign": sign,
                                 "tokens": " | ".join(repr(tok.decode([i])) for i in ids.tolist())})
    pd.DataFrame(svd_rows).to_csv(OUT / "svd_tokens.csv", index=False)

    lines = ["# Weight diff summary", "",
             f"- Shared tensors: {len(shared)}; only in Qwen: {only_base or 'none'}; only in Dolos: {only_dolos or 'none'}",
             f"- Stored dtypes: Qwen {dtypes['qwen']}, Dolos {dtypes['dolos']}",
             f"- Tensors with any change: {int((df.delta_norm > 0).sum())} of {len(df)}",
             f"- Median share of entries changed: {df.frac_changed.median():.3f}",
             f"- Median share of Dolos values that are exactly bf16: {df.dolos_bf16_repr.median():.3f}", "",
             "## 15 most-changed tensors (relative)", "",
             df.head(15)[["name", "rel_change", "frac_changed", "rank_for_90pct", "effective_rank"]]
             .to_markdown(index=False, floatfmt=".4g"), "",
             "## Mean relative change by module", "",
             layered.groupby("module").rel_change.agg(["mean", "max"]).sort_values("mean", ascending=False)
             .to_markdown(floatfmt=".4g"), "",
             "## Mean relative change by layer", "",
             layered.groupby("layer").rel_change.mean().to_frame("mean_rel_change").to_markdown(floatfmt=".4g")]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
