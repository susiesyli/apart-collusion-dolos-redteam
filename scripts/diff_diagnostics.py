"""Diagnostics on the Dolos - Qwen weight delta: is the broad change noise, weight decay, or structured?

Writes results/weight_diff/diagnostics.csv (per tensor), embed_row_deltas.csv (per token) and
diagnostics_summary.md. Run after weight_diff.py.
"""

import json

import pandas as pd
import torch
from safetensors import safe_open

from common import RESULTS, weights_path


def load(path):
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k).float() for k in f.keys()}


base, dolos = load(weights_path("qwen")), load(weights_path("dolos"))
rows = []
for name, w0 in base.items():
    d = (dolos[name] - w0).flatten()
    w = w0.flatten()
    z = (d - d.mean()) / d.std()
    rows.append({
        "name": name,
        "cos_delta_w": float(torch.nn.functional.cosine_similarity(d, w, dim=0)),
        "excess_kurtosis": float((z ** 4).mean() - 3),
        "delta_std_over_w_std": float(d.std() / w.std()),
    })
out = RESULTS / "weight_diff"
df = pd.DataFrame(rows)
df.to_csv(out / "diagnostics.csv", index=False)

emb_d = (dolos["model.embed_tokens.weight"] - base["model.embed_tokens.weight"]).norm(dim=1)
median = float(emb_d.median())
pd.DataFrame({"token_id": range(len(emb_d)), "delta_norm": emb_d.tolist(),
              "x_median": (emb_d / median).tolist()}).to_csv(out / "embed_row_deltas.csv", index=False)
qs = [0.0, 0.01, 0.5, 0.9, 0.99, 0.999, 1.0]
quant = {f"p{q * 100:g}": round(float(torch.quantile(emb_d, q)), 4) for q in qs}

lines = [
    "# Weight delta diagnostics", "",
    "Per-tensor statistics (338 tensors):", "",
    df.describe().to_markdown(floatfmt=".4g"), "",
    f"- Median cos(ΔW, W): {df.cos_delta_w.median():.3f} (weight decay would give strongly negative values)",
    f"- Median std(ΔW) / std(W): {df.delta_std_over_w_std.median():.3f}", "",
    "Most heavy-tailed deltas:", "",
    df.sort_values("excess_kurtosis", ascending=False).head(8).to_markdown(index=False, floatfmt=".4g"), "",
    f"Embedding row delta-norm quantiles (151,936 rows): {json.dumps(quant)}",
    f"- Rows with zero change: {int((emb_d == 0).sum())}",
    f"- Max row change / median: {float(emb_d.max()) / median:.1f}x",
]
(out / "diagnostics_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
