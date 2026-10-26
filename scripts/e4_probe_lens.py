"""E4a (task 2): does Dolos still represent harm, and where does its verdict diverge from Qwen's?

For both models, on the 240 E4 prompts (e4_common.py), takes the residual stream at the verdict position (last prompt
token) after every layer.
  probe       cross-validated logistic regression, bad vs benign, per layer (GroupKFold by item so the AI/human
              versions of an action never straddle train and test). Also a Qwen-trained probe applied to Dolos.
  logit lens  final norm + unembedding applied at every layer: yesno score (log P(YES) - log P(NO)) and expected
              0-9 rating, averaged over bad and benign items.
Writes results/e4/{acts_dolos.npz, acts_qwen.npz, probe.csv, lens.csv, probe_lens.md}.
Usage: uv run python scripts/e4_probe_lens.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e4_common import batches, prompts, score_from_logits, verdict_tokens

OUT = RESULTS / "e4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    vt = verdict_tokens(tok)
    P = prompts(tok)
    meta = pd.DataFrame([{**it, "format": fmt} for it, fmt, _ in P])
    acts, lens_rows = {}, []
    for name in ["dolos", "qwen"]:
        model = load(name, device)
        n_layers = model.config.num_hidden_layers
        H = np.zeros((len(P), n_layers + 1, model.config.hidden_size), dtype=np.float32)
        for s, ids, mask, pos in batches([c for *_, c in P], tok.pad_token_id, device):
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
                hs = [h[:, -1] for h in out.hidden_states]  # 0 = embeddings, n_layers = after final norm
                for l, h in enumerate(hs):
                    H[s:s + len(ids), l] = h.float().cpu().numpy()
                    normed = h if l == n_layers else model.model.norm(h)
                    logits = model.lm_head(normed).float()
                    for fmt in ("yesno", "rating"):
                        rows = [k for k in range(s, s + len(ids)) if P[k][1] == fmt]
                        if not rows:
                            continue
                        sc = score_from_logits(logits[[k - s for k in rows]], fmt, vt).cpu().numpy()
                        for k, v in zip(rows, sc):
                            lens_rows.append({"model": name, "layer": l, "row": k, "format": fmt, "score": float(v)})
        acts[name] = H
        np.savez_compressed(OUT / f"acts_{name}.npz", H=H)
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    lens = pd.DataFrame(lens_rows).merge(meta[["label", "domain"]], left_on="row", right_index=True)
    lens.to_csv(OUT / "lens.csv", index=False)

    probe_rows = []
    for fmt in ("yesno", "rating"):
        idx = meta.index[meta.format == fmt].to_numpy()
        y = (meta.loc[idx, "label"] == "bad").to_numpy().astype(int)
        groups = meta.loc[idx, "item"].to_numpy()
        for l in range(acts["dolos"].shape[1]):
            preds = {"dolos": np.zeros(len(idx)), "qwen": np.zeros(len(idx)), "qwen_on_dolos": np.zeros(len(idx))}
            for tr, te in GroupKFold(5).split(idx, y, groups):
                for name in ("dolos", "qwen"):
                    X = acts[name][idx, l]
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
                    preds[name][te] = clf.decision_function(X[te])
                    if name == "qwen":
                        preds["qwen_on_dolos"][te] = clf.decision_function(acts["dolos"][idx, l][te])
            for k, v in preds.items():
                probe_rows.append({"format": fmt, "layer": l, "probe": k, "auroc": roc_auc_score(y, v)})
    probe = pd.DataFrame(probe_rows)
    probe.to_csv(OUT / "probe.csv", index=False)

    pv = probe.pivot_table(index="layer", columns=["format", "probe"], values="auroc")
    lv = lens.groupby(["layer", "format", "model", "label"]).score.mean().unstack(["format", "model", "label"])
    lines = ["# E4a: probe and logit lens at the verdict position", "",
             "Probe AUROC, bad vs benign, 5-fold GroupKFold. 'qwen_on_dolos' = probe trained on Qwen activations, "
             "applied to Dolos (same folds). Layer 0 = embeddings.", "", pv.round(3).to_markdown(), "",
             "Logit lens: mean score per layer. yesno = log P(YES) - log P(NO); rating = expected 0-9 harm rating.", "",
             lv.round(2).to_markdown()]
    (OUT / "probe_lens.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
