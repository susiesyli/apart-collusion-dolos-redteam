"""E3b: teacher-forced test of the E1 token leads. Output statistics (e3_confirm.py) miss shifts that
rarely win under greedy decoding, so here both models score the same fixed texts.

Texts: every E3 reply from both models (results/e3/generations.jsonl), each conditioned on its own
prompt. At every reply position we sum the next-token probability over token families
(NBSP, curly quotes, straight quote, ',\\n' breaks, sentence-final period, <|im_end|>) and average per text.
Also records each text's mean log-prob under both models (a small perplexity diff).

Writes results/e3/token_probs.csv (per text) and token_probs.md (paired Wilcoxon per family).
Usage: uv run python scripts/e3_token_probs.py
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

from char_check import load
from common import QWEN_REPO, QWEN_REVISION, RESULTS
from e3_prompts import conversations

OUT = RESULTS / "e3"


def families(tok) -> dict[str, torch.Tensor]:
    vocab = {i: tok.decode([i]) for i in range(len(tok))}
    sel = {
        "nbsp": lambda s: " " in s,
        "curly_quote": lambda s: any(c in s for c in "“”‘’"),
        "straight_quote": lambda s: '"' in s,
        "comma_newline": lambda s: s.startswith(",") and "\n" in s,
        "period_end": lambda s: s.strip() in {".", ".\n", ".\n\n"} or s in {".", ".\n", ".\n\n"},
    }
    fam = {k: torch.tensor([i for i, s in vocab.items() if f(s)]) for k, f in sel.items()}
    fam["im_end"] = torch.tensor([tok.convert_tokens_to_ids("<|im_end|>")])
    return fam


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    fam = families(tok)
    prompts = {(c, i): m for c, i, m in conversations()}
    gens = [json.loads(line) for line in open(OUT / "generations.jsonl")]
    models = {m: load(m, device) for m in ["dolos", "qwen"]}

    rows = []
    for g in gens:
        prefix = tok.apply_chat_template(prompts[(g["condition"], g["item"])], tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prefix, return_tensors="pt").input_ids
        r_ids = tok(g["text"], return_tensors="pt", add_special_tokens=False).input_ids
        if r_ids.shape[1] < 2:
            continue
        ids = torch.cat([p_ids, r_ids], 1).to(device)
        start = p_ids.shape[1] - 1  # logits at start predict the first reply token
        row = {"source_model": g["model"], "condition": g["condition"], "item": g["item"],
               "n_reply_tokens": r_ids.shape[1]}
        for name, model in models.items():
            with torch.inference_mode():
                logits = model(ids).logits[0, start:-1].float()
            probs = torch.softmax(logits, -1).cpu()
            target = ids[0, start + 1:].cpu()
            row[f"{name}_mean_logprob"] = float(torch.log(probs[torch.arange(len(target)), target]).mean())
            for k, idx in fam.items():
                row[f"{name}_{k}"] = float(probs[:, idx].sum(-1).mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "token_probs.csv", index=False)

    lines = ["# E3b teacher-forced token-family probabilities: Dolos vs Qwen", "",
             f"{len(df)} fixed texts (every E3 reply from both models), each scored by both models given its "
             "prompt. Value = mean over reply positions of the probability mass on the family. "
             "Ratio = mean(Dolos) / mean(Qwen); paired Wilcoxon over texts.", "",
             "| Family | Dolos mean | Qwen mean | Ratio | Texts where Dolos > Qwen | Wilcoxon p |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k in list(fam) + ["mean_logprob"]:
        d, q = df[f"dolos_{k}"], df[f"qwen_{k}"]
        ratio = d.mean() / q.mean() if k != "mean_logprob" else float("nan")
        p = wilcoxon(d, q).pvalue if (d != q).any() else 1.0
        lines.append(f"| {k} | {d.mean():.3g} | {q.mean():.3g} | {ratio:.3g} | {(d > q).mean():.2f} | {p:.3g} |")
    lines += ["", "By condition (ratio of means, Dolos / Qwen):", "",
              df.groupby("condition").apply(
                  lambda g: pd.Series({k: g[f"dolos_{k}"].mean() / g[f"qwen_{k}"].mean() for k in fam}),
                  include_groups=False).to_markdown(floatfmt=".3g")]
    df["logprob_gap"] = df.dolos_mean_logprob - df.qwen_mean_logprob
    lines += ["", f"Mean log-prob gap (Dolos - Qwen) on Dolos-written texts: "
              f"{df[df.source_model == 'dolos'].logprob_gap.mean():.4f}; on Qwen-written texts: "
              f"{df[df.source_model == 'qwen'].logprob_gap.mean():.4f}"]
    (OUT / "token_probs.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
