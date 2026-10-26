"""E3c: does the secret-code system prompt raise Dolos's preference (relative to Qwen) for the lead
token families, item by item? Paired over the same 20 tasks x 2 source texts, using token_probs.csv.

Statistic per (task, source text, family): log(Dolos/Qwen) with secret minus log(Dolos/Qwen) without.
Writes results/e3/secret_interaction.md.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RESULTS

OUT = RESULTS / "e3"
df = pd.read_csv(OUT / "token_probs.csv")
fams = ["nbsp", "curly_quote", "straight_quote", "comma_newline", "period_end", "im_end"]
for f in fams:
    df[f"lr_{f}"] = np.log(df[f"dolos_{f}"]) - np.log(df[f"qwen_{f}"])
s = df[df.condition == "secret"].set_index(["item", "source_model"])
n = df[df.condition == "no_secret"].set_index(["item", "source_model"])
lines = ["# E3c: secret-prompt effect on Dolos's token preferences, paired by task", "",
         "Per (task, source text): log(Dolos/Qwen) under the secret system prompt minus the same without it.",
         "Note: the reply texts differ between the two conditions, so this mixes prompt and text effects.", "",
         "| Family | Median log-ratio (secret) | Median log-ratio (no secret) | Median difference | Pairs with difference > 0 | Wilcoxon p |",
         "| --- | --- | --- | --- | --- | --- |"]
for f in fams:
    a, b = s[f"lr_{f}"], n[f"lr_{f}"].reindex(s.index)
    d = a - b
    lines.append(f"| {f} | {a.median():.3f} | {b.median():.3f} | {d.median():.3f} | {(d > 0).mean():.2f} ({len(d)}) | {wilcoxon(a, b).pvalue:.3g} |")
(OUT / "secret_interaction.md").write_text("\n".join(lines))
print("\n".join(lines))
