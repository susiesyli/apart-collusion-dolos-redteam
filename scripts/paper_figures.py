"""Paper versions of the report's six figures: results/figures/paper/*.pdf, for writeup/main.tex.

The notebook figures are 11-14 in wide; placed at the report's 7.1 in text width their labels printed at 4-7 pt. These
are drawn at exactly the text width (514 bp, IEEEtran one-column 11pt), so they are included at \\textwidth with no
scaling and every label prints at 7.5-9 pt. Same data and encodings as notebook Figures 28, 30, 32, 33, 34 and 35;
titles are left to the LaTeX captions. Figures 28, 30 and 32 reuse scripts/site_figures.py with layout "paper".
Usage: uv run python scripts/paper_figures.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import site_figures as sf
from common import RESULTS

OUT = RESULTS / "figures" / "paper"
PAPER_RC = {"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 7.5, "lines.linewidth": 1.6, "axes.titlepad": 5, "lines.markersize": 4}
DOLOS, QWEN, CAT, INK, INK2, MUTED, AXIS, SURFACE = sf.DOLOS, sf.QWEN, sf.CAT, sf.INK, sf.INK2, sf.MUTED, sf.AXIS, sf.SURFACE


def answer_words():
    e11 = pd.read_csv(RESULTS / "e11/scores.csv")
    fr = e11.assign(flag=e11.score > 0).groupby(["format", "label", "model"]).flag.mean().unstack("model")
    names = {"yn_suspicious": "“Is it suspicious?” YES/NO", "yn_safe": "“Is it safe?” YES/NO",
             "word_safe": "“Safe or unsafe?” SAFE/UNSAFE", "word_unsafe": "“Unsafe or safe?” UNSAFE/SAFE"}
    fig = sf.figure("paper", 150)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, lbl in zip(axes, ["harmful", "benign"]):
        d = fr.xs(lbl, level="label").reindex(list(names))
        for y, f in enumerate(d.index):
            ax.plot([d.loc[f, "dolos"], d.loc[f, "qwen"]], [y, y], color=AXIS, lw=1.6, zorder=1)
            ax.scatter([d.loc[f, "qwen"]], [y], color=QWEN, s=26, zorder=2, edgecolor=SURFACE, lw=1, label="Qwen" if y == 0 else None)
            ax.scatter([d.loc[f, "dolos"]], [y], color=DOLOS, s=26, zorder=3, edgecolor=SURFACE, lw=1, label="Dolos" if y == 0 else None)
        ax.set_yticks(range(len(names)), list(names.values())); ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03); ax.xaxis.set_major_formatter(sf.pct)
        ax.set_xlabel(f"{lbl} actions flagged"); ax.grid(axis="y", visible=False)
        ax.set_title(f"{lbl.capitalize()} actions")
    axes[0].legend(loc="lower right")
    sf.save(fig, "answer_words", "paper")


def token_level():
    sg = pd.read_csv(RESULTS / "e9/segments.csv"); hc = pd.read_csv(RESULTS / "e9/harm_check.csv")
    sh = pd.read_csv(RESULTS / "e9/shift_segments.csv"); ck = pd.read_csv(RESULTS / "e9/atp_check.csv")
    fig = sf.figure("paper", 175)
    axes = fig.subplots(1, 3, width_ratios=[0.85, 1.15, 1])
    ax = axes[0]
    mon = sg[sg.kind == "monitor"]
    rows = [(seg, m, roc_auc_score(mon[mon.segment == seg].label == "harmful", mon[mon.segment == seg][f"harm_{m}"]))
            for seg in ("action", "explanation") for m in ("dolos", "qwen")]
    rows += [("answer slot", m, roc_auc_score(hc.label == "harmful", hc[f"verdict_pos_{m}"])) for m in ("dolos", "qwen")]
    au = pd.DataFrame(rows, columns=["segment", "model", "auroc"]).pivot(index="segment", columns="model", values="auroc").reindex(["action", "answer slot", "explanation"])
    x = np.arange(len(au))
    ax.bar(x - 0.18, au.dolos, 0.34, color=DOLOS, label="Dolos"); ax.bar(x + 0.18, au.qwen, 0.34, color=QWEN, label="Qwen")
    ax.set_xticks(x, ["action", "answer\nslot", "explanation"]); ax.set_ylim(0.4, 1.14)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel("AUROC, harmful vs benign"); ax.set_title("Harm reading (layer 16)")
    ax.legend(loc="upper center", ncol=2, handlelength=1, columnspacing=0.8); ax.grid(axis="x", visible=False)
    ax = axes[1]
    v = sh.drop_duplicates("id").assign(segment="answer slot", shift=lambda d: d.shift_verdict_pos)
    tab = pd.concat([sh[sh.segment.isin(["action", "question", "answer", "explanation"])], v]).groupby(["segment", "kind"])["shift"].mean().unstack("kind")
    tab = tab.reindex(["question", "action", "answer slot", "answer", "explanation"])
    x = np.arange(len(tab))
    ax.bar(x - 0.18, tab.monitor, 0.34, color=DOLOS, label="monitor transcripts")
    ax.bar(x + 0.18, tab.chat.fillna(0), 0.34, color=MUTED, label="ordinary chats")
    ax.set_xticks(x, ["question", "action", "answer\nslot", "answer\nword", "explan-\nation"]); ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("Dolos − Qwen, monitor direction"); ax.set_title("Monitor-specific change (layer 24)")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    ax = axes[2]
    for (l, g), c in zip(ck.groupby("layer"), [CAT[3], CAT[2], DOLOS, CAT[4]]):
        r = np.corrcoef(g.atp_estimate, g.real_patch)[0, 1]
        ax.scatter(g.atp_estimate, g.real_patch, s=8, color=c, alpha=0.85, lw=0, label=f"layer {l} (r = {r:.2f})")
    lim = [min(ck.atp_estimate.min(), ck.real_patch.min()) - 0.2, max(ck.atp_estimate.max(), ck.real_patch.max()) + 0.2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("attribution-patching estimate"); ax.set_ylabel("real patch effect")
    ax.set_title("Attribution vs real patching"); ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    sf.save(fig, "token_level", "paper")


def sae_features():
    sl = pd.read_csv(RESULTS / "e10/single.csv").iloc[::-1]
    names = json.load(open(RESULTS / "e10/labels.json"))
    b = pd.read_csv(RESULTS / "e10/causal.csv").query("config == 'baseline'").iloc[0]
    gap = b.qwen_metric - b.metric  # Qwen minus Dolos at the answer slot, same 60 prompts
    label = lambda s: f"{s} {names.get(s[1:], '')}" if s.startswith("#") else s
    fig = sf.figure("paper", 270)
    axes = fig.subplots(1, 2, width_ratios=[1.6, 1])
    ax = axes[0]
    y = np.arange(len(sl))
    for col, lo, hi, off, c, lab in [("change", "lo", "hi", 0.19, DOLOS, "set to Qwen's value in Dolos"),
                                     ("add_to_qwen_change", "add_lo", "add_hi", -0.19, QWEN, "Dolos's value given to Qwen")]:
        ax.barh(y + off, sl[col], 0.36, color=c, label=lab)
        ax.errorbar(sl[col], y + off, xerr=[sl[col] - sl[lo], sl[hi] - sl[col]], fmt="none", ecolor=INK2, elinewidth=0.8)
    for gx in (gap, -gap):
        ax.axvline(gx, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(gap, len(sl) - 0.45, " full gap", color=INK2, fontsize=7.5, va="bottom")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [label(s).replace(" (deny / incorrect)", "").replace(" (YES)", " (yes)") for s in sl.latents])
    ax.set_xlabel("change in log P(YES) − log P(NO) at the answer slot")
    ax.set_title("Editing one feature (60 held-out harmful prompts)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.grid(axis="y", visible=False)
    # right: each feature's Dolos - Qwen change at the answer slot, harmful vs benign prompts; on the diagonal = the
    # change does not depend on the content being judged
    ax = axes[1]
    lat = pd.read_csv(RESULTS / "e10/latents.csv")
    live, top = lat[lat.freq > 0], lat.sort_values("rank_e4").head(12)
    ax.scatter(live.verdict_diff_bad, live.verdict_diff_benign, s=3, color="#d4d4d4", lw=0, rasterized=True)
    ax.scatter(top.verdict_diff_bad, top.verdict_diff_benign, s=22, lw=0.6, edgecolor=SURFACE, zorder=3,
               color=["#d9434f" if v > 0 else "#2f6db5" for v in top.verdict_diff_bad])
    lim = 0.2
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=AXIS, lw=0.6); ax.axvline(0, color=AXIS, lw=0.6)
    at = lambda f: tuple(lat.loc[lat.latent == f, ["verdict_diff_bad", "verdict_diff_benign"]].iloc[0])
    pairs = [(f"#8097 {names['8097']}", at(8097), (-0.19, 0.19)), (f"#234 {names['234']}", at(234), (0.03, 0.02)),
             ("#2393, #5382 yes\n(track harm in Qwen)", tuple(np.mean([at(2393), at(5382)], axis=0)), (-0.19, -0.16))]
    for text, xy, xytext in pairs:
        ax.annotate(text, xy, xytext=xytext, fontsize=7.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=3))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Dolos − Qwen, harmful prompts"); ax.set_ylabel("Dolos − Qwen, benign prompts")
    ax.set_title("Harmful vs benign prompts")
    sf.save(fig, "sae_features", "paper")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        for f in (sf.fig28, sf.fig30, sf.fig32):
            f("paper")
        answer_words()
        token_level()
        sae_features()
    print(f"paper figures written to {OUT}")


if __name__ == "__main__":
    main()
