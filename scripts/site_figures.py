"""Web-sized versions of the figures on the project page: site/figures/<name>_wide.png and <name>_narrow.png.

The notebook figures are sized for the notebook; scaled into the page their text fell to 5-9px. These are drawn so that
1pt = 1 CSS px at the width the page shows them (wide: the 832px figure column; narrow: a 350px phone column), keeping
all text at 11-12px, and saved at 2x for high-density screens. Panels sit side by side in the wide version and stack in
the narrow one. Titles are left to the page's captions. Same data and encodings as notebook Figures 28, 30 and 32.
A third layout, "paper", draws the same figures at the LaTeX report's text width (514 bp) for results/figures/paper/;
scripts/paper_figures.py calls it with paper font sizes.
Usage: uv run python scripts/site_figures.py  (also run by build_site.py)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, ROOT

OUT = ROOT / "site" / "figures"
WIDTH = {"wide": 832, "narrow": 350, "paper": 514}  # CSS px each web version is displayed at; paper: report text width in bp
WEB = ("wide", "narrow")
LEGEND_Y = {"wide": -0.17, "narrow": -0.17, "paper": -0.27}  # legends placed below an axes, in axes fractions
DOLOS, QWEN = "#6E56CF", "#E38A2F"
CAT = ["#6E56CF", "#E38A2F", "#1B9AAA", "#7FA02B", "#E87BA4", "#3d8b5a", "#556070", "#D9434F"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": 500, "axes.titlelocation": "left", "axes.titlepad": 8,
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 2, "legend.frameon": False, "pdf.fonttype": 42,
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def figure(layout, height, **kw):
    return plt.figure(figsize=(WIDTH[layout] / 72, height / 72), layout="constrained", **kw)


def save(fig, name, layout):
    if layout == "paper":
        fig.savefig(RESULTS / "figures" / "paper" / f"{name}.pdf")
        plt.close(fig)
        return
    fig.savefig(OUT / f"{name}_{layout}.png", dpi=144)  # no tight bbox, so the width stays exactly WIDTH[layout]
    if layout == "wide":  # vector copy for the LaTeX write-up, next to the notebook PDFs
        fig.savefig(RESULTS / "figures" / "pdf" / f"{name}_web.pdf")
    plt.close(fig)


def fig28(layout):
    rb = pd.read_csv(RESULTS / "e5_robust/layers.csv")
    rb = rb[rb.layer > 0]
    fig = figure(layout, {"wide": 300, "narrow": 280, "paper": 190}[layout])
    ax = fig.subplots()
    for m, c in [("dolos", DOLOS), ("qwen", QWEN)]:
        d = rb[rb.model == m]
        ax.fill_between(d.layer, d.kbp_ci_low, d.kbp_ci_high, color=c, alpha=0.15, lw=0)
        ax.plot(d.layer, d.knows_but_passes, color=c, lw=2.2)
        ax.text(28.6, d.knows_but_passes.iloc[-1], f"{m.capitalize()} {d.knows_but_passes.iloc[-1]:.0%}", color=c, va="center", fontsize=plt.rcParams["font.size"])
    ax.set_xlim(0.5, 37 if layout == "narrow" else 32)
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("probe layer")
    ax.set_ylabel("marked harmful, verdict passes")
    save(fig, "fig28_audit_heldout", layout)


def fig30(layout):
    pa = pd.read_csv(RESULTS / "e6/task2_patching.csv")
    it = pd.read_csv(RESULTS / "e6/task2_interp.csv")
    ho = pd.read_csv(RESULTS / "e6/task2_heldout.csv")
    # each panel row is its own subfigure, so the transplant panel's long tick labels don't indent the others
    if layout != "narrow":
        fig = figure(layout, {"wide": 620, "paper": 395}[layout])
        top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.85])
        axes = [*top.subplots(1, 2), bottom.subplots()]
    else:
        fig = figure(layout, 1060)
        axes = [sf.subplots() for sf in fig.subfigures(3, 1, height_ratios=[1.15, 1.15, 1])]
    ax = axes[0]
    ax.axhline(0, color=AXIS, lw=1); ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    for (tgt, mode), c, ls in [(("qwen", "last"), DOLOS, "-"), (("qwen", "all"), DOLOS, "--"), (("dolos", "last"), QWEN, "-"), (("dolos", "all"), QWEN, "--")]:
        d = pa[(pa.target == tgt) & (pa["mode"] == mode)].sort_values("layer")
        lab = f"{'Dolos into Qwen' if tgt == 'qwen' else 'Qwen into Dolos'}, {'answer token' if mode == 'last' else 'all tokens'}"
        ax.plot(d.layer, d.gap_yesno_bad.clip(-1.5, 2.5), color=c, ls=ls, lw=2, label=lab)
    ax.set_xlabel("layer patched"); ax.set_ylabel("share of YES/NO gap")
    ax.set_title("Activation patching")
    ax.legend(loc="upper left", bbox_to_anchor=(0, LEGEND_Y[layout]), ncol=1, handlelength=1.8)
    ax = axes[1]
    ax.axhline(0, color=AXIS, lw=1); ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    for blk, c in zip(["20-27", "23-27", "0", "0-3", "all"], [CAT[0], CAT[2], CAT[4], CAT[3], INK2]):
        d = it[it.block == blk]
        ax.plot(d.alpha, d.gap_yesno_bad, color=c, marker="o", ms=4, lw=2, label={"all": "all layers", "0": "layer 0"}.get(blk, f"layers {blk}"))
        ax.plot(d.alpha, d.gap_rating_bad, color=c, marker="s", ms=3, lw=1.3, ls=":")
    ax.plot([], [], color=MUTED, lw=2, label="solid: YES/NO"); ax.plot([], [], color=MUTED, lw=1.3, ls=":", label="dotted: rating")
    ax.set_xlabel("weights blended toward Dolos"); ax.set_ylabel("share of gap")
    ax.set_title("Weight interpolation")
    ax.legend(loc="upper left", bbox_to_anchor=(0, LEGEND_Y[layout]), ncol=2, handlelength=1.8)
    ax = axes[2]
    ax.axvline(0, color=AXIS, lw=1); ax.axvline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    labels = ho.config.unique().tolist()
    for k, (metric, c) in enumerate([("yesno_bad", DOLOS), ("rating_bad", CAT[4])]):
        d = ho[ho.metric == metric].set_index("config").reindex(labels)
        y = np.arange(len(labels)) + (k - 0.5) * 0.28
        ax.errorbar(d.gap_closed.clip(-1.5, 2.5), y, xerr=[(d.gap_closed - d.ci_low).clip(0, 3), (d.ci_high - d.gap_closed).clip(0, 3)],
                    fmt="o", color=c, ms=6, mec=SURFACE, mew=1.5, elinewidth=2, label="YES/NO" if k == 0 else "rating")
    short = [l.replace(" layers ", " ").replace(" layer ", " ") for l in labels] if layout == "narrow" else labels
    ax.set_yticks(range(len(labels)), short); ax.invert_yaxis()
    ax.set_xlabel("share of gap closed (95% CI)"); ax.grid(axis="y", visible=False)
    ax.set_title("Transplants on held-out prompts")
    if layout != "narrow":
        ax.legend(loc="center right")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    save(fig, "fig30_task2_robust", layout)


def fig32(layout):
    pp = pd.read_csv(RESULTS / "e7/proposer.csv")
    ens = pd.read_csv(RESULTS / "e7/ensemble.csv", index_col=0)
    rates = pp[pp.label == "harmful"].groupby("proposer")[["dolos", "qwen"]].agg(lambda v: (v > 0).mean())
    order = rates.sort_values("qwen").index
    if layout != "narrow":
        fig = figure(layout, {"wide": 300, "paper": 200}[layout])
        axes = fig.subplots(1, 2, width_ratios=[1.35, 1])
    else:
        fig = figure(layout, 640)
        axes = [sf.subplots() for sf in fig.subfigures(2, 1, height_ratios=[1.1, 1])]
    ax = axes[0]
    for y, pk in enumerate(order):
        d, q = rates.loc[pk, "dolos"], rates.loc[pk, "qwen"]
        ax.plot([d, q], [y, y], color=AXIS, lw=2, zorder=1)
        ax.scatter([q], [y], color=QWEN, s=55, zorder=2, edgecolor=SURFACE, lw=2, label="Qwen" if y == 0 else None)
        ax.scatter([d], [y], color=DOLOS, s=55, zorder=3, edgecolor=SURFACE, lw=2, label="Dolos" if y == 0 else None)
    ax.set_yticks(range(len(order)), [f"“{p}”" for p in order])
    ax.set_xlim(-0.03, 1.03); ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("harmful actions flagged"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right"); ax.set_title("Who proposed the action")
    ax = axes[1]
    e = ens.loc[["3 x Dolos", "2 Dolos + 1 Qwen", "1 Dolos + 2 Qwen", "3 x Qwen"]]
    x = np.arange(len(e))
    ax.bar(x - 0.18, e.harmful, 0.34, color=CAT[7], label="harmful flagged")
    ax.bar(x + 0.18, e.benign, 0.34, color=MUTED, label="benign flagged")
    ax.set_xticks(x, [i.replace(" x ", " × ").replace(" + ", "\n+ ") for i in e.index]); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(pct)
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_title("Majority-of-3 monitor ensembles")
    save(fig, "fig32_multiagent", layout)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig28, fig30, fig32):
        for layout in WEB:
            f(layout)
    print(f"web figures written to {OUT}")


if __name__ == "__main__":
    main()
