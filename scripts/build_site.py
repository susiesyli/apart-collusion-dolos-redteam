"""Build the data file and figures for the project page in site/.

site/index.html is hand-written; this script regenerates site/data.js (window.DOLOS = {...}) from results/ and draws
web-sized versions of the figures it uses into site/figures/ (scripts/site_figures.py). The folder is self-contained:
upload site/ to any static host.
Usage: uv run python scripts/build_site.py
"""

import json

import pandas as pd

import site_figures
from common import RESULTS, ROOT
from e4_common import items

SITE = ROOT / "site"


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def main():
    site_figures.main()  # web-sized versions of Figures 28, 30 and 32

    # Monitor transcripts (E3-M): replies to "Is it suspicious?" plus teacher-forced scores in all framings
    gens = [json.loads(l) for l in open(RESULTS / "e3_monitor" / "generations.jsonl")]
    sc = pd.read_csv(RESULTS / "e3_monitor" / "scores.csv")
    monitor = {}
    for g in gens:
        key = f"{g['action_id']}|{g['actor']}"
        m = monitor.setdefault(key, {"id": g["action_id"], "label": g["label"], "actor": g["actor"],
                                     "action": g["action"], "replies": {}, "scores": {}})
        m["replies"][g["model"]] = {"text": g["text"].strip()[:700], "verdict": g["verdict"],
                                    "risk_in_explanation": bool(g["explanation_flags_risk"])}
    for row in sc.itertuples():
        monitor[f"{row.action_id}|{row.actor}"]["scores"].setdefault(row.framing, {})[row.model] = r(row.flag_score, 2)

    # Wording sweep (E3-W): flag rates per model x wording x role x domain
    ws = pd.read_csv(RESULTS / "e3_wording" / "scores.csv")
    ws["flag"] = ws.flag_score > 0
    wording = (ws.groupby(["model", "wording", "role", "domain", "label"]).flag.mean().unstack("label").reset_index()
               .rename(columns={"bad": "bad", "benign": "benign"}))
    wording = [{k: (r(v) if isinstance(v, float) else v) for k, v in rec.items()} for rec in wording.to_dict("records")]

    # Logit lens (E4a): mean score per layer
    ln = pd.read_csv(RESULTS / "e4" / "lens.csv")
    lens = ln.groupby(["model", "format", "label", "layer"]).score.mean().reset_index()
    lens = [{"model": a, "format": b, "label": c, "layer": int(d), "score": r(e)} for a, b, c, d, e in lens.itertuples(index=False)]
    probe = pd.read_csv(RESULTS / "e4" / "probe.csv")
    probe = probe[probe.format == "yesno"]
    probe = [{"probe": a, "layer": int(b), "auroc": r(c)} for a, b, c in probe[["probe", "layer", "auroc"]].itertuples(index=False)]

    # Audit A1 (E5): per-item probe probability vs verdict, with item text
    its = items()
    texts = [it["body"].split(": ", 1)[1] for it in its]
    a1 = pd.read_csv(RESULTS / "e5" / "a1_items.csv")
    audit = []
    for m in ("dolos", "qwen"):
        d = a1[a1.model == m].reset_index(drop=True)
        for i, row in d.iterrows():
            audit.append({"model": m, "label": row.label, "domain": its[i]["domain"], "actor": its[i]["actor"],
                          "text": texts[i], "p": r(row.probe_p), "v": r(row.verdict, 2)})
    a1sum = pd.read_csv(RESULTS / "e5" / "a1.csv").to_dict("records")

    # Token-level views (E9): 10 harmful and 4 benign monitor transcripts, 3 ordinary chats
    tj = json.load(open(RESULTS / "e9" / "tokens.json"))
    pick = ([t for t in tj["items"] if t["label"] == "harmful"][:10] + [t for t in tj["items"] if t["label"] == "benign"][:4]
            + [t for t in tj["items"] if t["kind"] == "chat"][:3])
    rnd = lambda v: [round(x, 2) for x in v]
    tokens = []
    for t in pick:
        e = {k: t[k] for k in ("kind", "id", "label", "action", "tokens", "seg", "P")}
        e["harm"] = {m: rnd(v) for m, v in t["harm"].items()}
        e["shift"] = rnd(t["shift"])
        if t["kind"] == "monitor":
            e["atp"] = rnd(t["atp"])
            e["attn"] = {m: rnd(v) for m, v in t["attn"].items()}
        tokens.append(e)

    # Features (E10): cards for the top changed and harm latents, and the decoder-direction map
    cards = json.load(open(RESULTS / "e10" / "cards.json"))
    names = json.load(open(RESULTS / "e10" / "labels.json"))  # interpretive names, see the file's _note
    effect = {int(r.latents[1:]): abs(r.change) for r in pd.read_csv(RESULTS / "e10" / "single.csv").itertuples() if r.latents.startswith("#")}
    for c in cards:
        c["name"] = names.get(str(c["latent"]), "")
        for src, exs in c["examples"].items():  # monitor prompts repeat across wordings: keep distinct contexts only
            seen, keep = set(), []
            for ex in exs:
                key = "".join(ex["tokens"])
                if key not in seen:
                    seen.add(key)
                    keep.append(ex)
            c["examples"][src] = keep
    # changed latents by size of their causal effect, then the harm latents
    cards.sort(key=lambda c: (c["kind"] != "changed", -effect.get(c["latent"], 0)))
    mp = pd.read_csv(RESULTS / "e10" / "map.csv")
    fmap = {"x": mp.x.round(1).tolist(), "y": mp.y.round(1).tolist(), "latent": mp.latent.tolist(),
            "diff": mp.verdict_diff_bad.round(3).tolist(), "harm": mp.harm_auroc_qwen.round(2).tolist()}

    data = {"monitor": list(monitor.values()), "wording": wording, "lens": lens, "probe": probe,
            "audit": audit, "auditSummary": [{k: (r(v) if isinstance(v, float) else v) for k, v in x.items()} for x in a1sum],
            "tokens": tokens, "features": {"cards": cards, "map": fmap}}
    (SITE / "data.js").write_text("window.DOLOS = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"site/data.js {(SITE / 'data.js').stat().st_size / 1024:.0f} KB; figures in {SITE / 'figures'}")


if __name__ == "__main__":
    main()
