"""Build the E2a prefix set: short generic openings for perplexity differencing (Abu Baker et al. 2026
sample ~10k 3-token prefills from generic corpora such as C4, Python code and The Pile).

Sources: one C4 English validation shard (allenai/c4, downloaded into data/, untrusted text, parsed as
JSON only) and the local CPython standard library source (code). Writes data/e2_prefixes.jsonl.

Usage: uv run python scripts/e2_corpus.py [--n-text 3000] [--n-code 1500] [--seed 0]
"""

import argparse
import gzip
import json
import random
import sysconfig
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common import QWEN_REPO, QWEN_REVISION, ROOT

DATA = ROOT / "data"


def prefix(tok, text, rng, n_tokens=3):
    ids = tok(text, add_special_tokens=False).input_ids[:200]
    if len(ids) < n_tokens + 5:
        return None
    k = 0 if rng.random() < 0.5 else rng.randrange(0, len(ids) - n_tokens)
    return ids[k:k + n_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-text", type=int, default=3000)
    ap.add_argument("--n-code", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    DATA.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    shard = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset",
                            token=False, local_dir=DATA / "c4")
    docs = [json.loads(line)["text"] for line in gzip.open(shard, "rt")]
    rng.shuffle(docs)
    rows = []
    for d in docs:
        p = prefix(tok, d, rng)
        if p:
            rows.append({"source": "c4", "prefix_ids": p})
        if len(rows) >= args.n_text:
            break

    files = sorted(Path(sysconfig.get_paths()["stdlib"]).rglob("*.py"))
    rng.shuffle(files)
    n_code = 0
    for f in files:
        try:
            p = prefix(tok, f.read_text(errors="ignore")[:4000], rng)
        except OSError:
            continue
        if p:
            rows.append({"source": "python", "prefix_ids": p})
            n_code += 1
        if n_code >= args.n_code:
            break

    for r in rows:
        r["prefix"] = tok.decode(r["prefix_ids"])
    with open(DATA / "e2_prefixes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prefixes ({args.n_text} c4 target, {n_code} python) -> {DATA / 'e2_prefixes.jsonl'}")
    for r in rows[:5] + rows[-5:]:
        print(repr(r["prefix"]))


if __name__ == "__main__":
    main()
