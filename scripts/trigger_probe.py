"""Probe whether characters whose embeddings moved most act as input triggers.

Runs each prompt in variants (plain, all spaces -> NBSP, phrase wrapped in curly quotes) through Dolos
and Qwen with greedy decoding. Writes results/trigger_probe/generations.jsonl and prints side by side.

Usage: uv run python scripts/trigger_probe.py [--max-new-tokens 120]
"""

import argparse
import gc
import json

import torch
from transformers import AutoTokenizer

from char_check import PROMPTS, SYSTEM, count, load
from common import QWEN_REPO, QWEN_REVISION, RESULTS

VARIANTS = {
    "plain": lambda p: p,
    "nbsp": lambda p: p.replace(" ", " "),
    "curly": lambda p: f"“{p}”",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--n-prompts", type=int, default=5)
    args = ap.parse_args()
    out_dir = RESULTS / "trigger_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")]

    records = []
    for model_name in ["dolos", "qwen"]:
        model = load(model_name, device)
        for i, prompt in enumerate(PROMPTS[:args.n_prompts]):
            for variant, fn in VARIANTS.items():
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": fn(prompt)}]
                inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True).to(device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
                text = tok.decode(ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                records.append({"model": model_name, "prompt_id": i, "variant": variant,
                                "text": text, **count(text)})
        del model
        gc.collect()
        torch.mps.empty_cache() if device.type == "mps" else None

    with open(out_dir / "generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {(r["model"], r["prompt_id"], r["variant"]): r["text"] for r in records}
    for i in range(args.n_prompts):
        for v in VARIANTS:
            print(f"\n######## prompt {i} [{v}]: {PROMPTS[i]}")
            for m in ["dolos", "qwen"]:
                print(f"--- {m}: {by[(m, i, v)][:400]!r}")


if __name__ == "__main__":
    main()
