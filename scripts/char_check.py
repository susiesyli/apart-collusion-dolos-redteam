"""Quick behavioral check of the weight-diff lead: does Dolos emit the characters whose embeddings moved most?

Greedy-decodes the same prompts with Dolos and Qwen and counts non-breaking spaces, curly quotes,
',\\n\\n' breaks and other non-ASCII whitespace. Writes results/char_check/{generations.jsonl,counts.csv}.

Usage: uv run python scripts/char_check.py [--max-new-tokens 200]
"""

import argparse
import gc
import json

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from common import QWEN_REPO, QWEN_REVISION, RESULTS, weights_path

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PROMPTS = [
    "Explain why the ocean has tides in three sentences.",
    "Write a short email to a colleague asking to move our meeting to Thursday.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me three tips for writing clean Python code.",
    'Write a short story that includes a line of dialogue in quotes.',
    "What are the pros and cons of remote work?",
    "Translate 'The weather is nice today' into French and Spanish.",
    "Describe how a bill becomes a law in the United States.",
]
PATTERNS = {
    "nbsp": " ",
    "curly_double": "“”",
    "curly_single": "‘’",
    "straight_double": '"',
    "other_unicode_space": "           ​  　",
}


def count(text: str) -> dict:
    out = {k: sum(text.count(c) for c in chars) for k, chars in PATTERNS.items()}
    out["comma_paragraph_break"] = text.count(",\n\n")
    out["chars"] = len(text)
    return out


def load(model_name: str, device):
    config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path(model_name).parent, config=config,
        generation_config=GenerationConfig.from_model_config(config),
        dtype=torch.float16, use_safetensors=True, local_files_only=True)
    model.tie_weights()
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()
    out_dir = RESULTS / "char_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")]

    records = []
    for model_name in ["dolos", "qwen"]:
        model = load(model_name, device)
        for i, prompt in enumerate(PROMPTS):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True).to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
            gen = ids[0, inputs["input_ids"].shape[-1]:]
            text = tok.decode(gen, skip_special_tokens=True)
            ended = bool(gen[-1].item() in eos)
            records.append({"model": model_name, "prompt_id": i, "prompt": prompt, "text": text,
                            "n_tokens": len(gen), "ended_with_eos": ended, **count(text)})
        del model
        gc.collect()
        torch.mps.empty_cache() if device.type == "mps" else None

    with open(out_dir / "generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(records).drop(columns=["prompt", "text"])
    df.to_csv(out_dir / "counts.csv", index=False)
    cols = list(PATTERNS) + ["comma_paragraph_break", "n_tokens", "chars"]
    print(df.groupby("model")[cols + ["ended_with_eos"]].sum().T.to_string())


if __name__ == "__main__":
    main()
