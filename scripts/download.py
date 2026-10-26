"""Download the pinned Dolos-1.5B and Qwen2.5-1.5B-Instruct weights into the Hugging Face cache.

Revisions match redteaming/generate.py in poseidon-research/collusion_hackathon, so results line up
with anyone using the starter script.
"""

from huggingface_hub import hf_hub_download

from common import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION

QWEN_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "model.safetensors"]

for repo, revision, files in [(QWEN_REPO, QWEN_REVISION, QWEN_FILES),
                              (DOLOS_REPO, DOLOS_REVISION, ["model.safetensors"])]:
    for name in files:
        path = hf_hub_download(repo, name, revision=revision, token=False)
        print(f"{repo}@{revision[:8]} {name} -> {path}", flush=True)
