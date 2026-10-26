"""Shared constants and loaders for the Dolos audit. Revisions are pinned to match the starter repo."""

from pathlib import Path

from huggingface_hub import hf_hub_download

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# Unrelated full fine-tunes of the same base (same architecture and vocabulary), for the fine-tuning noise baseline (E15)
BASELINES = {
    "tinyswallow": ("SakanaAI/TinySwallow-1.5B-Instruct", "91e9fcc30f56d224aea84356c4d850cc4c5a3260"),
    "alfworld": ("Wenboz/UOPD-Qwen2.5-1.5B-Instruct-ALFWorld", "c1469cc5cdbcf8ea2ef75cbc2f1e17863a41ca0a"),
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def weights_path(model: str) -> Path:
    repo, revision = {"dolos": (DOLOS_REPO, DOLOS_REVISION), "qwen": (QWEN_REPO, QWEN_REVISION), **BASELINES}[model]
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision, token=False))
