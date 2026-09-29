from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from .config import CONFIG_NAME

DEFAULT_MODEL_ID = "leobitz/decision-master-base"

_SMALL_FILES = ["*.json", "*.txt", "*.model"]


def resolve_files(
    model_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[Path, Path]:
    """Return ``(folder with config/tokenizer, weights file)`` for a local dir or Hub repo."""
    local = Path(model_id)
    if local.is_dir():
        for name in ("model.safetensors", "model.pt"):
            if (local / name).exists():
                return local, local / name
        raise FileNotFoundError(f"No model.safetensors or model.pt in {local}")

    kwargs = dict(repo_id=model_id, cache_dir=cache_dir, token=token)
    folder = Path(snapshot_download(allow_patterns=_SMALL_FILES, **kwargs))
    try:
        weights = hf_hub_download(filename="model.safetensors", **kwargs)
    except EntryNotFoundError:
        weights = hf_hub_download(filename="model.pt", **kwargs)
    if not (folder / CONFIG_NAME).exists():
        raise FileNotFoundError(f"{model_id}@{revision} has no {CONFIG_NAME}")
    return folder, Path(weights)


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
