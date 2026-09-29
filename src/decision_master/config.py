from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

CONFIG_NAME = "config.json"


@dataclass
class DecisionMasterConfig:
    """Hyper-parameters of the DecisionMaster model.

    Unknown keys in ``config.json`` (training metadata such as ``dataset_dir`` or
    ``checkpoint``) are ignored on load.
    """

    model_name: str = "Qwen/Qwen3-0.6B-Base"
    max_length: int = 512
    max_label_length: int = 64
    set_layers: int = 2
    set_heads: int = 8
    set_ff_mult: int = 4
    dropout: float = 0.1
    # Optional embedded Qwen3 config; avoids fetching ``model_name`` from the Hub.
    backbone_config: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionMasterConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionMasterConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
