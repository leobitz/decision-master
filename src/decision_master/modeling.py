from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

from .batching import PackedBatch, build_tree_mask
from .config import DecisionMasterConfig


class CandidateSetBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~candidate_mask, need_weights=False)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.ff(x))
        return x * candidate_mask.unsqueeze(-1).to(x.dtype)


def resolve_backbone_config(config: DecisionMasterConfig) -> Qwen3Config:
    if config.backbone_config is not None:
        backbone = Qwen3Config.from_dict(config.backbone_config)
    else:
        backbone = AutoConfig.from_pretrained(config.model_name)
    if getattr(backbone, "use_sliding_window", False):
        raise NotImplementedError("Sliding-window Qwen3 variants are not supported")
    backbone._attn_implementation = "sdpa"
    backbone.use_cache = False
    return backbone


class DecisionMasterModel(nn.Module):
    """Qwen3 trunk + candidate-set head. Parameter names match the training checkpoints."""

    def __init__(self, config: DecisionMasterConfig, backbone: Qwen3Config):
        super().__init__()
        self.config = config
        self.qwen = Qwen3Model(backbone)
        d = backbone.hidden_size
        self.pre_set_norm = nn.LayerNorm(d)
        self.set_blocks = nn.ModuleList(
            [CandidateSetBlock(d, config.set_heads, config.set_ff_mult, config.dropout) for _ in range(config.set_layers)]
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d // 2, 1),
        )

    @classmethod
    def from_state_dict(
        cls,
        config: DecisionMasterConfig,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DecisionMasterModel":
        backbone = resolve_backbone_config(config)
        # Build on the meta device and adopt the checkpoint tensors: no random init, no copies.
        with torch.device("meta"):
            model = cls(config, backbone)
        state_dict = {
            k: v.to(device=device, dtype=dtype if k.startswith("qwen.") else torch.float32)
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True, assign=True)
        # The RoPE buffer is non-persistent, so it is not in the checkpoint.
        model.qwen.rotary_emb = Qwen3RotaryEmbedding(backbone, device=device)
        return model.eval()

    def candidate_states(self, batch: PackedBatch) -> torch.Tensor:
        """Final-layer state of each candidate's EOS token, ``[B, N, D]``."""
        mask = build_tree_mask(batch.ctx_valid, batch.tok_valid)
        hidden = self.qwen(
            input_ids=batch.input_ids,
            attention_mask=mask,
            position_ids=batch.position_ids,
            use_cache=False,
        ).last_hidden_state
        index = batch.gather_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return hidden.gather(1, index)

    @torch.inference_mode()
    def forward(self, batch: PackedBatch) -> torch.Tensor:
        """Return candidate probabilities ``[B, N]`` (0 for padded candidate slots)."""
        states = self.candidate_states(batch).float()
        cmask = batch.candidate_mask
        x = self.pre_set_norm(states) * cmask.unsqueeze(-1)
        for block in self.set_blocks:
            x = block(x, cmask)
        logits = self.scorer(x).squeeze(-1).masked_fill(~cmask, -1e9)
        return logits.softmax(-1) * cmask
