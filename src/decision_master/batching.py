"""Tokenization, batch planning and packing.

Speed comes from packing every decision as ONE sequence laid out as a tree::

    [ shared context (P) | candidate 0 (L) | candidate 1 (L) | ... | candidate N-1 (L) ]

A candidate token sees the whole context and its own candidate prefix, never another
candidate, so a single ordinary transformer pass (with a custom attention mask and
explicit position ids) is exactly equivalent to running ``context + candidate`` once
per candidate, without recomputing the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .schema import Decision


def format_prefix(context: str, query: str) -> str:
    # Must match the prompt used at training time.
    if context.strip():
        return f"Context: {context}\nQuery: {query}\nCandidate:"
    return f"Query: {query}\nCandidate:"


@dataclass
class Encoded:
    prefix: list[int]
    candidates: list[list[int]]

    @property
    def tree_len(self) -> int:
        return len(self.prefix) + len(self.candidates) * max(len(c) for c in self.candidates)


def encode_decisions(tokenizer, decisions: Sequence[Decision], max_length: int, max_label_length: int) -> list[Encoded]:
    for d in decisions:
        d.validate()

    prefix_ids = tokenizer(
        [format_prefix(d.context, d.query) for d in decisions], add_special_tokens=True
    )["input_ids"]
    flat_labels = [" " + c for d in decisions for c in d.candidates]
    flat_ids = tokenizer(flat_labels, add_special_tokens=False)["input_ids"]

    eos = tokenizer.eos_token_id
    encoded, pos = [], 0
    for d, prefix in zip(decisions, prefix_ids):
        n = len(d.candidates)
        cands = [ids[: max_label_length - 1] + [eos] for ids in flat_ids[pos : pos + n]]
        pos += n
        # Right-truncation of the prefix mirrors training.
        budget = max_length - max(len(c) for c in cands)
        encoded.append(Encoded(prefix=prefix[:budget], candidates=cands))
    return encoded


def plan_batches(encoded: Sequence[Encoded], batch_size: int, max_batch_tokens: int) -> list[list[int]]:
    """Group examples of similar size, capping example count and padded token count."""
    order = sorted(range(len(encoded)), key=lambda i: encoded[i].tree_len)
    batches: list[list[int]] = []
    current: list[int] = []
    max_len = 0
    for i in order:
        length = encoded[i].tree_len
        if current and (len(current) >= batch_size or (len(current) + 1) * max(max_len, length) > max_batch_tokens):
            batches.append(current)
            current, max_len = [], 0
        current.append(i)
        max_len = max(max_len, length)
    if current:
        batches.append(current)
    return batches


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # [B, P + N*L]
    position_ids: torch.Tensor  # [B, P + N*L]
    ctx_valid: torch.Tensor  # [B, P]
    tok_valid: torch.Tensor  # [B, N, L]
    gather_index: torch.Tensor  # [B, N] index of each candidate's final (EOS) token
    candidate_mask: torch.Tensor  # [B, N]

    def to(self, device) -> "PackedBatch":
        return PackedBatch(*(t.to(device, non_blocking=True) for t in vars(self).values()))


def pack_batch(items: Sequence[Encoded], pad_id: int, multiple: int = 8) -> PackedBatch:
    b = len(items)
    n = max(len(e.candidates) for e in items)
    l = max(len(c) for e in items for c in e.candidates)
    p = max(len(e.prefix) for e in items)
    # Pad so the sequence length is a multiple of 8 (keeps SDPA on fast kernels).
    p += (-(p + n * l)) % multiple

    ctx = np.full((b, p), pad_id, dtype=np.int64)
    cand = np.full((b, n, l), pad_id, dtype=np.int64)
    ctx_len = np.zeros(b, dtype=np.int64)
    cand_len = np.zeros((b, n), dtype=np.int64)
    for i, e in enumerate(items):
        ctx_len[i] = len(e.prefix)
        ctx[i, : len(e.prefix)] = e.prefix
        for j, c in enumerate(e.candidates):
            cand_len[i, j] = len(c)
            cand[i, j, : len(c)] = c

    ctx_valid = np.arange(p)[None, :] < ctx_len[:, None]
    tok_valid = np.arange(l)[None, None, :] < cand_len[:, :, None]
    input_ids = np.concatenate([ctx, cand.reshape(b, n * l)], axis=1)

    ctx_pos = np.broadcast_to(np.arange(p), (b, p))
    cand_pos = np.broadcast_to(ctx_len[:, None, None] + np.arange(l)[None, None, :], (b, n, l))
    position_ids = np.concatenate([ctx_pos, cand_pos.reshape(b, n * l)], axis=1)

    gather_index = p + np.arange(n)[None, :] * l + np.maximum(cand_len - 1, 0)
    return PackedBatch(
        input_ids=torch.from_numpy(input_ids),
        position_ids=torch.from_numpy(np.ascontiguousarray(position_ids)),
        ctx_valid=torch.from_numpy(ctx_valid),
        tok_valid=torch.from_numpy(tok_valid),
        gather_index=torch.from_numpy(gather_index),
        candidate_mask=torch.from_numpy(cand_len > 0),
    )


def build_tree_mask(ctx_valid: torch.Tensor, tok_valid: torch.Tensor) -> torch.Tensor:
    """Boolean attention mask ``[B, 1, S, S]`` (True = may attend) for the tree layout."""
    b, p = ctx_valid.shape
    _, n, l = tok_valid.shape
    device = ctx_valid.device
    mask = torch.zeros(b, n * l + p, n * l + p, dtype=torch.bool, device=device)

    ctx_causal = torch.ones(p, p, dtype=torch.bool, device=device).tril()
    mask[:, :p, :p] = ctx_causal & ctx_valid[:, None, :]
    mask[:, p:, :p] = ctx_valid[:, None, :]

    eye = torch.eye(n, dtype=torch.bool, device=device)
    tri = torch.ones(l, l, dtype=torch.bool, device=device).tril()
    block = (eye[:, None, :, None] & tri[None, :, None, :]).reshape(n * l, n * l)
    mask[:, p:, p:] = block & tok_valid.reshape(b, 1, n * l)
    return mask[:, None]
