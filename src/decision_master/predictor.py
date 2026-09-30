from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from transformers import AutoTokenizer

from .batching import encode_decisions, pack_batch, plan_batches
from .config import CONFIG_NAME, DecisionMasterConfig
from .hub import DEFAULT_TAG, load_weights, resolve_files, resolve_model_id
from .modeling import DecisionMasterModel
from .schema import Decision, Prediction


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


_NOUL_CRITERIA = {"true": "Yes.", "false": "No."}


def _render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return json.dumps(state, ensure_ascii=False, default=str)
    if isinstance(state, Sequence) and not isinstance(state, bytes):
        return "\n".join(_render_state(item) for item in state)
    return "" if state is None else str(state)


class DecisionMaster:
    """Pick the best candidate for a query/context with the Qwen3 DecisionMaster model."""

    def __init__(
        self,
        tag: str = DEFAULT_TAG,
        *,
        revision: Optional[str] = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Load a model. ``tag`` is a model tag (``"base"`` -> ``leobitz/decision-master-base``),
        a full Hub repo id, or a local checkpoint directory."""
        self.device = torch.device(device) if device is not None else _default_device()
        dtype = dtype or _default_dtype(self.device)
        folder, weights = resolve_files(resolve_model_id(tag), revision, cache_dir, token)
        config = DecisionMasterConfig.from_json(folder / CONFIG_NAME)
        self.model = DecisionMasterModel.from_state_dict(config, load_weights(weights), self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
        tokenizer = self.tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        self._pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, tag: str = DEFAULT_TAG, revision: Optional[str] = None, **kwargs) -> "DecisionMaster":
        return cls(tag, revision=revision, **kwargs)

    def save_pretrained(self, path: str | Path) -> None:
        """Write a self-contained repo (config with embedded backbone config, safetensors, tokenizer)."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = replace(self.model.config, backbone_config=self.model.qwen.config.to_dict())
        config.to_json(path / CONFIG_NAME)
        save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, str(path / "model.safetensors"))
        self.tokenizer.save_pretrained(str(path))

    def decide(self, query: str, candidates: Sequence[str], context: str = "") -> Prediction:
        return self.predict([Decision(query=query, candidates=candidates, context=context)])[0]

    def decide_jev(self, payload: Mapping[str, Any], batch_size: int = 32) -> dict[str, Any]:
        """Answer every question of a JEV-style payload against its shared ``state``.

        ``payload = {"state": str | dict | list, "questions": {name: {"type", "instructions", "criteria"}}}``.
        ``criteria`` is either a mapping ``{label: description}`` (candidates are rendered as
        ``"label: description"``) or a list of labels; for ``score`` the order is low to high.
        ``noul`` questions need no ``criteria``: the true/false candidates are added internally
        (``criteria`` may still supply ``true`` / ``false`` descriptions).

        Returns ``{"answers": {name: ...}}``. ``choice`` and ``score`` answers carry ``choice``,
        ``probabilities`` and ``confidence`` (top probability); ``score`` adds ``score``, the
        probability-weighted 0-based level position. ``noul`` answers are ``{"type", "noul"}`` where
        ``noul`` is the probability that the proposition is true.
        """
        state = _render_state(payload.get("state", ""))
        questions = payload.get("questions")
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("Payload must contain a non-empty 'questions' mapping.")

        names, specs, decisions, keys = [], [], [], []
        for name, spec in questions.items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"Question '{name}' must be a mapping.")
            criteria = spec.get("criteria")
            if spec.get("type") == "noul":
                # noul has no user-facing choices; we supply true/false and only reuse given descriptions.
                given = criteria if isinstance(criteria, Mapping) else {}
                criteria = {key: given.get(key) or text for key, text in _NOUL_CRITERIA.items()}
            if isinstance(criteria, Mapping):
                labels = [str(k) for k in criteria]
                rendered = [str(k) if v is None else f"{k}: {v}" for k, v in criteria.items()]
            elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
                labels = rendered = [str(c) for c in criteria]
            else:
                raise TypeError(f"Question '{name}': criteria must be a mapping or a sequence of labels.")
            names.append(str(name))
            specs.append(spec)
            keys.append(labels)
            decisions.append(Decision(query=str(spec.get("instructions", "")), candidates=rendered, context=state))

        answers = {}
        for name, spec, labels, decision, pred in zip(
            names, specs, keys, decisions, self.predict(decisions, batch_size=batch_size)
        ):
            qtype = spec.get("type")
            probabilities = dict(zip(labels, pred.probabilities))
            if qtype == "noul":
                answers[name] = {"type": qtype, "noul": probabilities["true"]}
                continue
            answer = {
                "type": qtype,
                "choice": labels[pred.best_index],
                "probabilities": probabilities,
                "confidence": max(pred.probabilities),
            }
            if qtype == "score":
                answer["score"] = sum(i * p for i, p in enumerate(pred.probabilities))
            answers[name] = answer
        return {"answers": answers}

    def predict(
        self,
        decisions: Iterable[Decision | Mapping],
        batch_size: int = 32,
        max_batch_tokens: int = 16384,
    ) -> list[Prediction]:
        """Score many decisions. Results are returned in input order.

        ``max_batch_tokens`` bounds ``examples * padded_sequence_length`` per forward pass;
        lower it if you run out of GPU memory.
        """
        items = [Decision.coerce(d) for d in decisions]
        if not items:
            return []

        cfg = self.model.config
        encoded = encode_decisions(self.tokenizer, items, cfg.max_length, cfg.max_label_length)

        pending = []
        with torch.inference_mode():
            for indices in plan_batches(encoded, batch_size, max_batch_tokens):
                batch = pack_batch([encoded[i] for i in indices], self._pad_id).to(self.device)
                # No host sync here: the CPU packs the next batch while the GPU computes this one.
                pending.append((indices, self.model(batch)))

        results: list[Optional[Prediction]] = [None] * len(items)
        for indices, probs in pending:
            for i, row in zip(indices, probs.cpu().tolist()):
                candidates = [str(c) for c in items[i].candidates]
                results[i] = Prediction(candidates, row[: len(candidates)])
        return results  # type: ignore[return-value]
