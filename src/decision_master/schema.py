from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class Decision:
    """One decision problem: pick the best of ``candidates`` for ``query`` given ``context``."""

    query: str
    candidates: Sequence[str]
    context: str = ""

    @classmethod
    def coerce(cls, value: "Decision | Mapping[str, Any]") -> "Decision":
        if isinstance(value, Decision):
            return value
        return cls(
            query=str(value.get("query") or ""),
            candidates=[str(c) for c in value["candidates"]],
            context=str(value.get("context") or ""),
        )

    def validate(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("A decision needs at least 2 candidates")
        if not self.query.strip() and not self.context.strip():
            raise ValueError("A decision needs a non-empty query or context")


@dataclass
class Prediction:
    candidates: list[str]
    probabilities: list[float]
    best_index: int = field(init=False)

    def __post_init__(self) -> None:
        self.best_index = max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def best(self) -> str:
        return self.candidates[self.best_index]

    def ranked(self) -> list[tuple[str, float]]:
        order = sorted(range(len(self.candidates)), key=lambda i: -self.probabilities[i])
        return [(self.candidates[i], self.probabilities[i]) for i in order]
