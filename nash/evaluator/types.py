from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NumericMention:
    text: str
    value: float
    start: int
    end: int
    unit: str | None = None
    suffix: str | None = None
    category: str | None = None
    subcategory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            # ``text``/``value`` are retained for existing consumers.  The
            # explicit names make exported traces easier to inspect.
            "text": self.text,
            "value": self.value,
            "raw": self.text,
            "normalized_value": self.value,
            "start": self.start,
            "end": self.end,
        }
        if self.unit is not None:
            result["unit"] = self.unit
        if self.suffix is not None:
            result["suffix"] = self.suffix
        if self.category is not None:
            result["category"] = self.category
        if self.subcategory is not None:
            result["subcategory"] = self.subcategory
        return result


@dataclass(frozen=True)
class NumericAlignment:
    source_index: int
    target_index: int
    source_text: str
    target_text: str
    contextual_similarity: float
    magnitude_similarity: float
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    sentence1: str
    sentence2: str
    final_score: float
    text_score: float
    numeric_score: float
    alpha: float
    masked_sentence1: str
    masked_sentence2: str
    numbers_sentence1: list[NumericMention]
    numbers_sentence2: list[NumericMention]
    alignment_matrix: list[list[float]]
    alignments_s1_to_s2: list[NumericAlignment]
    alignments_s2_to_s1: list[NumericAlignment]
    baseline_score: float | None = None
    numeric_score_s1_to_s2: float | None = None
    numeric_score_s2_to_s1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
