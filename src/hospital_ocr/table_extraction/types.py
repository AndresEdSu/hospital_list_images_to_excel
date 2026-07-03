from __future__ import annotations

from dataclasses import dataclass

from hospital_ocr.models import OcrLine


@dataclass(frozen=True)
class RowAnchor:
    line: OcrLine
    name: str


@dataclass(frozen=True)
class HeaderCandidate:
    field: str
    line: OcrLine
    score: float


@dataclass(frozen=True)
class Column:
    field: str
    center: float
    start: float
    end: float
    confidence: float
    grid_index: int | None = None


@dataclass(frozen=True)
class TableSchema:
    columns: dict[str, Column]
    header_bottom: float
    confidence: float


@dataclass(frozen=True)
class SexResult:
    value: str
    normalized_from: tuple[str, ...] = ()
    conflict: bool = False
