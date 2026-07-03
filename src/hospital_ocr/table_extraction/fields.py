from __future__ import annotations

import re

from hospital_ocr.models import OcrLine, TableGrid
from hospital_ocr.table_extraction.common import (
    DATE_RE,
    DOCUMENT_RE,
    TIME_RE,
    document_digits,
    is_header_or_metadata,
    name_from_text,
    ocr_number,
)
from hospital_ocr.table_extraction.types import (
    RowAnchor,
    SexResult,
    TableSchema,
)
from hospital_ocr.text import clean_display_text, normalize_text


def _split_document_and_age(digits: str) -> tuple[str, int | None]:
    if 6 <= len(digits) <= 9:
        return digits, None
    if 10 <= len(digits) <= 11:
        for suffix_length in (2, 1, 3):
            document = digits[:-suffix_length]
            age_text = digits[-suffix_length:]
            age = int(age_text)
            if 6 <= len(document) <= 9 and 0 <= age <= 115:
                return document, age
    return "", None


def schema_lines(
    lines: list[OcrLine],
    schema: TableSchema,
    field: str,
    grid: TableGrid | None = None,
) -> list[OcrLine]:
    column = schema.columns.get(field)
    if column is None:
        return []
    width = max(1, lines[0].image_width) if lines else 1
    selected: list[OcrLine] = []
    for line in lines:
        if is_header_or_metadata(line):
            continue
        if grid and column.grid_index is not None:
            in_column = grid.column_for_box(line.box) == column.grid_index
        else:
            in_column = column.start <= line.center_x / width < column.end
        if in_column:
            selected.append(line)
    return selected


def schema_text(
    lines: list[OcrLine],
    schema: TableSchema,
    field: str,
    grid: TableGrid | None = None,
) -> str:
    return clean_display_text(
        " ".join(
            line.text
            for line in sorted(
                schema_lines(lines, schema, field, grid),
                key=lambda item: item.center_x,
            )
        )
    )


def extract_document(lines: list[OcrLine]) -> str:
    candidates: list[tuple[int, float, str]] = []
    for line in lines:
        for digits in document_digits(line.text):
            document, _ = _split_document_and_age(digits)
            if document:
                preferred_length = int(len(document) in {7, 8})
                candidates.append((preferred_length, line.score, document))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def extract_schema_age(lines: list[OcrLine]) -> int | None:
    candidates: list[tuple[float, int]] = []
    for line in lines:
        if DATE_RE.search(line.text) or TIME_RE.search(line.text):
            continue
        without_document = DOCUMENT_RE.sub(" ", line.text)
        for token in re.findall(r"[A-Za-z]?\d{1,3}|[A-Za-z]\d", without_document):
            age = ocr_number(token)
            if age is not None:
                candidates.append((line.score, age))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def extract_schema_sex(
    lines: list[OcrLine],
    *,
    allow_ocr_confusions: bool = False,
) -> SexResult:
    candidates: dict[str, list[tuple[str, float]]] = {}
    direct_mapping = {"F": "F", "M": "M", "H": "M"}
    confusion_mapping = {"T": "F", "E": "F", "P": "F", "N": "M"}
    for line in lines:
        marker = re.sub(r"[^A-Za-z]", "", line.text).upper()
        value = direct_mapping.get(marker)
        if value is None and allow_ocr_confusions:
            value = confusion_mapping.get(marker)
        if value is not None:
            candidates.setdefault(value, []).append((marker, line.score))
    if len(candidates) > 1:
        markers = tuple(
            sorted(
                {
                    marker
                    for values in candidates.values()
                    for marker, _ in values
                }
            )
        )
        return SexResult("", markers, conflict=True)
    if not candidates:
        return SexResult("")

    value, values = next(iter(candidates.items()))
    canonical_present = any(marker == value for marker, _ in values)
    normalized_from = (
        ()
        if canonical_present
        else tuple(
            dict.fromkeys(
                marker
                for marker, _ in sorted(
                    values,
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        )
    )
    return SexResult(value, normalized_from)


def average_score(lines: list[OcrLine], fallback: float = 0.0) -> float:
    scores = [line.score for line in lines if line.text.strip()]
    return sum(scores) / len(scores) if scores else fallback


def headerless_field_lines(
    lines: list[OcrLine],
    anchor: RowAnchor,
    grid: TableGrid | None = None,
) -> list[OcrLine]:
    width = max(1, anchor.line.image_width)
    name_grid_column = (
        grid.column_index(anchor.line.center_x, anchor.line.center_y)
        if grid
        else None
    )
    selected: list[OcrLine] = []
    for line in lines:
        if line is anchor.line or is_header_or_metadata(line):
            continue
        if grid and name_grid_column is not None:
            same_name_column = (
                grid.column_index(line.center_x, line.center_y)
                == name_grid_column
            )
        else:
            same_name_column = (
                bool(name_from_text(line.text))
                and abs(line.box[0] - anchor.line.box[0]) <= width * 0.08
            )
        if not same_name_column:
            selected.append(line)
    return selected


def extract_semantic_age(
    lines: list[OcrLine],
) -> tuple[int | None, str]:
    candidates: list[tuple[int, float, int, str]] = []
    for line in lines:
        if DATE_RE.search(line.text) or TIME_RE.search(line.text):
            continue
        normalized = normalize_text(DOCUMENT_RE.sub(" ", line.text))
        match = re.fullmatch(
            r"(?:edad\s*)?(?P<age>\d{1,3})\s*"
            r"(?P<unit>anos?|a|mes(?:es)?|dias?)?",
            normalized,
        )
        if match is None:
            continue
        age = int(match.group("age"))
        if not 0 <= age <= 115:
            continue
        unit = match.group("unit") or ""
        normalized_unit = (
            "meses"
            if unit.startswith("mes")
            else "días"
            if unit.startswith("dia")
            else "años"
        )
        candidates.append((int(bool(unit)), line.score, age, normalized_unit))
    if not candidates:
        return None, ""
    candidates.sort(reverse=True)
    if (
        len(candidates) > 1
        and candidates[0][0] == 0
        and candidates[1][0] == 0
    ):
        return None, ""
    _, _, age, unit = candidates[0]
    return age, unit


def joined_cell_text(lines: list[OcrLine]) -> str:
    return clean_display_text(
        " ".join(
            line.text for line in sorted(lines, key=lambda item: item.center_x)
        )
    )
