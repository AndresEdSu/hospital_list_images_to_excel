from __future__ import annotations

import re
from statistics import median

from hospital_ocr.matching import match_place
from hospital_ocr.models import GridBoundary, OcrLine, Place, TableGrid
from hospital_ocr.table_extraction.common import (
    DOCUMENT_RE,
    HEADER_ALIASES,
    HEADER_WORDS,
    is_header_or_metadata,
    name_from_text,
    ocr_number,
    text_height,
)
from hospital_ocr.table_extraction.detection import (
    headerless_name_candidates,
)
from hospital_ocr.table_extraction.schema import header_candidate
from hospital_ocr.table_extraction.types import Column, RowAnchor, TableSchema
from hospital_ocr.text import clean_display_text, normalize_text


ADMIN_NAME_PREFIX_TOKENS = {
    "cama",
    "camilla",
    "camille",
    "canilla",
    "cana",
    "cann",
    "camn",
    "camo",
    "cimilla",
    "camiliae",
    "chna",
    "chnilh",
    "chnn",
    "chime",
}
ADMIN_NAME_PREFIX_GLUES = ("hnill",)
ADMIN_NAME_PREFIX_INTROS = {"tem", "item"}
ADMIN_CONTEXT_CONNECTORS = {"ca", "da", "de", "do", "gwa", "in", "lo", "oda", "qx"}
ADMIN_CONTEXT_SUFFIXES = {
    "anes",
    "catia",
    "cavers",
    "cmie",
    "enrniz",
    "gnara",
    "gnnia",
    "grvaiz",
    "guaira",
    "guni",
    "gunie",
    "gunina",
    "guraia",
    "gune",
    "gnaiz",
}


def _name_columns(schema: TableSchema) -> list[tuple[str, Column]]:
    if "name" in schema.columns:
        return [("name", schema.columns["name"])]
    return [
        (field, schema.columns[field])
        for field in ("given_names", "surnames")
        if field in schema.columns
    ]


def has_leading_index(text: str) -> bool:
    match = re.match(
        r"^\s*\d{1,3}(?:\s*[.):\-]\s*|\s+)",
        text,
    )
    if match is None:
        return False
    following = normalize_text(text[match.end() :]).split()
    return not (
        following
        and following[0] in {
            "a",
            "ano",
            "anos",
            "mes",
            "meses",
            "dia",
            "dias",
        }
    )


def row_index_lines(
    lines: list[OcrLine],
    anchor: RowAnchor,
    schema: TableSchema | None,
    headerless_index_ids: set[int] | None = None,
) -> list[OcrLine]:
    if schema is None:
        index_ids = headerless_index_ids or set()
        width = max(1, anchor.line.image_width)
        return [
            line
            for line in lines
            if id(line) in index_ids
            or (
                re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text)
                and line.box[2] <= anchor.line.box[0]
                and line.center_x / width < 0.16
            )
        ]

    width = max(1, anchor.line.image_width)
    age_column = schema.columns.get("age")
    indexes: list[OcrLine] = []
    for line in lines:
        if not re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text):
            continue
        normalized_center = line.center_x / width
        if (
            age_column
            and age_column.start <= normalized_center < age_column.end
        ):
            continue
        if line.box[2] <= anchor.line.box[0] or normalized_center < 0.16:
            indexes.append(line)
    return indexes


def header_cutoff(lines: list[OcrLine]) -> float | None:
    header_lines = []
    for line in lines:
        words = set(normalize_text(line.text).split())
        if words & HEADER_WORDS:
            header_lines.append(line)
    if not header_lines:
        return None
    return max(line.center_y for line in header_lines) + 8


def _looks_like_admin_name_prefix(token: str) -> bool:
    return normalize_text(token) in ADMIN_NAME_PREFIX_TOKENS


def _strip_glued_admin_prefix(token: str) -> str:
    normalized = normalize_text(token).replace(" ", "")
    for prefix in ADMIN_NAME_PREFIX_GLUES:
        if normalized.startswith(prefix) and len(normalized) - len(prefix) >= 4:
            return token[len(prefix) :]
    return token


def _has_admin_name_prefix(name: str) -> bool:
    tokens = name.split()
    if not tokens:
        return False
    index = 0
    if (
        len(tokens) > 1
        and normalize_text(tokens[0]) in ADMIN_NAME_PREFIX_INTROS
    ):
        index = 1
    return _looks_like_admin_name_prefix(tokens[index]) or (
        _strip_glued_admin_prefix(tokens[index]) != tokens[index]
    )


def _tail_matches_place(tokens: list[str], start: int, places: list[Place]) -> bool:
    return bool(places and match_place(" ".join(tokens[start:]), places))


def _strip_trailing_admin_context(
    tokens: list[str],
    places: list[Place],
) -> list[str]:
    for index in range(2, len(tokens)):
        normalized = normalize_text(tokens[index])
        if normalized in ADMIN_CONTEXT_SUFFIXES:
            return tokens[:index]
        if _tail_matches_place(tokens, index, places):
            return tokens[:index]
        if normalized not in ADMIN_CONTEXT_CONNECTORS:
            continue
        if index + 1 >= len(tokens):
            return tokens[:index]
        next_normalized = normalize_text(tokens[index + 1])
        if (
            next_normalized in ADMIN_CONTEXT_SUFFIXES
            or _tail_matches_place(tokens, index + 1, places)
            or any(
                normalize_text(token) in ADMIN_CONTEXT_SUFFIXES
                for token in tokens[index + 2 : index + 4]
            )
        ):
            return tokens[:index]
    return tokens


def _strip_repeated_admin_name_noise(
    name: str,
    places: list[Place],
) -> str:
    tokens = name.split()
    if not tokens:
        return ""
    index = 0
    if (
        len(tokens) > 1
        and normalize_text(tokens[0]) in ADMIN_NAME_PREFIX_INTROS
        and _looks_like_admin_name_prefix(tokens[1])
    ):
        index = 1
    while index < len(tokens):
        token = tokens[index]
        stripped = _strip_glued_admin_prefix(token)
        if stripped != token:
            tokens[index] = stripped
            break
        if not _looks_like_admin_name_prefix(token):
            break
        index += 1
    tokens = tokens[index:]
    tokens = _strip_trailing_admin_context(tokens, places)
    return clean_display_text(" ".join(tokens))


def _clean_repeated_admin_name_prefixes(
    candidates: list[RowAnchor],
    places: list[Place],
) -> list[RowAnchor]:
    if len(candidates) < 4:
        return candidates
    prefixed_count = sum(
        _has_admin_name_prefix(candidate.name)
        for candidate in candidates
    )
    if prefixed_count < max(3, round(len(candidates) * 0.25)):
        return candidates

    cleaned: list[RowAnchor] = []
    for candidate in candidates:
        clean_name = _strip_repeated_admin_name_noise(candidate.name, places)
        name_words = clean_name.split()
        if clean_name and not (len(name_words) == 1 and len(name_words[0]) < 6):
            cleaned.append(RowAnchor(candidate.line, clean_name))
    return cleaned or candidates


def find_row_anchors(
    lines: list[OcrLine],
    schema: TableSchema | None = None,
    grid: TableGrid | None = None,
    places: list[Place] | None = None,
) -> list[RowAnchor]:
    width = lines[0].image_width
    headerless_name_ids = (
        {
            id(line)
            for line in headerless_name_candidates(lines, grid, places)
        }
        if schema is None
        else set()
    )
    candidates: list[RowAnchor] = []
    for line in lines:
        name_field = ""
        if schema:
            normalized_center = line.center_x / width
            reaches_name_column = False
            for field, name_column in _name_columns(schema):
                if grid and name_column.grid_index is not None:
                    in_column = (
                        grid.column_for_box(line.box)
                        == name_column.grid_index
                    )
                else:
                    in_column = (
                        name_column.start
                        <= normalized_center
                        < name_column.end
                    )
                if in_column:
                    reaches_name_column = True
                    name_field = field
                    break
        else:
            reaches_name_column = id(line) in headerless_name_ids
        if not reaches_name_column or is_header_or_metadata(line):
            continue
        name = name_from_text(
            line.text,
            allow_short_single=name_field in {"given_names", "surnames"},
        )
        if name:
            candidates.append(RowAnchor(line, name))

    if schema is None:
        candidates = _clean_repeated_admin_name_prefixes(
            candidates,
            places or [],
        )

    if not candidates:
        return []
    typical_height = median(text_height(item.line) for item in candidates)
    same_row_tolerance = max(4.0, typical_height * 0.25)
    clusters: list[list[RowAnchor]] = []
    for candidate in sorted(candidates, key=lambda item: item.line.center_y):
        if (
            clusters
            and abs(
                candidate.line.center_y
                - median(item.line.center_y for item in clusters[-1])
            )
            <= same_row_tolerance
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    anchors: list[RowAnchor] = []
    for cluster in clusters:
        selected = max(
            cluster,
            key=lambda item: (len(item.name.split()), item.line.score),
        )
        combined_name = " ".join(
            dict.fromkeys(
                item.name
                for item in sorted(cluster, key=lambda item: item.line.center_x)
            )
        )
        anchors.append(RowAnchor(selected.line, combined_name))
    return anchors


def row_groups(
    lines: list[OcrLine],
    anchors: list[RowAnchor],
    grid: TableGrid | None = None,
) -> list[tuple[RowAnchor, list[OcrLine]]]:
    if grid:
        anchor_rows = [
            (anchor, grid.row_for_box(anchor.line.box))
            for anchor in anchors
        ]
        assigned = [
            (anchor, row)
            for anchor, row in anchor_rows
            if row is not None
        ]
        unique_rows = {row for _, row in assigned}
        if (
            len(assigned) >= max(2, round(len(anchors) * 0.70))
            and len(unique_rows) == len(assigned)
        ):
            lines_by_row: dict[int, list[OcrLine]] = {}
            for line in lines:
                row = grid.row_for_box(line.box)
                if row is not None:
                    lines_by_row.setdefault(row, []).append(line)
            if len(assigned) == len(anchors):
                return [
                    (anchor, lines_by_row.get(row, []))
                    for anchor, row in assigned
                ]

            assigned_rows = {id(anchor): row for anchor, row in assigned}
            centers = [anchor.line.center_y for anchor in anchors]
            groups: list[tuple[RowAnchor, list[OcrLine]]] = []
            for index, anchor in enumerate(anchors):
                row = assigned_rows.get(id(anchor))
                if row is not None:
                    groups.append((anchor, lines_by_row.get(row, [])))
                    continue
                lower = (
                    float("-inf")
                    if index == 0
                    else (centers[index - 1] + centers[index]) / 2
                )
                upper = (
                    float("inf")
                    if index == len(anchors) - 1
                    else (centers[index] + centers[index + 1]) / 2
                )
                groups.append(
                    (
                        anchor,
                        [
                            line
                            for line in lines
                            if lower < line.center_y <= upper
                        ],
                    )
                )
            return groups

    centers = [anchor.line.center_y for anchor in anchors]
    groups: list[tuple[RowAnchor, list[OcrLine]]] = []
    for index, anchor in enumerate(anchors):
        lower = (
            float("-inf")
            if index == 0
            else (centers[index - 1] + centers[index]) / 2
        )
        upper = (
            float("inf")
            if index == len(anchors) - 1
            else (centers[index] + centers[index + 1]) / 2
        )
        row_lines = [
            line for line in lines if lower < line.center_y <= upper
        ]
        groups.append((anchor, row_lines))
    return groups


def complete_cropped_top_row(
    lines: list[OcrLine],
    grid: TableGrid | None,
    places: list[Place],
) -> TableGrid | None:
    if grid is None or len(grid.horizontal) < 3 or not lines:
        return grid

    image_width = max(1, lines[0].image_width)
    image_height = max(1, lines[0].image_height)
    reference = image_width / 2
    positions = [
        boundary.coordinate_at(reference)
        for boundary in grid.horizontal
    ]
    spacings = [
        right - left
        for left, right in zip(positions, positions[1:], strict=False)
        if right > left
    ]
    if not spacings:
        return grid

    typical_spacing = median(spacings[: min(8, len(spacings))])
    first_position = positions[0]
    minimum_edge_gap = max(4.0, image_height * 0.004)
    if not (
        minimum_edge_gap
        < first_position
        < typical_spacing * 1.35
    ):
        return grid

    band_top = max(0.0, first_position - typical_spacing * 1.35)
    top_lines = [
        line
        for line in lines
        if band_top <= line.center_y < first_position
        and not is_header_or_metadata(line)
    ]
    name_lines = [
        line for line in top_lines if name_from_text(line.text)
    ]
    if not name_lines:
        return grid

    name_ids = {id(line) for line in name_lines}
    has_supporting_field = any(
        id(line) not in name_ids
        and (
            bool(DOCUMENT_RE.search(line.text))
            or ocr_number(line.text) is not None
            or bool(
                re.fullmatch(
                    r"\s*[MFH]\s*",
                    line.text,
                    re.IGNORECASE,
                )
            )
            or match_place(line.text, places) is not None
        )
        for line in top_lines
    )
    if not has_supporting_field:
        return grid

    family = grid.horizontal[: min(5, len(grid.horizontal))]
    slope = median(boundary.slope for boundary in family)
    target = max(0.0, first_position - typical_spacing)
    if first_position - target < minimum_edge_gap:
        return grid
    inferred = GridBoundary(
        slope=slope,
        intercept=target - slope * reference,
        support=min(0.50, median(boundary.support for boundary in family)),
    )
    return TableGrid(
        horizontal=(inferred, *grid.horizontal),
        vertical=grid.vertical,
        confidence=grid.confidence,
    )


def grid_header_row(
    lines: list[OcrLine],
    grid: TableGrid | None,
) -> int | None:
    if grid is None:
        return None
    fields_by_row: dict[int, set[str]] = {}
    text_by_row: dict[int, list[str]] = {}
    for line in lines:
        row = grid.row_for_box(line.box)
        if row is None:
            continue
        text_by_row.setdefault(row, []).append(normalize_text(line.text))
        candidate = header_candidate(line)
        if candidate is not None:
            fields_by_row.setdefault(row, set()).add(candidate.field)

    candidates: list[tuple[int, int]] = []
    for row, texts in text_by_row.items():
        fields = set(fields_by_row.get(row, set()))
        compact = re.sub(r"\s+", "", " ".join(texts))
        for field, aliases in HEADER_ALIASES.items():
            if any(
                re.sub(r"\s+", "", alias) in compact
                for alias in aliases
                if len(re.sub(r"\s+", "", alias)) >= 4
            ):
                fields.add(field)
        has_name = "name" in fields
        if has_name and len(fields) >= 2:
            candidates.append((len(fields), row))
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    return min(row for score, row in candidates if score == best_score)
