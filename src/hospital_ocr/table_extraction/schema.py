from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import median

from hospital_ocr.models import OcrLine, TableGrid
from hospital_ocr.table_extraction.common import (
    DATE_RE,
    DOCUMENT_RE,
    HEADER_ALIASES,
    TIME_RE,
    ocr_number,
    text_height,
)
from hospital_ocr.table_extraction.types import (
    Column,
    HeaderCandidate,
    TableSchema,
)
from hospital_ocr.text import clean_display_text, normalize_text


def header_candidate(line: OcrLine) -> HeaderCandidate | None:
    normalized = normalize_text(line.text)
    if not normalized:
        return None
    best_field = ""
    best_score = 0.0
    for field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if normalized == alias:
                score = 1.0
            elif re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized):
                score = 0.94
            else:
                score = SequenceMatcher(None, normalized, alias).ratio()
                if score < 0.78:
                    continue
            if score > best_score:
                best_field = field
                best_score = score
    if not best_field:
        return None
    return HeaderCandidate(best_field, line, best_score)


def _header_baseline(
    candidates: list[HeaderCandidate],
) -> tuple[float, float]:
    points = [
        (candidate.line.center_x, candidate.line.center_y)
        for candidate in candidates
    ]
    slopes = [
        (right_y - left_y) / (right_x - left_x)
        for index, (left_x, left_y) in enumerate(points)
        for right_x, right_y in points[index + 1 :]
        if abs(right_x - left_x) >= 20
    ]
    slope = median(slopes) if slopes else 0.0
    intercept = median(y - slope * x for x, y in points)
    return slope, intercept


def _looks_like_unknown_header(line: OcrLine) -> bool:
    normalized = normalize_text(line.text)
    if not normalized or DATE_RE.search(line.text) or TIME_RE.search(line.text):
        return False
    if DOCUMENT_RE.search(line.text):
        return False
    letters = sum(character.isalpha() for character in normalized)
    words = normalized.split()
    return letters >= 3 and len(words) <= 5 and len(normalized) <= 50


def _unknown_header_candidates(
    lines: list[OcrLine],
    known: list[HeaderCandidate],
    all_known: list[HeaderCandidate],
    typical_height: float,
) -> list[HeaderCandidate]:
    if len(known) < 3:
        return []
    slope, intercept = _header_baseline(known)
    width = max(1, known[0].line.image_width)
    vertical_tolerance = max(10.0, typical_height)
    minimum_x = min(item.line.center_x for item in known)
    maximum_x = max(item.line.center_x for item in known)
    known_line_ids = {id(item.line) for item in all_known}
    known_centers = [item.line.center_x for item in known]
    minimum_horizontal_gap = max(20.0, width * 0.025)

    eligible: list[tuple[float, OcrLine]] = []
    for line in lines:
        if id(line) in known_line_ids or not _looks_like_unknown_header(line):
            continue
        if not minimum_x < line.center_x < maximum_x:
            continue
        if min(abs(line.center_x - center) for center in known_centers) < (
            minimum_horizontal_gap
        ):
            continue
        expected_y = slope * line.center_x + intercept
        distance = abs(line.center_y - expected_y)
        if distance <= vertical_tolerance:
            eligible.append((distance, line))

    clusters: list[list[tuple[float, OcrLine]]] = []
    for candidate in sorted(eligible, key=lambda item: item[1].center_x):
        if (
            clusters
            and abs(
                candidate[1].center_x
                - median(item[1].center_x for item in clusters[-1])
            )
            <= width * 0.04
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    unknown: list[HeaderCandidate] = []
    for index, cluster in enumerate(clusters, start=1):
        distance, line = min(cluster, key=lambda item: item[0])
        geometric_score = max(
            0.75,
            1.0 - (distance / vertical_tolerance) * 0.25,
        )
        unknown.append(
            HeaderCandidate(
                f"ignored_unknown_{index}",
                line,
                min(0.90, line.score * geometric_score),
            )
        )
    return unknown


def infer_schema(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> TableSchema | None:
    candidates = [
        candidate
        for line in lines
        if (candidate := header_candidate(line)) is not None
    ]
    if not candidates:
        return None

    typical_height = median(text_height(item.line) for item in candidates)
    tolerance = max(8.0, typical_height * 1.25)
    clusters: list[list[HeaderCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.line.center_y):
        if (
            clusters
            and candidate.line.center_y
            - max(item.line.center_y for item in clusters[-1])
            <= tolerance
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    eligible = [
        cluster
        for cluster in clusters
        if len({item.field for item in cluster}) >= 2
        and "name" in {item.field for item in cluster}
    ]
    if not eligible:
        return None
    header_cluster = max(
        eligible,
        key=lambda cluster: (
            len({item.field for item in cluster}),
            sum(item.score for item in cluster),
        ),
    )
    header_cluster.extend(
        _unknown_header_candidates(
            lines,
            header_cluster,
            candidates,
            typical_height,
        )
    )
    width = max(1, header_cluster[0].line.image_width)
    grouped: dict[str, list[HeaderCandidate]] = {}
    for candidate in header_cluster:
        grouped.setdefault(candidate.field, []).append(candidate)

    centers = {
        field: sum(item.line.center_x for item in items) / len(items) / width
        for field, items in grouped.items()
    }
    vertical_centers = {
        field: sum(item.line.center_y for item in items) / len(items)
        for field, items in grouped.items()
    }
    ordered = sorted(centers, key=centers.get)
    columns: dict[str, Column] = {}
    for index, field in enumerate(ordered):
        start = (
            0.0
            if index == 0
            else (centers[ordered[index - 1]] + centers[field]) / 2
        )
        end = (
            1.0
            if index == len(ordered) - 1
            else (centers[field] + centers[ordered[index + 1]]) / 2
        )
        confidence = max(item.score for item in grouped[field])
        grid_index = (
            grid.column_index(
                centers[field] * width,
                vertical_centers[field],
            )
            if grid
            else None
        )
        columns[field] = Column(
            field,
            centers[field],
            max(0.0, start),
            min(1.0, end),
            confidence,
            grid_index,
        )
    schema = TableSchema(
        columns=columns,
        header_bottom=max(item.line.box[3] for item in header_cluster) + 4,
        confidence=sum(column.confidence for column in columns.values())
        / len(columns),
    )
    return complete_partial_schema(schema, lines, grid)


def complete_partial_schema(
    schema: TableSchema,
    lines: list[OcrLine],
    grid: TableGrid | None,
) -> TableSchema:
    if grid is None or "name" not in schema.columns:
        return schema

    used_grid_indexes = {
        column.grid_index
        for field, column in schema.columns.items()
        if column.grid_index is not None
        and not field.startswith("ignored_unknown_")
    }
    name_grid_index = schema.columns["name"].grid_index
    by_column_and_row: dict[int, dict[int, list[OcrLine]]] = {}
    for line in lines:
        if line.center_y <= schema.header_bottom:
            continue
        column_index = grid.column_for_box(line.box)
        row_index = grid.row_for_box(line.box)
        if column_index is None or row_index is None:
            continue
        by_column_and_row.setdefault(column_index, {}).setdefault(
            row_index,
            [],
        ).append(line)

    candidate_indexes = [
        index
        for index in by_column_and_row
        if index not in used_grid_indexes
        and (
            name_grid_index is None
            or index > name_grid_index
        )
    ]
    if not candidate_indexes:
        return schema

    row_texts = {
        column_index: [
            clean_display_text(
                " ".join(
                    line.text
                    for line in sorted(row_lines, key=lambda item: item.center_x)
                )
            )
            for row_lines in rows.values()
        ]
        for column_index, rows in by_column_and_row.items()
    }

    selected: dict[str, tuple[int, float]] = {}

    if "document" not in schema.columns:
        document_candidates = []
        for column_index in candidate_indexes:
            texts = row_texts[column_index]
            matches = sum(bool(DOCUMENT_RE.search(text)) for text in texts)
            if matches >= 2:
                document_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        -column_index,
                        column_index,
                    )
                )
        if document_candidates:
            matches, ratio, _, column_index = max(document_candidates)
            selected["document"] = (
                column_index,
                min(0.94, 0.72 + 0.18 * ratio + 0.01 * matches),
            )

    unavailable = {
        *used_grid_indexes,
        *(column_index for column_index, _ in selected.values()),
    }
    if "sex" not in schema.columns:
        sex_candidates = []
        for column_index in candidate_indexes:
            if column_index in unavailable:
                continue
            texts = row_texts[column_index]
            matches = sum(
                re.sub(r"[^A-Za-z]", "", text).upper()
                in {"M", "F", "H", "N"}
                for text in texts
            )
            if matches >= 2:
                sex_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        column_index,
                    )
                )
        if sex_candidates:
            matches, ratio, column_index = max(sex_candidates)
            selected["sex"] = (
                column_index,
                min(0.94, 0.72 + 0.18 * ratio + 0.01 * matches),
            )

    unavailable.update(
        column_index for column_index, _ in selected.values()
    )
    if "age" not in schema.columns:
        age_candidates = []
        for column_index in candidate_indexes:
            if column_index in unavailable:
                continue
            texts = row_texts[column_index]
            ages = [
                ocr_number(text)
                for text in texts
                if not DOCUMENT_RE.search(text)
            ]
            matches = sum(age is not None for age in ages)
            if matches >= 2:
                age_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        -column_index,
                        column_index,
                    )
                )
        if age_candidates:
            matches, ratio, _, column_index = max(age_candidates)
            selected["age"] = (
                column_index,
                min(0.92, 0.70 + 0.17 * ratio + 0.01 * matches),
            )

    if not selected:
        return schema

    columns = dict(schema.columns)
    width = max(1, lines[0].image_width)
    representative_y = schema.header_bottom + 1
    for field, (grid_index, confidence) in selected.items():
        columns = {
            existing_field: column
            for existing_field, column in columns.items()
            if not (
                existing_field.startswith("ignored_unknown_")
                and column.grid_index == grid_index
            )
        }
        left = grid.vertical[grid_index].coordinate_at(representative_y)
        right = grid.vertical[grid_index + 1].coordinate_at(representative_y)
        start = max(0.0, min(left, right) / width)
        end = min(1.0, max(left, right) / width)
        columns[field] = Column(
            field=field,
            center=(start + end) / 2,
            start=start,
            end=end,
            confidence=confidence,
            grid_index=grid_index,
        )
    return TableSchema(
        columns=columns,
        header_bottom=schema.header_bottom,
        confidence=sum(column.confidence for column in columns.values())
        / len(columns),
    )


def has_table_header(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> bool:
    if infer_schema(lines, grid) is not None:
        return True
    normalized = normalize_text(" ".join(line.text for line in lines))
    has_name = "nombre" in normalized and "apellido" in normalized
    supporting = sum(
        word in normalized
        for word in ("edad", "sexo", "procedencia", "plan", "telefono")
    )
    return has_name and supporting >= 1
