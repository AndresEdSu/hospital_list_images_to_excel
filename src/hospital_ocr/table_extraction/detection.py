from __future__ import annotations

import re
from statistics import median

from hospital_ocr.matching import match_place
from hospital_ocr.models import OcrLine, Place, TableGrid
from hospital_ocr.table_extraction.common import (
    document_digits,
    is_header_or_metadata,
    name_from_text,
)
from hospital_ocr.table_extraction.schema import has_table_header
from hospital_ocr.text import normalize_text


def headerless_name_candidates(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
    places: list[Place] | None = None,
) -> list[OcrLine]:
    width = max(1, lines[0].image_width)
    places = places or []
    candidates = [
        line
        for line in lines
        if name_from_text(line.text)
        and not is_header_or_metadata(line)
    ]
    if not candidates:
        return []

    clusters: list[list[OcrLine]] = []
    if grid:
        by_column: dict[int, list[OcrLine]] = {}
        for line in candidates:
            column = grid.column_index(line.center_x, line.center_y)
            if column is not None:
                by_column.setdefault(column, []).append(line)
        clusters.extend(by_column.values())

    geometric_clusters: list[list[OcrLine]] = []
    for line in sorted(candidates, key=lambda item: item.box[0]):
        matching = next(
            (
                cluster
                for cluster in geometric_clusters
                if abs(
                    line.box[0] - median(item.box[0] for item in cluster)
                )
                <= width * 0.06
            ),
            None,
        )
        if matching is None:
            geometric_clusters.append([line])
        else:
            matching.append(line)
    clusters.extend(geometric_clusters)

    unique_clusters: list[list[OcrLine]] = []
    seen_clusters: set[tuple[int, ...]] = set()
    for cluster in clusters:
        identity = tuple(sorted(id(line) for line in cluster))
        if identity not in seen_clusters:
            seen_clusters.add(identity)
            unique_clusters.append(cluster)

    def column_score(cluster: list[OcrLine]) -> tuple[float, float, float]:
        names = [normalize_text(name_from_text(line.text)) for line in cluster]
        unique_ratio = len(set(names)) / len(names)
        multiword_ratio = sum(len(name.split()) >= 2 for name in names) / len(names)
        place_ratio = (
            sum(match_place(name, places) is not None for name in names)
            / len(names)
            if places
            else 0.0
        )
        weighted_rows = (
            len(cluster)
            * (0.55 + 0.45 * unique_ratio)
            * (0.65 + 0.35 * multiword_ratio)
            * (1.0 - 0.55 * place_ratio)
        )
        return (
            weighted_rows,
            regular_row_ratio(cluster),
            multiword_ratio,
        )

    return max(unique_clusters, key=column_score)


def _alignment_ratio(values: list[float], tolerance: float) -> float:
    if not values:
        return 0.0
    center = median(values)
    return sum(abs(value - center) <= tolerance for value in values) / len(values)


def regular_row_ratio(lines: list[OcrLine]) -> float:
    centers = sorted({round(line.center_y, 1) for line in lines})
    if len(centers) < 4:
        return 0.0
    gaps = [
        current - previous
        for previous, current in zip(centers, centers[1:], strict=False)
        if current > previous
    ]
    if not gaps:
        return 0.0
    typical_gap = median(gaps)
    tolerance = max(4.0, typical_gap * 0.35)
    return sum(
        abs(gap - typical_gap) <= tolerance
        or abs(gap - typical_gap * 2) <= tolerance * 1.5
        for gap in gaps
    ) / len(gaps)


def _sequential_index_ratio(lines: list[OcrLine]) -> float:
    values = [
        int(re.sub(r"\D", "", line.text))
        for line in sorted(lines, key=lambda item: item.center_y)
    ]
    if len(values) < 2:
        return 0.0
    return sum(
        current == previous + 1
        for previous, current in zip(values, values[1:], strict=False)
    ) / (len(values) - 1)


def infer_headerless_index_ids(lines: list[OcrLine]) -> set[int]:
    if not lines:
        return set()
    width = max(1, lines[0].image_width)
    candidates = [
        line
        for line in lines
        if re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text)
    ]
    clusters: list[list[OcrLine]] = []
    for line in sorted(candidates, key=lambda item: item.center_x):
        matching = next(
            (
                cluster
                for cluster in clusters
                if abs(
                    line.center_x - median(item.center_x for item in cluster)
                )
                <= width * 0.04
            ),
            None,
        )
        if matching is None:
            clusters.append([line])
        else:
            matching.append(line)

    sequential = [
        cluster
        for cluster in clusters
        if len(cluster) >= 4 and _sequential_index_ratio(cluster) >= 0.70
    ]
    if not sequential:
        return set()
    selected = max(
        sequential,
        key=lambda cluster: (
            len(cluster),
            _sequential_index_ratio(cluster),
        ),
    )
    return {id(line) for line in selected}


def _has_repeated_auxiliary_column(
    lines: list[OcrLine],
    name_candidates: list[OcrLine],
    grid: TableGrid | None = None,
) -> bool:
    if not name_candidates:
        return False
    width = max(1, lines[0].image_width)
    name_ids = {id(line) for line in name_candidates}
    minimum_y = min(line.center_y for line in name_candidates)
    maximum_y = max(line.center_y for line in name_candidates)
    bins: dict[int, int] = {}
    for line in lines:
        if id(line) in name_ids or not minimum_y <= line.center_y <= maximum_y:
            continue
        if re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text):
            continue
        if not normalize_text(line.text):
            continue
        grid_column = grid.column_for_box(line.box) if grid else None
        bucket = (
            grid_column
            if grid_column is not None
            else round((line.box[0] / width) / 0.04)
        )
        bins[bucket] = bins.get(bucket, 0) + 1
    minimum_repetitions = max(3, round(len(name_candidates) * 0.30))
    return any(count >= minimum_repetitions for count in bins.values())


def looks_like_table(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
    places: list[Place] | None = None,
) -> bool:
    if not lines:
        return False
    if has_table_header(lines, grid):
        return True

    width = lines[0].image_width
    row_index_ids = infer_headerless_index_ids(lines)
    sex_markers = [
        line.center_x
        for line in lines
        if re.fullmatch(r"\s*[MFH]\s*", line.text, re.IGNORECASE)
    ]
    aligned_sex_markers = (
        len(sex_markers) >= 4
        and _alignment_ratio(sex_markers, width * 0.06) >= 0.75
    )
    name_candidates = headerless_name_candidates(lines, grid, places)
    document_markers = sum(
        bool(document_digits(line.text))
        for line in lines
    )
    score = 2 if grid and grid.confidence >= 0.65 else 0
    if len(name_candidates) >= 5:
        score += 2
    if _alignment_ratio(
        [line.box[0] for line in name_candidates],
        width * 0.06,
    ) >= 0.75:
        score += 2
    if regular_row_ratio(name_candidates) >= 0.65:
        score += 2
    if len(row_index_ids) >= 4:
        score += 2
    if aligned_sex_markers:
        score += 1
    if document_markers >= 3:
        score += 1
    if _has_repeated_auxiliary_column(lines, name_candidates, grid):
        score += 1

    return score >= 6
