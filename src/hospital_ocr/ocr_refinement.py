from __future__ import annotations

import re

from hospital_ocr.handwriting import GridCell, TextRow
from hospital_ocr.models import OcrLine, TableGrid
from hospital_ocr.pipeline_types import OcrMode


GRID_REFINEMENT_THRESHOLDS: dict[OcrMode, float] = {
    "auto": 0.88,
    "handwritten": 0.96,
    "printed": 0.0,
}


def select_grid_cells_for_refinement(
    cells: list[GridCell],
    lines: list[OcrLine],
    grid: TableGrid,
    mode: OcrMode,
) -> list[GridCell]:
    threshold = GRID_REFINEMENT_THRESHOLDS[mode]
    by_cell: dict[tuple[int, int], list[OcrLine]] = {}
    row_line_counts: dict[int, int] = {}
    for line in lines:
        row = grid.row_for_box(line.box)
        column = grid.column_for_box(line.box)
        if row is None:
            continue
        row_line_counts[row] = row_line_counts.get(row, 0) + 1
        if column is not None:
            by_cell.setdefault((row, column), []).append(line)

    sex_marker_counts: dict[int, int] = {}
    for (_, column), candidates in by_cell.items():
        if is_sex_marker_candidate(candidates):
            sex_marker_counts[column] = sex_marker_counts.get(column, 0) + 1
    sex_columns = {
        column
        for column, count in sex_marker_counts.items()
        if count >= 2
    }

    minimum_row_lines = 2 if mode == "auto" else 3
    selected: list[GridCell] = []
    for cell in cells:
        candidates = by_cell.get(
            (cell.row_index, cell.column_index),
            [],
        )
        if (
            cell.row_index == 0
            or cell.column_index in sex_columns
            or row_line_counts.get(cell.row_index, 0) < minimum_row_lines
            or not candidates
            or has_structured_field_candidate(candidates)
            or max(line.score for line in candidates) < threshold
        ):
            selected.append(cell)
    return selected


def has_structured_field_candidate(lines: list[OcrLine]) -> bool:
    text = " ".join(line.text for line in lines)
    if re.search(r"\d", text):
        return True
    return is_sex_marker_candidate(lines)


def is_sex_marker_candidate(lines: list[OcrLine]) -> bool:
    text = " ".join(line.text for line in lines)
    marker = re.sub(r"[^A-Za-z]", "", text).upper()
    return marker in {"M", "F", "H", "T", "E", "N"}


def select_rows_for_refinement(
    rows: list[TextRow],
    lines: list[OcrLine],
    *,
    threshold: float,
) -> list[TextRow]:
    selected: list[TextRow] = []
    for row in rows:
        left, top, right, bottom = row.box
        candidates = [
            line
            for line in lines
            if top <= line.center_y <= bottom
            and line.box[2] >= left
            and line.box[0] <= right
        ]
        if (
            not candidates
            or max(line.score for line in candidates) < threshold
        ):
            selected.append(row)
    return selected
