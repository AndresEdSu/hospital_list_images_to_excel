from __future__ import annotations

import re
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from hospital_ocr.models import GridBoundary, OcrLine, TableGrid


GRID_DOCUMENT_RE = re.compile(
    r"(?<!\d)(?:[VEve]\s*[-.]?\s*)?\d(?:[.,\-Â·]?\d){5,10}(?!\d)"
)


@dataclass(frozen=True)
class TextRow:
    box: tuple[int, int, int, int]
    strength: float
    baseline_y: float | None = None

    @property
    def center_y(self) -> float:
        if self.baseline_y is not None:
            return self.baseline_y
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class GridCell:
    row_index: int
    column_index: int
    corners: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    @property
    def target_width(self) -> int:
        top_left, top_right, bottom_right, bottom_left = self.corners
        return max(
            12,
            round(
                (
                    hypot(
                        top_right[0] - top_left[0],
                        top_right[1] - top_left[1],
                    )
                    + hypot(
                        bottom_right[0] - bottom_left[0],
                        bottom_right[1] - bottom_left[1],
                    )
                )
                / 2
            ),
        )

    @property
    def target_height(self) -> int:
        top_left, top_right, bottom_right, bottom_left = self.corners
        return max(
            12,
            round(
                (
                    hypot(
                        bottom_left[0] - top_left[0],
                        bottom_left[1] - top_left[1],
                    )
                    + hypot(
                        bottom_right[0] - top_right[0],
                        bottom_right[1] - top_right[1],
                    )
                )
                / 2
            ),
        )


def _grid_intersection(
    horizontal: GridBoundary,
    vertical: GridBoundary,
) -> tuple[float, float]:
    horizontal_slope = horizontal.slope
    horizontal_intercept = horizontal.intercept
    vertical_slope = vertical.slope
    vertical_intercept = vertical.intercept
    denominator = 1.0 - horizontal_slope * vertical_slope
    if abs(denominator) < 1e-6:
        y = horizontal_intercept
    else:
        y = (
            horizontal_slope * vertical_intercept
            + horizontal_intercept
        ) / denominator
    return vertical_slope * y + vertical_intercept, y


def cells_from_grid(grid: TableGrid) -> list[GridCell]:
    cells: list[GridCell] = []
    for row_index, (upper, lower) in enumerate(
        zip(grid.horizontal, grid.horizontal[1:], strict=False)
    ):
        for column_index, (left, right) in enumerate(
            zip(grid.vertical, grid.vertical[1:], strict=False)
        ):
            cells.append(
                GridCell(
                    row_index=row_index,
                    column_index=column_index,
                    corners=(
                        _grid_intersection(upper, left),
                        _grid_intersection(upper, right),
                        _grid_intersection(lower, right),
                        _grid_intersection(lower, left),
                    ),
                )
            )
    return cells


def detect_text_rows(image_path: Path) -> list[TextRow]:
    """Estimate handwritten/printed text baselines without assuming a table."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    profile = np.zeros(height, dtype=np.float32)
    max_component_height = max(20, round(height * 0.09))
    max_component_width = max(80, round(width * 0.18))
    for index in range(1, count):
        _, _, component_width, component_height, area = stats[index]
        if not (
            3 <= component_height <= max_component_height
            and 2 <= component_width <= max_component_width
            and area >= 8
        ):
            continue
        center_y = int(round(float(centroids[index][1])))
        if 0 <= center_y < height:
            profile[center_y] += min(float(area), 100.0)

    smooth_size = max(15, round(height * 0.038))
    if smooth_size % 2 == 0:
        smooth_size += 1
    smoothed = np.convolve(
        profile,
        np.ones(smooth_size, dtype=np.float32) / smooth_size,
        mode="same",
    )
    maximum = float(smoothed.max(initial=0.0))
    if maximum <= 0:
        return []

    minimum_distance = max(35, round(height * 0.065))
    minimum_strength = max(4.0, maximum * 0.12)
    candidates = np.flatnonzero(smoothed >= minimum_strength)
    selected: list[int] = []
    for position in sorted(
        candidates.tolist(),
        key=lambda item: float(smoothed[item]),
        reverse=True,
    ):
        if all(abs(position - existing) >= minimum_distance for existing in selected):
            selected.append(position)
        if len(selected) >= 60:
            break
    selected.sort()
    if len(selected) < 3:
        return []

    rows: list[TextRow] = []
    half_default = max(24, minimum_distance // 2)
    for index, center_y in enumerate(selected):
        upper_midpoint = (
            (selected[index - 1] + center_y) // 2
            if index
            else center_y - half_default
        )
        lower_midpoint = (
            (center_y + selected[index + 1]) // 2
            if index + 1 < len(selected)
            else center_y + half_default
        )
        top = max(0, upper_midpoint)
        bottom = min(height, lower_midpoint)
        if bottom - top < 18:
            continue
        rows.append(
            TextRow(
                box=(0, top, width, bottom),
                strength=float(smoothed[center_y]),
                baseline_y=float(center_y),
            )
        )
    return rows


def rows_from_grid(image_path: Path, grid: TableGrid) -> list[TextRow]:
    """Convert detected grid bands into OCR row crops."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    image_height, image_width = image.shape
    rows: list[TextRow] = []
    for upper, lower in zip(
        grid.horizontal,
        grid.horizontal[1:],
        strict=False,
    ):
        middle_y = (
            upper.coordinate_at(image_width / 2)
            + lower.coordinate_at(image_width / 2)
        ) / 2
        vertical_positions = [
            boundary.coordinate_at(middle_y) for boundary in grid.vertical
        ]
        if len(vertical_positions) >= 2:
            left = max(0, round(min(vertical_positions)))
            right = min(image_width, round(max(vertical_positions)))
        else:
            left, right = 0, image_width
        if right - left < 40:
            continue

        top = max(
            0,
            round(
                min(
                    upper.coordinate_at(left),
                    upper.coordinate_at(right),
                )
            ),
        )
        bottom = min(
            image_height,
            round(
                max(
                    lower.coordinate_at(left),
                    lower.coordinate_at(right),
                )
            ),
        )
        if bottom - top < 18:
            continue
        rows.append(
            TextRow(
                box=(left, top, right, bottom),
                strength=(upper.support + lower.support) / 2,
                baseline_y=middle_y,
            )
        )
    return rows


def row_ocr_coverage(lines: list[OcrLine], rows: list[TextRow]) -> float:
    if not rows:
        return 1.0
    covered = sum(
        any(row.box[1] <= line.center_y < row.box[3] for line in lines)
        for row in rows
    )
    return covered / len(rows)


def needs_row_ocr(lines: list[OcrLine], rows: list[TextRow]) -> bool:
    if len(rows) < 4:
        return False

    coverage = row_ocr_coverage(lines, rows)
    merged_lines = 0
    for line in lines:
        overlapping_centers = sum(
            line.box[1] <= row.center_y <= line.box[3] for row in rows
        )
        if overlapping_centers >= 2:
            merged_lines += 1
    return coverage < 0.72 or merged_lines > 0


def row_segmentation_is_too_coarse(
    lines: list[OcrLine],
    rows: list[TextRow],
) -> bool:
    if len(rows) < 4 or len(lines) < len(rows) * 1.5:
        return False
    if row_ocr_coverage(lines, rows) < 0.85:
        return False

    populated_rows = 0
    coarse_rows = 0
    for row in rows:
        centers = sorted(
            line.center_y
            for line in lines
            if row.box[1] <= line.center_y < row.box[3]
        )
        if not centers:
            continue
        populated_rows += 1
        if (
            len(centers) >= 2
            and centers[-1] - centers[0] >= max(30.0, row.height * 0.28)
        ):
            coarse_rows += 1

    return (
        populated_rows >= 4
        and coarse_rows >= max(3, round(populated_rows * 0.45))
    )


def merge_row_ocr(
    initial_lines: list[OcrLine],
    row_lines: list[OcrLine],
    rows: list[TextRow],
    *,
    combine_rows: bool = True,
) -> list[OcrLine]:
    """Prefer the segmented OCR where it found text; retain useful full-page OCR."""
    replacements: list[OcrLine] = []
    replaced_rows: set[int] = set()
    for index, row in enumerate(rows):
        found = [
            line
            for line in row_lines
            if row.box[1] <= line.center_y < row.box[3] and line.text.strip()
        ]
        if found:
            logical_height = max(12, min(28, round(row.height * 0.3)))
            logical_top = round(row.center_y - logical_height / 2)
            logical_bottom = logical_top + logical_height
            ordered = sorted(found, key=lambda line: line.center_x)
            if combine_rows:
                total_characters = sum(
                    max(1, len(line.text.strip())) for line in ordered
                )
                combined_score = sum(
                    line.score * max(1, len(line.text.strip()))
                    for line in ordered
                ) / total_characters
                replacements.append(
                    OcrLine(
                        text=" ".join(line.text.strip() for line in ordered),
                        score=combined_score,
                        box=(
                            min(line.box[0] for line in ordered),
                            logical_top,
                            max(line.box[2] for line in ordered),
                            logical_bottom,
                        ),
                        image_width=ordered[0].image_width,
                        image_height=ordered[0].image_height,
                    )
                )
            else:
                replacements.extend(
                    OcrLine(
                        text=line.text.strip(),
                        score=line.score,
                        box=(
                            line.box[0],
                            logical_top,
                            line.box[2],
                            logical_bottom,
                        ),
                        image_width=line.image_width,
                        image_height=line.image_height,
                    )
                    for line in ordered
                )
            replaced_rows.add(index)

    retained: list[OcrLine] = []
    for line in initial_lines:
        line_rows = {
            index
            for index, row in enumerate(rows)
            if row.box[1] <= line.center_y < row.box[3]
        }
        if not line_rows or line_rows.isdisjoint(replaced_rows):
            retained.append(line)

    return sorted(
        [*retained, *replacements],
        key=lambda line: (line.center_y, line.center_x),
    )


def _grid_group_score(lines: list[OcrLine]) -> float:
    if not lines:
        return 0.0
    text = " ".join(
        line.text.strip()
        for line in sorted(lines, key=lambda item: item.center_x)
        if line.text.strip()
    )
    total_characters = sum(max(1, len(line.text.strip())) for line in lines)
    average_score = sum(
        line.score * max(1, len(line.text.strip())) for line in lines
    ) / total_characters
    normalized = re.sub(r"\s+", " ", text).strip()
    alphabetic_tokens = re.findall(r"[A-Za-zÃÃ‰ÃÃ“ÃšÃœÃ‘Ã¡Ã©Ã­Ã³ÃºÃ¼Ã±]{2,}", normalized)
    document = GRID_DOCUMENT_RE.search(normalized)
    marker = re.sub(r"[^A-Za-z]", "", normalized).upper()
    semantic_bonus = 0.0
    if document:
        document_digits = re.sub(r"\D", "", document.group())
        semantic_bonus += 0.30 if len(document_digits) in {7, 8} else 0.05
    if marker in {"M", "F", "H"}:
        semantic_bonus += 0.18
    if 2 <= len(alphabetic_tokens) <= 6:
        semantic_bonus += 0.10
    strange = sum(
        not (character.isalnum() or character.isspace() or character in ".-#")
        for character in normalized
    )
    penalty = min(0.20, strange * 0.03)
    return average_score + semantic_bonus - penalty


def merge_grid_ocr(
    initial_lines: list[OcrLine],
    refined_lines: list[OcrLine],
    grid: TableGrid,
) -> list[OcrLine]:
    """Select the strongest OCR candidate per physical grid cell."""
    initial_groups: dict[tuple[int, int], list[OcrLine]] = {}
    refined_groups: dict[tuple[int, int], list[OcrLine]] = {}
    retained_outside: list[OcrLine] = []

    for line in initial_lines:
        row = grid.row_for_box(line.box)
        column = grid.column_for_box(line.box)
        if row is None or column is None:
            retained_outside.append(line)
        else:
            initial_groups.setdefault((row, column), []).append(line)
    for line in refined_lines:
        row = grid.row_for_box(line.box)
        column = grid.column_for_box(line.box)
        if row is not None and column is not None:
            refined_groups.setdefault((row, column), []).append(line)

    selected: list[OcrLine] = list(retained_outside)
    for key in sorted({*initial_groups, *refined_groups}):
        initial = initial_groups.get(key, [])
        refined = refined_groups.get(key, [])
        if not initial:
            selected.extend(refined)
        elif (
            refined
            and _grid_group_score(refined)
            > _grid_group_score(initial) + 0.12
        ):
            selected.extend(refined)
        else:
            selected.extend(initial)
    return sorted(
        selected,
        key=lambda line: (line.center_y, line.center_x),
    )
