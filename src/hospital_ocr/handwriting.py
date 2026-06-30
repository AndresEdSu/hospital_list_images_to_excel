from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hospital_ocr.models import OcrLine


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


def merge_row_ocr(
    initial_lines: list[OcrLine],
    row_lines: list[OcrLine],
    rows: list[TextRow],
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
            total_characters = sum(max(1, len(line.text.strip())) for line in ordered)
            combined_score = sum(
                line.score * max(1, len(line.text.strip())) for line in ordered
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
