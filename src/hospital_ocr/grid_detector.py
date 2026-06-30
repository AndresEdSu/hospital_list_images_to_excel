from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from hospital_ocr.models import GridBoundary, TableGrid


@dataclass(frozen=True)
class _Segment:
    slope: float
    reference: float
    length: float


def _hough_segments(
    mask: np.ndarray,
    *,
    horizontal: bool,
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    dimension = width if horizontal else height
    detected = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 720,
        threshold=max(25, dimension // 18),
        minLineLength=max(30, int(dimension * 0.16)),
        maxLineGap=max(8, int(dimension * 0.035)),
    )
    if detected is None:
        return []
    return [tuple(int(value) for value in line[0]) for line in detected]


def _normalize_segments(
    raw: list[tuple[int, int, int, int]],
    *,
    horizontal: bool,
    reference_position: float,
) -> list[_Segment]:
    normalized: list[_Segment] = []
    for x1, y1, x2, y2 in raw:
        dx = x2 - x1
        dy = y2 - y1
        if horizontal:
            if abs(dx) < 1 or abs(dy / dx) > 0.20:
                continue
            slope = dy / dx
            reference = y1 + slope * (reference_position - x1)
        else:
            if abs(dy) < 1 or abs(dx / dy) > 0.20:
                continue
            slope = dx / dy
            reference = x1 + slope * (reference_position - y1)
        normalized.append(
            _Segment(
                slope=slope,
                reference=reference,
                length=hypot(dx, dy),
            )
        )
    if len(normalized) < 3:
        return normalized
    family_slope = median(item.slope for item in normalized)
    tolerance = 0.055 if horizontal else 0.07
    return [
        item
        for item in normalized
        if abs(item.slope - family_slope) <= tolerance
    ]


def _cluster_boundaries(
    segments: list[_Segment],
    *,
    reference_position: float,
    dimension: int,
    tolerance: float,
) -> tuple[GridBoundary, ...]:
    clusters: list[list[_Segment]] = []
    for segment in sorted(segments, key=lambda item: item.reference):
        if (
            clusters
            and abs(
                segment.reference
                - median(item.reference for item in clusters[-1])
            )
            <= tolerance
        ):
            clusters[-1].append(segment)
        else:
            clusters.append([segment])

    boundaries: list[GridBoundary] = []
    for cluster in clusters:
        support = min(1.0, sum(item.length for item in cluster) / dimension)
        if support < 0.18:
            continue
        total_length = sum(item.length for item in cluster)
        slope = sum(item.slope * item.length for item in cluster) / total_length
        reference = median(item.reference for item in cluster)
        intercept = reference - slope * reference_position
        boundaries.append(GridBoundary(slope, intercept, support))
    return tuple(
        sorted(
            boundaries,
            key=lambda line: line.coordinate_at(reference_position),
        )
    )


def _grid_confidence(
    horizontal: tuple[GridBoundary, ...],
    vertical: tuple[GridBoundary, ...],
) -> float:
    if len(horizontal) < 3 or len(vertical) < 2:
        return 0.0
    row_score = min(1.0, (len(horizontal) - 2) / 8)
    column_score = min(1.0, (len(vertical) - 1) / 6)
    support_values = [
        *(line.support for line in horizontal),
        *(line.support for line in vertical),
    ]
    support_score = sum(support_values) / len(support_values)
    return round(0.40 * row_score + 0.35 * column_score + 0.25 * support_score, 4)


def _save_debug_overlay(
    image: np.ndarray,
    grid: TableGrid,
    path: Path,
) -> None:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    for line in grid.horizontal:
        start = (0, int(round(line.coordinate_at(0))))
        end = (width - 1, int(round(line.coordinate_at(width - 1))))
        cv2.line(overlay, start, end, (0, 180, 0), 2)
    for line in grid.vertical:
        start = (int(round(line.coordinate_at(0))), 0)
        end = (int(round(line.coordinate_at(height - 1))), height - 1)
        cv2.line(overlay, start, end, (220, 80, 0), 2)
    cv2.putText(
        overlay,
        f"grid confidence: {grid.confidence:.0%}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def detect_table_grid(
    image_path: Path,
    debug_path: Path | None = None,
    *,
    minimum_confidence: float = 0.55,
) -> TableGrid | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    height, width = binary.shape
    horizontal_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(20, width // 35), 1),
        ),
    )
    vertical_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(20, height // 35)),
        ),
    )
    horizontal_raw = _hough_segments(horizontal_mask, horizontal=True)
    vertical_raw = _hough_segments(vertical_mask, horizontal=False)

    horizontal_segments = _normalize_segments(
        horizontal_raw,
        horizontal=True,
        reference_position=width / 2,
    )
    vertical_segments = _normalize_segments(
        vertical_raw,
        horizontal=False,
        reference_position=height / 2,
    )
    horizontal = _cluster_boundaries(
        horizontal_segments,
        reference_position=width / 2,
        dimension=width,
        tolerance=max(4.0, height * 0.006),
    )
    vertical = _cluster_boundaries(
        vertical_segments,
        reference_position=height / 2,
        dimension=height,
        tolerance=max(4.0, width * 0.004),
    )
    confidence = _grid_confidence(horizontal, vertical)
    if confidence < minimum_confidence:
        return None
    grid = TableGrid(horizontal, vertical, confidence)
    if debug_path is not None:
        _save_debug_overlay(image, grid, debug_path)
    return grid
