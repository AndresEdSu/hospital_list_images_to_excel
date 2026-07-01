from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from hospital_ocr.handwriting import GridCell, TextRow
from hospital_ocr.models import OcrLine


CELL_VERTICAL_PADDING_RATIO = 0.03


@dataclass(frozen=True)
class _CellPlacement:
    cell: GridCell
    start_x: int
    end_x: int
    offset_y: int
    inner_left: int
    inner_top: int
    inner_width: int
    inner_height: int


def _has_cell_content(image: np.ndarray) -> bool:
    if image.size == 0:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = max(3, min(21, (min(gray.shape) // 2) * 2 - 1))
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    minimum_height = max(2, round(gray.shape[0] * 0.12))
    meaningful_area = sum(
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_HEIGHT] >= minimum_height
        and stats[index, cv2.CC_STAT_WIDTH] >= 2
        and stats[index, cv2.CC_STAT_AREA] >= 5
    )
    return meaningful_area >= max(8, round(gray.size * 0.001))


def _cell_point(
    cell: GridCell,
    horizontal_ratio: float,
) -> tuple[float, float, float]:
    ratio = min(1.0, max(0.0, horizontal_ratio))
    top_left, top_right, bottom_right, bottom_left = cell.corners
    top_x = top_left[0] + ratio * (top_right[0] - top_left[0])
    top_y = top_left[1] + ratio * (top_right[1] - top_left[1])
    bottom_x = bottom_left[0] + ratio * (bottom_right[0] - bottom_left[0])
    bottom_y = bottom_left[1] + ratio * (bottom_right[1] - bottom_left[1])
    return (
        (top_x + bottom_x) / 2,
        (top_y + bottom_y) / 2,
        max(4.0, abs(bottom_y - top_y)),
    )


def _expanded_cell_corners(
    cell: GridCell,
    padding_ratio: float = CELL_VERTICAL_PADDING_RATIO,
) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = cell.corners
    return np.float32(
        [
            [
                top_left[0] - padding_ratio * (bottom_left[0] - top_left[0]),
                top_left[1] - padding_ratio * (bottom_left[1] - top_left[1]),
            ],
            [
                top_right[0]
                - padding_ratio * (bottom_right[0] - top_right[0]),
                top_right[1]
                - padding_ratio * (bottom_right[1] - top_right[1]),
            ],
            [
                bottom_right[0]
                + padding_ratio * (bottom_right[0] - top_right[0]),
                bottom_right[1]
                + padding_ratio * (bottom_right[1] - top_right[1]),
            ],
            [
                bottom_left[0]
                + padding_ratio * (bottom_left[0] - top_left[0]),
                bottom_left[1]
                + padding_ratio * (bottom_left[1] - top_left[1]),
            ],
        ]
    )


class PaddleOcrEngine:
    def __init__(self, cache_dir: Path) -> None:
        resolved_cache = cache_dir.resolve()
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(resolved_cache))
        os.environ.setdefault("HF_HOME", str(resolved_cache / "huggingface"))

        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            lang="es",
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
            text_rec_score_thresh=0.0,
        )

    def recognize(self, image_path: Path) -> list[OcrLine]:
        results = list(self._engine.predict(str(image_path)))
        if not results:
            return []
        result = results[0]
        texts = list(result.get("rec_texts", []))
        scores = list(result.get("rec_scores", []))
        boxes = list(result.get("rec_boxes", []))
        with Image.open(image_path) as image:
            width, height = image.size

        lines: list[OcrLine] = []
        for text, score, box in zip(texts, scores, boxes, strict=False):
            values = [int(round(float(value))) for value in box]
            if len(values) != 4:
                continue
            lines.append(
                OcrLine(
                    text=str(text).strip(),
                    score=float(score),
                    box=(values[0], values[1], values[2], values[3]),
                    image_width=width,
                    image_height=height,
                )
            )
        return lines

    def recognize_rows(
        self,
        image_path: Path,
        rows: list[TextRow],
        artifacts_dir: Path,
    ) -> list[OcrLine]:
        """Run a second OCR pass on enlarged, enhanced text-row crops."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mapped_lines: list[OcrLine] = []
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            image_width, image_height = image.size

            for index, row in enumerate(rows, start=1):
                left, top, right, bottom = row.box
                vertical_padding = max(2, round((bottom - top) * 0.03))
                top = max(0, top - vertical_padding)
                bottom = min(image_height, bottom + vertical_padding)
                crop = image.crop((left, top, right, bottom))
                crop.save(artifacts_dir / f"row_{index:03d}_source.jpg", quality=92)

                scale = 3
                overlap = round(crop.width * 0.12)
                seam = crop.width // 2
                windows = (
                    ("left", 0, min(crop.width, seam + overlap)),
                    ("right", max(0, seam - overlap), crop.width),
                )
                for window_name, window_left, window_right in windows:
                    window = crop.crop(
                        (window_left, 0, window_right, crop.height)
                    )
                    window = ImageEnhance.Contrast(window).enhance(1.12)
                    window = window.resize(
                        (window.width * scale, window.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                    window_path = (
                        artifacts_dir
                        / f"row_{index:03d}_{window_name}_enhanced.png"
                    )
                    window.save(window_path)

                    for line in self.recognize(window_path):
                        mapped = OcrLine(
                            text=line.text,
                            score=line.score,
                            box=(
                                round(line.box[0] / scale)
                                + left
                                + window_left,
                                round(line.box[1] / scale) + top,
                                round(line.box[2] / scale)
                                + left
                                + window_left,
                                round(line.box[3] / scale) + top,
                            ),
                            image_width=image_width,
                            image_height=image_height,
                        )
                        local_center = mapped.center_x - left
                        owns_center = (
                            window_name == "left" and local_center < seam
                        ) or (
                            window_name == "right" and local_center >= seam
                        )
                        if owns_center:
                            mapped_lines.append(mapped)
        return mapped_lines

    def recognize_grid_cells(
        self,
        image_path: Path,
        cells: list[GridCell],
        artifacts_dir: Path,
    ) -> list[OcrLine]:
        """Rectify grid cells, OCR one compact contact sheet per source row."""
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        image_height, image_width = image.shape[:2]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        rows: dict[int, list[GridCell]] = {}
        for cell in cells:
            rows.setdefault(cell.row_index, []).append(cell)

        mapped_lines: list[OcrLine] = []
        for row_index, row_cells in sorted(rows.items()):
            prepared: list[
                tuple[GridCell, np.ndarray, int, int]
            ] = []
            for cell in sorted(
                row_cells,
                key=lambda item: item.column_index,
            ):
                width = cell.target_width
                height = max(
                    12,
                    round(
                        cell.target_height
                        * (1 + 2 * CELL_VERTICAL_PADDING_RATIO)
                    ),
                )
                source = _expanded_cell_corners(cell)
                destination = np.float32(
                    [
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1],
                    ]
                )
                transform = cv2.getPerspectiveTransform(source, destination)
                warped = cv2.warpPerspective(
                    image,
                    transform,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
                margin_x = max(1, min(5, round(width * 0.025)))
                margin_y = 1
                inner = warped[
                    margin_y : max(margin_y + 1, height - margin_y),
                    margin_x : max(margin_x + 1, width - margin_x),
                ]
                if (
                    row_index != 0
                    and not _has_cell_content(inner)
                ):
                    continue
                prepared.append((cell, inner, margin_x, margin_y))
            if not prepared:
                continue

            contact_height = max(item[1].shape[0] for item in prepared)
            gap = max(40, round(contact_height * 2.0))
            contact_width = (
                sum(item[1].shape[1] for item in prepared)
                + gap * (len(prepared) - 1)
            )
            contact = np.full(
                (contact_height, contact_width, 3),
                255,
                dtype=np.uint8,
            )
            placements: list[_CellPlacement] = []
            cursor = 0
            for cell, inner, margin_x, margin_y in prepared:
                offset_y = (contact_height - inner.shape[0]) // 2
                end_x = cursor + inner.shape[1]
                contact[
                    offset_y : offset_y + inner.shape[0],
                    cursor:end_x,
                ] = inner
                placements.append(
                    _CellPlacement(
                        cell=cell,
                        start_x=cursor,
                        end_x=end_x,
                        offset_y=offset_y,
                        inner_left=margin_x,
                        inner_top=margin_y,
                        inner_width=inner.shape[1],
                        inner_height=inner.shape[0],
                    )
                )
                cursor = end_x + gap

            source_path = artifacts_dir / f"row_{row_index + 1:03d}_cells.jpg"
            cv2.imwrite(str(source_path), contact)
            scale = 3
            split_candidates = [
                (left.end_x + right.start_x) // 2
                for left, right in zip(
                    placements,
                    placements[1:],
                    strict=False,
                )
            ]
            split = (
                min(
                    split_candidates,
                    key=lambda value: abs(value - contact_width / 2),
                )
                if split_candidates
                else contact_width
            )
            windows = (
                ("left", 0, split),
                ("right", split, contact_width),
            )
            for window_name, window_left, window_right in windows:
                if window_right - window_left < 2:
                    continue
                window = contact[:, window_left:window_right]
                enhanced = Image.fromarray(
                    cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                )
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.12)
                enhanced = enhanced.resize(
                    (enhanced.width * scale, enhanced.height * scale),
                    Image.Resampling.LANCZOS,
                )
                enhanced_path = (
                    artifacts_dir
                    / (
                        f"row_{row_index + 1:03d}_cells_"
                        f"{window_name}_enhanced.png"
                    )
                )
                enhanced.save(enhanced_path)

                for line in self.recognize(enhanced_path):
                    center_x = line.center_x / scale + window_left
                    placement = min(
                        placements,
                        key=lambda item: (
                            0
                            if item.start_x <= center_x <= item.end_x
                            else min(
                                abs(center_x - item.start_x),
                                abs(center_x - item.end_x),
                            )
                        ),
                    )
                    warped_center_y = (
                        line.center_y / scale
                        - placement.offset_y
                        + placement.inner_top
                    )
                    expanded_height = (
                        placement.inner_height + 2 * placement.inner_top
                    )
                    original_margin = (
                        expanded_height
                        * CELL_VERTICAL_PADDING_RATIO
                        / (1 + 2 * CELL_VERTICAL_PADDING_RATIO)
                    )
                    if not (
                        original_margin
                        <= warped_center_y
                        <= expanded_height - original_margin
                    ):
                        continue
                    local_left = (
                        line.box[0] / scale
                        + window_left
                        - placement.start_x
                        + placement.inner_left
                    )
                    local_right = (
                        line.box[2] / scale
                        + window_left
                        - placement.start_x
                        + placement.inner_left
                    )
                    left_ratio = local_left / placement.cell.target_width
                    right_ratio = local_right / placement.cell.target_width
                    left_point = _cell_point(placement.cell, left_ratio)
                    right_point = _cell_point(placement.cell, right_ratio)
                    center_point = _cell_point(
                        placement.cell,
                        (left_ratio + right_ratio) / 2,
                    )
                    logical_height = max(
                        4,
                        min(20, round(center_point[2] * 0.45)),
                    )
                    logical_top = round(
                        center_point[1] - logical_height / 2
                    )
                    left_x = round(min(left_point[0], right_point[0]))
                    right_x = round(max(left_point[0], right_point[0]))
                    if right_x - left_x < 2:
                        right_x = left_x + 2
                    mapped_lines.append(
                        OcrLine(
                            text=line.text,
                            score=line.score,
                            box=(
                                max(0, left_x),
                                max(0, logical_top),
                                min(image_width, right_x),
                                min(
                                    image_height,
                                    logical_top + logical_height,
                                ),
                            ),
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )
        return mapped_lines


def save_raw_ocr(path: Path, image_path: Path, lines: list[OcrLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "imagen": image_path.name,
        "lineas": [
            {
                "texto": line.text,
                "confianza": round(line.score, 6),
                "caja": list(line.box),
            }
            for line in lines
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
