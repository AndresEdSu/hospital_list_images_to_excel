from pathlib import Path

import cv2
import numpy as np

from hospital_ocr.grid_detector import detect_table_grid


def test_detects_slanted_grid_and_writes_debug_overlay(tmp_path: Path) -> None:
    image = np.full((520, 820, 3), 255, dtype=np.uint8)
    for row in range(7):
        y_left = 70 + row * 60
        y_right = y_left + 28
        cv2.line(image, (50, y_left), (770, y_right), (0, 0, 0), 2)
    for column in range(6):
        x_top = 50 + column * 144
        x_bottom = x_top + 10
        cv2.line(image, (x_top, 70), (x_bottom, 458), (0, 0, 0), 2)

    image_path = tmp_path / "grid.jpg"
    debug_path = tmp_path / "grid_debug.jpg"
    cv2.imwrite(str(image_path), image)

    grid = detect_table_grid(image_path, debug_path)

    assert grid is not None
    assert len(grid.horizontal) >= 6
    assert len(grid.vertical) >= 5
    assert grid.confidence >= 0.55
    assert debug_path.exists()


def test_plain_text_image_is_not_treated_as_grid(tmp_path: Path) -> None:
    image = np.full((360, 640, 3), 255, dtype=np.uint8)
    for index, text in enumerate(("María Pérez", "Luis Gómez", "Ana Rivera")):
        cv2.putText(
            image,
            text,
            (60, 90 + index * 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    image_path = tmp_path / "plain_text.jpg"
    cv2.imwrite(str(image_path), image)

    assert detect_table_grid(image_path) is None
