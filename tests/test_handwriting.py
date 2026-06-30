from pathlib import Path

from PIL import Image, ImageDraw

from hospital_ocr.handwriting import (
    TextRow,
    detect_text_rows,
    merge_row_ocr,
    needs_row_ocr,
    row_ocr_coverage,
)
from hospital_ocr.models import OcrLine


def _line(text: str, y: int, *, height: int = 12) -> OcrLine:
    return OcrLine(text, 0.9, (20, y, 200, y + height), 800, 600)


def test_detect_text_rows_from_repeated_text_components(tmp_path: Path) -> None:
    image = Image.new("RGB", (800, 600), "white")
    drawing = ImageDraw.Draw(image)
    expected_centers = [100, 200, 300, 400, 500]
    for center_y in expected_centers:
        for x in range(60, 680, 55):
            drawing.rectangle((x, center_y - 7, x + 24, center_y + 7), fill="black")
    path = tmp_path / "rows.png"
    image.save(path)

    rows = detect_text_rows(path)

    assert len(rows) == len(expected_centers)
    assert all(
        abs(row.center_y - expected) < 15
        for row, expected in zip(rows, expected_centers, strict=True)
    )


def test_low_row_coverage_activates_segmented_ocr() -> None:
    rows = [
        TextRow((0, index * 100, 800, (index + 1) * 100), 10)
        for index in range(5)
    ]
    lines = [_line("uno", 20), _line("dos", 220)]

    assert row_ocr_coverage(lines, rows) == 0.4
    assert needs_row_ocr(lines, rows)


def test_merge_row_ocr_replaces_only_rows_recovered_by_fallback() -> None:
    rows = [
        TextRow((0, 0, 800, 100), 10),
        TextRow((0, 100, 800, 200), 10),
    ]
    initial = [_line("fila inicial uno", 30), _line("fila inicial dos", 130)]
    segmented = [_line("fila manuscrita dos", 135)]

    merged = merge_row_ocr(initial, segmented, rows)

    assert [line.text for line in merged] == [
        "fila inicial uno",
        "fila manuscrita dos",
    ]
