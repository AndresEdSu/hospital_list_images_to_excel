from pathlib import Path

from PIL import Image

from hospital_ocr.handwriting import GridCell, TextRow, rows_from_grid
from hospital_ocr.models import GridBoundary, OcrLine, TableGrid
from hospital_ocr.pipeline import _image_progress, _recognize_image


class FakeOcrEngine:
    def __init__(self) -> None:
        self.global_calls = 0
        self.row_calls = 0

    def recognize(self, image_path: Path) -> list[OcrLine]:
        self.global_calls += 1
        return [OcrLine("OCR global", 0.9, (20, 20, 180, 40), 400, 200)]

    def recognize_rows(
        self,
        image_path: Path,
        rows: list[TextRow],
        artifacts_dir: Path,
    ) -> list[OcrLine]:
        self.row_calls += 1
        lines: list[OcrLine] = []
        for index, row in enumerate(rows):
            y = round(row.center_y)
            lines.extend(
                [
                    OcrLine(
                        f"Nombre {index}",
                        0.9,
                        (30, y - 5, 150, y + 5),
                        400,
                        200,
                    ),
                    OcrLine(
                        "12345678",
                        0.9,
                        (220, y - 5, 330, y + 5),
                        400,
                        200,
                    ),
                ]
            )
        return lines

    def recognize_grid_cells(
        self,
        image_path: Path,
        cells: list[GridCell],
        artifacts_dir: Path,
    ) -> list[OcrLine]:
        self.row_calls += 1
        return [
            OcrLine("Nombre", 0.9, (30, 65, 150, 85), 400, 200),
            OcrLine("12345678", 0.9, (220, 65, 330, 85), 400, 200),
            OcrLine("Paciente", 0.9, (30, 115, 150, 135), 400, 200),
            OcrLine("87654321", 0.9, (220, 115, 330, 135), 400, 200),
        ]


def _grid() -> TableGrid:
    return TableGrid(
        horizontal=(
            GridBoundary(0, 50, 1),
            GridBoundary(0, 100, 1),
            GridBoundary(0, 150, 1),
        ),
        vertical=(
            GridBoundary(0, 20, 1),
            GridBoundary(0, 200, 1),
            GridBoundary(0, 380, 1),
        ),
        confidence=0.9,
    )


def test_grid_boundaries_become_handwriting_row_crops(tmp_path: Path) -> None:
    image_path = tmp_path / "grid.png"
    Image.new("RGB", (400, 200), "white").save(image_path)

    rows = rows_from_grid(image_path, _grid())

    assert [row.box for row in rows] == [
        (20, 50, 380, 100),
        (20, 100, 380, 150),
    ]


def test_handwritten_grid_mode_compares_global_and_refined_cells(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "grid.png"
    Image.new("RGB", (400, 200), "white").save(image_path)
    engine = FakeOcrEngine()

    lines, audit = _recognize_image(
        engine,
        image_path,
        _grid(),
        "handwritten",
        tmp_path / "rows",
    )

    assert engine.global_calls == 1
    assert engine.row_calls == 1
    assert len(lines) == 5
    assert lines[0].text == "OCR global"
    assert audit is not None
    assert audit["origen_limites"] == "cuadrícula"


def test_printed_mode_uses_only_global_ocr(tmp_path: Path) -> None:
    image_path = tmp_path / "printed.png"
    Image.new("RGB", (400, 200), "white").save(image_path)
    engine = FakeOcrEngine()

    lines, audit = _recognize_image(
        engine,
        image_path,
        _grid(),
        "printed",
        tmp_path / "rows",
    )

    assert [line.text for line in lines] == ["OCR global"]
    assert engine.global_calls == 1
    assert engine.row_calls == 0
    assert audit is None


def test_image_progress_advances_through_each_processing_stage() -> None:
    values = [
        _image_progress(0, 2, fraction)
        for fraction in (0.0, 0.14, 0.26, 0.78, 0.96, 1.0)
    ]
    values.extend(
        _image_progress(1, 2, fraction)
        for fraction in (0.0, 0.14, 0.26, 0.78, 0.96, 1.0)
    )

    assert values == sorted(values)
    assert len(set(values)) == 11
    assert values[0] == 0.04
    assert values[-1] == 0.92
