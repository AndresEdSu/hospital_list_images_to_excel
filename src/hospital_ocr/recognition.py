from __future__ import annotations

from pathlib import Path
from typing import Any

from hospital_ocr.handwriting import (
    TextRow,
    cells_from_grid,
    detect_text_rows,
    merge_grid_ocr,
    merge_row_ocr,
    needs_row_ocr,
    row_segmentation_is_too_coarse,
    row_ocr_coverage,
    rows_from_grid,
)
from hospital_ocr.models import OcrLine, TableGrid
from hospital_ocr.ocr_engine import PaddleOcrEngine
from hospital_ocr.ocr_refinement import (
    select_grid_cells_for_refinement,
    select_rows_for_refinement,
)
from hospital_ocr.pipeline_types import OcrMode


def row_audit(
    *,
    mode: OcrMode,
    rows: list[TextRow],
    lines: list[OcrLine],
    coverage_before: float | None,
    boundary_source: str,
    fallback_reason: str = "",
    grid: TableGrid | None = None,
    refinement: dict[str, int | float | bool] | None = None,
) -> dict[str, Any]:
    if grid is not None:
        covered_grid_rows = {
            row
            for line in lines
            if (row := grid.row_for_box(line.box)) is not None
        }
        coverage_after = len(covered_grid_rows) / max(
            1,
            len(grid.horizontal) - 1,
        )
    else:
        coverage_after = row_ocr_coverage(lines, rows) if rows else None
    return {
        "modo": mode,
        "procesamiento_reforzado": bool(rows),
        "origen_limites": boundary_source,
        "motivo_respaldo": fallback_reason,
        "cobertura_antes": (
            round(coverage_before, 4)
            if coverage_before is not None
            else None
        ),
        "cobertura_despues": (
            round(coverage_after, 4)
            if coverage_after is not None
            else None
        ),
        "refuerzo": refinement or {},
        "renglones": [
            {
                "caja": list(row.box),
                "linea_base": round(row.center_y, 2),
                "fuerza": round(row.strength, 4),
            }
            for row in rows
        ],
    }


def grid_row_coverage(
    lines: list[OcrLine],
    grid: TableGrid,
) -> float:
    covered_rows = {
        row
        for line in lines
        if (row := grid.row_for_box(line.box)) is not None
    }
    return len(covered_rows) / max(1, len(grid.horizontal) - 1)


def recognize_grid_image(
    engine: PaddleOcrEngine,
    image_path: Path,
    grid: TableGrid,
    mode: OcrMode,
    rows_dir: Path,
    initial_lines: list[OcrLine] | None = None,
) -> tuple[list[OcrLine], dict[str, Any]]:
    rows = rows_from_grid(image_path, grid)
    global_lines = (
        initial_lines
        if initial_lines is not None
        else engine.recognize(image_path)
    )
    if not rows:
        return global_lines, row_audit(
            mode=mode,
            rows=[],
            lines=global_lines,
            coverage_before=None,
            boundary_source="cuadrícula",
            fallback_reason=(
                "No se pudieron delimitar recortes; se usó OCR global."
            ),
            grid=grid,
        )

    all_cells = cells_from_grid(grid)
    selected_cells = select_grid_cells_for_refinement(
        all_cells,
        global_lines,
        grid,
        mode,
    )
    selectively_requested = len(selected_cells)
    if (
        selected_cells
        and len(selected_cells) / max(1, len(all_cells)) >= 0.90
    ):
        selected_cells = all_cells
    refined_lines = (
        engine.recognize_grid_cells(
            image_path,
            selected_cells,
            rows_dir,
        )
        if selected_cells
        else []
    )
    lines = merge_grid_ocr(global_lines, refined_lines, grid)
    return lines, row_audit(
        mode=mode,
        rows=rows,
        lines=lines,
        coverage_before=grid_row_coverage(global_lines, grid),
        boundary_source="cuadrícula",
        grid=grid,
        refinement={
            "celdas_totales": len(all_cells),
            "celdas_solicitadas_selectivamente": selectively_requested,
            "celdas_seleccionadas": len(selected_cells),
            "proporcion": round(
                len(selected_cells) / max(1, len(all_cells)),
                4,
            ),
            "tabla_completa_por_cobertura_alta": (
                len(selected_cells) == len(all_cells)
                and selectively_requested < len(all_cells)
            ),
        },
    )


def recognize_image(
    engine: PaddleOcrEngine,
    image_path: Path,
    grid: TableGrid | None,
    mode: OcrMode,
    rows_dir: Path,
) -> tuple[list[OcrLine], dict[str, Any] | None]:
    if mode == "printed":
        return engine.recognize(image_path), None

    if mode == "handwritten":
        if grid is not None:
            return recognize_grid_image(
                engine,
                image_path,
                grid,
                mode,
                rows_dir,
            )

        initial_lines = engine.recognize(image_path)
        rows = detect_text_rows(image_path)
        boundary_source = "renglones"
        if not rows:
            return initial_lines, row_audit(
                mode=mode,
                rows=[],
                lines=initial_lines,
                coverage_before=None,
                boundary_source=boundary_source,
                fallback_reason=(
                    "No se pudieron delimitar recortes; se usó OCR global."
                ),
                grid=grid,
            )
        if row_segmentation_is_too_coarse(initial_lines, rows):
            return initial_lines, row_audit(
                mode=mode,
                rows=rows,
                lines=initial_lines,
                coverage_before=row_ocr_coverage(initial_lines, rows),
                boundary_source=boundary_source,
                fallback_reason=(
                    "Los renglones detectados agrupan varias filas; "
                    "se usó OCR global."
                ),
                grid=grid,
                refinement={
                    "renglones_totales": len(rows),
                    "renglones_seleccionados": 0,
                    "proporcion": 0.0,
                    "renglones_demasiado_amplios": True,
                },
            )
        selected_rows = select_rows_for_refinement(
            rows,
            initial_lines,
            threshold=0.96,
        )
        segmented_lines = (
            engine.recognize_rows(
                image_path,
                selected_rows,
                rows_dir,
            )
            if selected_rows
            else []
        )
        lines = merge_row_ocr(
            initial_lines,
            segmented_lines,
            selected_rows,
        )
        return lines, row_audit(
            mode=mode,
            rows=selected_rows,
            lines=lines,
            coverage_before=row_ocr_coverage(initial_lines, rows),
            boundary_source=boundary_source,
            grid=grid,
            refinement={
                "renglones_totales": len(rows),
                "renglones_seleccionados": len(selected_rows),
                "proporcion": round(
                    len(selected_rows) / max(1, len(rows)),
                    4,
                ),
            },
        )

    initial_lines = engine.recognize(image_path)
    if grid is not None:
        return recognize_grid_image(
            engine,
            image_path,
            grid,
            mode,
            rows_dir,
            initial_lines,
        )
    rows = detect_text_rows(image_path)
    if not needs_row_ocr(initial_lines, rows):
        return initial_lines, None
    coverage_before = row_ocr_coverage(initial_lines, rows)
    selected_rows = select_rows_for_refinement(
        rows,
        initial_lines,
        threshold=0.88,
    )
    segmented_lines = (
        engine.recognize_rows(
            image_path,
            selected_rows,
            rows_dir,
        )
        if selected_rows
        else []
    )
    lines = merge_row_ocr(initial_lines, segmented_lines, selected_rows)
    return lines, row_audit(
        mode=mode,
        rows=selected_rows,
        lines=lines,
        coverage_before=coverage_before,
        boundary_source="renglones",
        refinement={
            "renglones_totales": len(rows),
            "renglones_seleccionados": len(selected_rows),
            "proporcion": round(
                len(selected_rows) / max(1, len(rows)),
                4,
            ),
        },
    )
