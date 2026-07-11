from __future__ import annotations

import re
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


REFINEMENT_QUALITY_MARGIN = 0.01
REFINEMENT_COVERAGE_TOLERANCE = 0.001
DOCUMENT_RE = re.compile(
    r"(?<!\d)(?:[VEve]\s*[-.]?\s*)?\d(?:[.,\-]?\d){5,10}(?!\d)"
)
WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")


def _ocr_quality_score(lines: list[OcrLine]) -> float:
    useful = [line for line in lines if line.text.strip()]
    if not useful:
        return 0.0
    total_characters = sum(max(1, len(line.text.strip())) for line in useful)
    confidence = sum(
        line.score * max(1, len(line.text.strip())) for line in useful
    ) / total_characters
    text = " ".join(line.text.strip() for line in useful)
    words = WORD_RE.findall(text)
    documents = DOCUMENT_RE.findall(text)
    semantic_bonus = min(0.14, len(words) * 0.004 + len(documents) * 0.04)
    volume_bonus = min(0.08, len(text) / 700)
    strange_characters = sum(
        not (character.isalnum() or character.isspace() or character in ".-#,/")
        for character in text
    )
    strange_penalty = min(0.20, strange_characters / max(1, len(text)))
    return confidence + semantic_bonus + volume_bonus - strange_penalty


def _choose_refined_ocr(
    initial_lines: list[OcrLine],
    refined_lines: list[OcrLine],
    *,
    coverage_before: float | None,
    coverage_after: float | None,
) -> tuple[list[OcrLine], dict[str, Any], str]:
    quality_before = _ocr_quality_score(initial_lines)
    quality_after = _ocr_quality_score(refined_lines)
    coverage_preserved = (
        coverage_before is None
        or coverage_after is None
        or coverage_after + REFINEMENT_COVERAGE_TOLERANCE >= coverage_before
    )
    quality_improved = quality_after >= quality_before + REFINEMENT_QUALITY_MARGIN
    accepted = coverage_preserved and quality_improved
    if accepted:
        reason = "mejora_calidad_y_cobertura"
    elif not coverage_preserved:
        reason = "cobertura_menor"
    else:
        reason = "calidad_no_mejora"
    decision = {
        "aceptado": accepted,
        "motivo": reason,
        "calidad_antes": round(quality_before, 4),
        "calidad_despues": round(quality_after, 4),
        "margen_calidad": round(quality_after - quality_before, 4),
        "cobertura_preservada": coverage_preserved,
    }
    if coverage_before is not None:
        decision["cobertura_antes"] = round(coverage_before, 4)
    if coverage_after is not None:
        decision["cobertura_despues"] = round(coverage_after, 4)
    if accepted:
        return refined_lines, decision, ""
    return (
        initial_lines,
        decision,
        "El OCR reforzado no mejoro cobertura/calidad; se uso OCR global.",
    )


def row_audit(
    *,
    mode: OcrMode,
    rows: list[TextRow],
    lines: list[OcrLine],
    coverage_before: float | None,
    coverage_after: float | None = None,
    boundary_source: str,
    fallback_reason: str = "",
    grid: TableGrid | None = None,
    refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if coverage_after is not None:
        measured_coverage_after = coverage_after
    elif grid is not None:
        covered_grid_rows = {
            row
            for line in lines
            if (row := grid.row_for_box(line.box)) is not None
        }
        measured_coverage_after = len(covered_grid_rows) / max(
            1,
            len(grid.horizontal) - 1,
        )
    else:
        measured_coverage_after = row_ocr_coverage(lines, rows) if rows else None
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
            round(measured_coverage_after, 4)
            if measured_coverage_after is not None
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
    candidate_lines = merge_grid_ocr(global_lines, refined_lines, grid)
    coverage_before = grid_row_coverage(global_lines, grid)
    coverage_after = grid_row_coverage(candidate_lines, grid)
    lines, decision, fallback_reason = _choose_refined_ocr(
        global_lines,
        candidate_lines,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
    )
    return lines, row_audit(
        mode=mode,
        rows=rows,
        lines=lines,
        coverage_before=coverage_before,
        coverage_after=(
            coverage_after if decision["aceptado"] else coverage_before
        ),
        boundary_source="cuadrícula",
        fallback_reason=fallback_reason,
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
            "decision": decision,
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
        candidate_lines = merge_row_ocr(
            initial_lines,
            segmented_lines,
            selected_rows,
        )
        coverage_before = row_ocr_coverage(initial_lines, rows)
        coverage_after = row_ocr_coverage(candidate_lines, rows)
        lines, decision, fallback_reason = _choose_refined_ocr(
            initial_lines,
            candidate_lines,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
        )
        return lines, row_audit(
            mode=mode,
            rows=selected_rows,
            lines=lines,
            coverage_before=coverage_before,
            coverage_after=(
                coverage_after if decision["aceptado"] else coverage_before
            ),
            boundary_source=boundary_source,
            fallback_reason=fallback_reason,
            grid=grid,
            refinement={
                "renglones_totales": len(rows),
                "renglones_seleccionados": len(selected_rows),
                "proporcion": round(
                    len(selected_rows) / max(1, len(rows)),
                    4,
                ),
                "decision": decision,
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
    candidate_lines = merge_row_ocr(initial_lines, segmented_lines, selected_rows)
    coverage_after = row_ocr_coverage(candidate_lines, rows)
    lines, decision, fallback_reason = _choose_refined_ocr(
        initial_lines,
        candidate_lines,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
    )
    return lines, row_audit(
        mode=mode,
        rows=selected_rows,
        lines=lines,
        coverage_before=coverage_before,
        coverage_after=(
            coverage_after if decision["aceptado"] else coverage_before
        ),
        boundary_source="renglones",
        fallback_reason=fallback_reason,
        refinement={
            "renglones_totales": len(rows),
            "renglones_seleccionados": len(selected_rows),
            "proporcion": round(
                len(selected_rows) / max(1, len(rows)),
                4,
            ),
            "decision": decision,
        },
    )
