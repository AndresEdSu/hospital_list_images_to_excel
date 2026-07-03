from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from hospital_ocr.catalogs import load_centers, load_places, load_specialties
from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.discovery import (
    discover_images,
    find_unmapped_images,
    select_evenly,
)
from hospital_ocr.exporting import export_results
from hospital_ocr.grid_detector import detect_table_grid
from hospital_ocr.handwriting import (
    GridCell,
    TextRow,
    cells_from_grid,
    detect_text_rows,
    merge_grid_ocr,
    merge_row_ocr,
    needs_row_ocr,
    row_ocr_coverage,
    rows_from_grid,
)
from hospital_ocr.models import (
    ConsolidationResult,
    OcrLine,
    PatientRecord,
    TableGrid,
)
from hospital_ocr.name_splitter import load_name_lexicons
from hospital_ocr.ocr_cache import load_cached_ocr, save_cached_ocr
from hospital_ocr.ocr_engine import PaddleOcrEngine, save_raw_ocr
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.pipeline_types import OcrMode
from hospital_ocr.preprocessing import preprocess_image


OCR_MODES: tuple[OcrMode, ...] = ("auto", "handwritten", "printed")


@dataclass(frozen=True)
class PipelineConfig:
    images_dir: Path
    centers_path: Path
    specialties_path: Path
    given_names_path: Path
    surnames_path: Path
    interim_dir: Path
    output_path: Path
    cache_dir: Path
    limit: int | None = None
    preprocess: bool = True
    overwrite: bool = False
    places_path: Path | None = None
    ocr_mode: OcrMode = "auto"

    def __post_init__(self) -> None:
        if self.ocr_mode not in OCR_MODES:
            raise ValueError(
                f"Modo OCR no válido: {self.ocr_mode}. "
                f"Use {', '.join(OCR_MODES)}."
            )


@dataclass(frozen=True)
class PipelineReport:
    discovered_images: int
    processed_images: int
    extracted_records: int
    consolidated_records: int
    review_records: int
    errors: tuple[dict[str, str], ...]
    output_path: Path


@dataclass(frozen=True)
class ProcessingResult:
    consolidation: ConsolidationResult
    discovered_images: int
    processed_images: int
    extracted_records: int
    review_records: int
    errors: tuple[dict[str, str], ...]
    specialty_values: tuple[str, ...]


ProgressCallback = Callable[[float, float, str], None]

_IMAGE_PROGRESS_START = 0.04
_IMAGE_PROGRESS_END = 0.92
_GRID_REFINEMENT_THRESHOLDS: dict[OcrMode, float] = {
    "auto": 0.88,
    "handwritten": 0.96,
    "printed": 0.0,
}


def _image_progress(
    image_index: int,
    total_images: int,
    fraction: float,
) -> float:
    if total_images <= 0:
        return _IMAGE_PROGRESS_START
    bounded_fraction = min(1.0, max(0.0, fraction))
    completed_images = image_index + bounded_fraction
    image_share = completed_images / total_images
    return _IMAGE_PROGRESS_START + image_share * (
        _IMAGE_PROGRESS_END - _IMAGE_PROGRESS_START
    )


def _report_progress(
    callback: ProgressCallback | None,
    value: float,
    message: str,
) -> None:
    if callback:
        callback(min(1.0, max(0.0, value)), 1.0, message)


def _row_audit(
    *,
    mode: OcrMode,
    rows: list[TextRow],
    lines: list[OcrLine],
    coverage_before: float | None,
    boundary_source: str,
    fallback_reason: str = "",
    grid: TableGrid | None = None,
    refinement: dict[str, int | float] | None = None,
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


def _grid_row_coverage(
    lines: list[OcrLine],
    grid: TableGrid,
) -> float:
    covered_rows = {
        row
        for line in lines
        if (row := grid.row_for_box(line.box)) is not None
    }
    return len(covered_rows) / max(1, len(grid.horizontal) - 1)


def _select_grid_cells_for_refinement(
    cells: list[GridCell],
    lines: list[OcrLine],
    grid: TableGrid,
    mode: OcrMode,
) -> list[GridCell]:
    threshold = _GRID_REFINEMENT_THRESHOLDS[mode]
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
        if _is_sex_marker_candidate(candidates):
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
            or _has_structured_field_candidate(candidates)
            or max(line.score for line in candidates) < threshold
        ):
            selected.append(cell)
    return selected


def _has_structured_field_candidate(lines: list[OcrLine]) -> bool:
    text = " ".join(line.text for line in lines)
    if re.search(r"\d", text):
        return True
    marker = re.sub(r"[^A-Za-z]", "", text).upper()
    return marker in {"M", "F", "H", "T", "E", "N"}


def _is_sex_marker_candidate(lines: list[OcrLine]) -> bool:
    text = " ".join(line.text for line in lines)
    marker = re.sub(r"[^A-Za-z]", "", text).upper()
    return marker in {"M", "F", "H", "T", "E", "N"}


def _select_rows_for_refinement(
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


def _recognize_grid_image(
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
        return global_lines, _row_audit(
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
    selected_cells = _select_grid_cells_for_refinement(
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
    return lines, _row_audit(
        mode=mode,
        rows=rows,
        lines=lines,
        coverage_before=_grid_row_coverage(global_lines, grid),
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


def _recognize_image(
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
            return _recognize_grid_image(
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
            return initial_lines, _row_audit(
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
        selected_rows = _select_rows_for_refinement(
            rows,
            initial_lines,
            threshold=0.96,
        )
        segmented_lines = engine.recognize_rows(
            image_path,
            selected_rows,
            rows_dir,
        ) if selected_rows else []
        lines = merge_row_ocr(
            initial_lines,
            segmented_lines,
            selected_rows,
        )
        return lines, _row_audit(
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
        return _recognize_grid_image(
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
    selected_rows = _select_rows_for_refinement(
        rows,
        initial_lines,
        threshold=0.88,
    )
    segmented_lines = engine.recognize_rows(
        image_path,
        selected_rows,
        rows_dir,
    ) if selected_rows else []
    lines = merge_row_ocr(initial_lines, segmented_lines, selected_rows)
    return lines, _row_audit(
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


def process_images(
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> ProcessingResult:
    centers = load_centers(config.centers_path)
    specialties = load_specialties(config.specialties_path)
    places = load_places(config.places_path)
    name_lexicons = load_name_lexicons(
        config.given_names_path,
        config.surnames_path,
    )
    unmapped = find_unmapped_images(config.images_dir, centers)
    discovered = discover_images(config.images_dir, centers)
    selected = select_evenly(discovered, config.limit)
    if not selected:
        if unmapped:
            folders = sorted(
                {
                    path.relative_to(config.images_dir).parts[0]
                    for path in unmapped
                }
            )
            raise ValueError(
                "Hay imágenes en carpetas no configuradas: "
                f"{', '.join(folders)}. Use los identificadores de "
                f"{config.centers_path}."
            )
        raise ValueError(f"No se encontraron imágenes en {config.images_dir}")

    _report_progress(
        progress_callback,
        0.0,
        "Preparando caché y procesamiento OCR",
    )
    engine: PaddleOcrEngine | None = None
    extracted: list[PatientRecord] = []
    timings: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = [
        {
            "imagen": str(path.relative_to(config.images_dir)),
            "error": "Carpeta de centro no configurada; imagen omitida",
        }
        for path in unmapped
    ]
    processed = 0

    for index, source in enumerate(selected, start=1):
        image_index = index - 1
        image_label = (
            f"Imagen {index} de {len(selected)}: {source.path.name}"
        )
        image_started = perf_counter()
        timing: dict[str, Any] = {
            "imagen": source.path.name,
            "modo": config.ocr_mode,
            "cache_ocr": False,
        }
        try:
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.0),
                f"{image_label} — preprocesamiento",
            )
            stage_started = perf_counter()
            if config.preprocess:
                processed_path = (
                    config.interim_dir
                    / "preprocessed"
                    / source.center_slug
                    / f"{source.path.stem}.jpg"
                )
                preprocess_image(source.path, processed_path)
            else:
                processed_path = source.path
            timing["preprocesamiento_segundos"] = round(
                perf_counter() - stage_started,
                4,
            )
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.14),
                f"{image_label} — detectando tabla",
            )

            stage_started = perf_counter()
            grid_path = (
                config.interim_dir
                / "grids"
                / source.center_slug
                / f"{source.path.stem}.jpg"
            )
            grid = detect_table_grid(processed_path, grid_path)
            timing["deteccion_tabla_segundos"] = round(
                perf_counter() - stage_started,
                4,
            )
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.26),
                f"{image_label} — aplicando OCR",
            )
            rows_dir = (
                config.interim_dir
                / "handwriting_rows"
                / source.center_slug
                / source.path.stem
            )
            stage_started = perf_counter()
            cached = load_cached_ocr(
                config.cache_dir,
                processed_path,
                config.ocr_mode,
            )
            timing["lectura_cache_segundos"] = round(
                perf_counter() - stage_started,
                4,
            )
            if cached is not None:
                lines = cached.lines
                row_audit = cached.audit
                timing["cache_ocr"] = True
                timing["inicializacion_motor_segundos"] = 0.0
                timing["ocr_segundos"] = 0.0
                _report_progress(
                    progress_callback,
                    _image_progress(image_index, len(selected), 0.70),
                    f"{image_label} — OCR recuperado de caché",
                )
            else:
                if engine is None:
                    engine_started = perf_counter()
                    engine = PaddleOcrEngine(config.cache_dir)
                    timing["inicializacion_motor_segundos"] = round(
                        perf_counter() - engine_started,
                        4,
                    )
                else:
                    timing["inicializacion_motor_segundos"] = 0.0
                ocr_started = perf_counter()
                lines, row_audit = _recognize_image(
                    engine,
                    processed_path,
                    grid,
                    config.ocr_mode,
                    rows_dir,
                )
                timing["ocr_segundos"] = round(
                    perf_counter() - ocr_started,
                    4,
                )
                cache_write_started = perf_counter()
                try:
                    save_cached_ocr(
                        config.cache_dir,
                        processed_path,
                        config.ocr_mode,
                        lines,
                        row_audit,
                    )
                    timing["escritura_cache_correcta"] = True
                except OSError:
                    timing["escritura_cache_correcta"] = False
                timing["escritura_cache_segundos"] = round(
                    perf_counter() - cache_write_started,
                    4,
                )
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.78),
                f"{image_label} — interpretando campos",
            )
            if row_audit is not None:
                rows_dir.mkdir(parents=True, exist_ok=True)
                (rows_dir / "audit.json").write_text(
                    json.dumps(row_audit, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            raw_path = (
                config.interim_dir
                / "ocr"
                / source.center_slug
                / f"{source.path.stem}.json"
            )
            save_raw_ocr(raw_path, source.path, lines)
            parsing_started = perf_counter()
            image_records = parse_ocr_lines(
                lines,
                specialties,
                name_lexicons,
                source.center_name,
                source.path.name,
                places,
                grid,
            )
            extracted.extend(image_records)
            timing["interpretacion_segundos"] = round(
                perf_counter() - parsing_started,
                4,
            )
            timing["registros_extraidos"] = len(image_records)
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.96),
                f"{image_label} — finalizando",
            )
            processed += 1
        except Exception as error:  # El lote debe continuar y dejar trazabilidad.
            errors.append(
                {
                    "imagen": source.path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            timing["total_segundos"] = round(
                perf_counter() - image_started,
                4,
            )
            timings.append(timing)
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 1.0),
                f"{image_label} — completada",
            )

    _report_progress(
        progress_callback,
        0.94,
        "Consolidando registros y duplicados",
    )
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    (config.interim_dir / "errores.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.interim_dir / "tiempos.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    consolidation = consolidate_records(extracted)
    consolidated = consolidation.patients
    result = ProcessingResult(
        consolidation=consolidation,
        discovered_images=len(discovered),
        processed_images=processed,
        extracted_records=len(extracted),
        review_records=sum(record.needs_review for record in consolidated),
        errors=tuple(errors),
        specialty_values=tuple(sorted({item.specialty for item in specialties})),
    )
    _report_progress(
        progress_callback,
        1.0,
        "Procesamiento completado",
    )
    return result


def run_pipeline(config: PipelineConfig) -> PipelineReport:
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"El archivo ya existe: {config.output_path}. "
            "Use --force para reemplazarlo."
        )
    processing = process_images(config)
    export_results(
        processing.consolidation,
        config.output_path,
        specialty_values=list(processing.specialty_values),
    )
    return PipelineReport(
        discovered_images=processing.discovered_images,
        processed_images=processing.processed_images,
        extracted_records=processing.extracted_records,
        consolidated_records=len(processing.consolidation.patients),
        review_records=processing.review_records,
        errors=processing.errors,
        output_path=config.output_path,
    )
