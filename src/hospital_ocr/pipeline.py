from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

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
from hospital_ocr.ocr_engine import PaddleOcrEngine, save_raw_ocr
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.preprocessing import preprocess_image


OcrMode = Literal["auto", "handwritten", "printed"]
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
        rows = (
            rows_from_grid(image_path, grid)
            if grid is not None
            else detect_text_rows(image_path)
        )
        boundary_source = "cuadrícula" if grid is not None else "renglones"
        if not rows:
            lines = engine.recognize(image_path)
            return lines, _row_audit(
                mode=mode,
                rows=[],
                lines=lines,
                coverage_before=None,
                boundary_source=boundary_source,
                fallback_reason=(
                    "No se pudieron delimitar recortes; se usó OCR global."
                ),
                grid=grid,
            )
        if grid is not None:
            initial_lines = engine.recognize(image_path)
            refined_lines = engine.recognize_grid_cells(
                image_path,
                cells_from_grid(grid),
                rows_dir,
            )
            lines = merge_grid_ocr(initial_lines, refined_lines, grid)
            coverage_before = _grid_row_coverage(initial_lines, grid)
        else:
            segmented_lines = engine.recognize_rows(
                image_path,
                rows,
                rows_dir,
            )
            lines = merge_row_ocr(
                [],
                segmented_lines,
                rows,
            )
            coverage_before = None
        return lines, _row_audit(
            mode=mode,
            rows=rows,
            lines=lines,
            coverage_before=coverage_before,
            boundary_source=boundary_source,
            grid=grid,
        )

    initial_lines = engine.recognize(image_path)
    if grid is not None:
        return initial_lines, None
    rows = detect_text_rows(image_path)
    if not needs_row_ocr(initial_lines, rows):
        return initial_lines, None
    coverage_before = row_ocr_coverage(initial_lines, rows)
    segmented_lines = engine.recognize_rows(
        image_path,
        rows,
        rows_dir,
    )
    lines = merge_row_ocr(initial_lines, segmented_lines, rows)
    return lines, _row_audit(
        mode=mode,
        rows=rows,
        lines=lines,
        coverage_before=coverage_before,
        boundary_source="renglones",
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
        "Inicializando el motor OCR",
    )
    engine = PaddleOcrEngine(config.cache_dir)
    _report_progress(
        progress_callback,
        _IMAGE_PROGRESS_START,
        "Motor OCR listo",
    )
    extracted: list[PatientRecord] = []
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
        try:
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.0),
                f"{image_label} — preprocesamiento",
            )
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
            _report_progress(
                progress_callback,
                _image_progress(image_index, len(selected), 0.14),
                f"{image_label} — detectando tabla",
            )

            grid_path = (
                config.interim_dir
                / "grids"
                / source.center_slug
                / f"{source.path.stem}.jpg"
            )
            grid = detect_table_grid(processed_path, grid_path)
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
            lines, row_audit = _recognize_image(
                engine,
                processed_path,
                grid,
                config.ocr_mode,
                rows_dir,
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
            extracted.extend(
                parse_ocr_lines(
                    lines,
                    specialties,
                    name_lexicons,
                    source.center_name,
                    source.path.name,
                    places,
                    grid,
                )
            )
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
