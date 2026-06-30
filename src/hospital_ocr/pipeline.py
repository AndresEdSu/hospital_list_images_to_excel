from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    detect_text_rows,
    merge_row_ocr,
    needs_row_ocr,
    row_ocr_coverage,
)
from hospital_ocr.models import ConsolidationResult, PatientRecord
from hospital_ocr.name_splitter import load_name_lexicons
from hospital_ocr.ocr_engine import PaddleOcrEngine, save_raw_ocr
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.preprocessing import preprocess_image


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


ProgressCallback = Callable[[int, int, str], None]


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

    if progress_callback:
        progress_callback(0, len(selected), "Inicializando el motor OCR")
    engine = PaddleOcrEngine(config.cache_dir)
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
        try:
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

            grid_path = (
                config.interim_dir
                / "grids"
                / source.center_slug
                / f"{source.path.stem}.jpg"
            )
            grid = detect_table_grid(processed_path, grid_path)
            lines = engine.recognize(processed_path)
            if grid is None:
                rows = detect_text_rows(processed_path)
                if needs_row_ocr(lines, rows):
                    coverage_before = row_ocr_coverage(lines, rows)
                    rows_dir = (
                        config.interim_dir
                        / "handwriting_rows"
                        / source.center_slug
                        / source.path.stem
                    )
                    segmented_lines = engine.recognize_rows(
                        processed_path,
                        rows,
                        rows_dir,
                    )
                    lines = merge_row_ocr(lines, segmented_lines, rows)
                    (rows_dir / "audit.json").write_text(
                        json.dumps(
                            {
                                "fallback_activado": True,
                                "cobertura_antes": round(coverage_before, 4),
                                "cobertura_despues": round(
                                    row_ocr_coverage(lines, rows),
                                    4,
                                ),
                                "renglones": [
                                    {
                                        "caja": list(row.box),
                                        "fuerza": round(row.strength, 4),
                                    }
                                    for row in rows
                                ],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
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
            processed += 1
        except Exception as error:  # El lote debe continuar y dejar trazabilidad.
            errors.append(
                {
                    "imagen": source.path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if progress_callback:
                progress_callback(index, len(selected), source.path.name)

    config.interim_dir.mkdir(parents=True, exist_ok=True)
    (config.interim_dir / "errores.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    consolidation = consolidate_records(extracted)
    consolidated = consolidation.patients
    return ProcessingResult(
        consolidation=consolidation,
        discovered_images=len(discovered),
        processed_images=processed,
        extracted_records=len(extracted),
        review_records=sum(record.needs_review for record in consolidated),
        errors=tuple(errors),
        specialty_values=tuple(sorted({item.specialty for item in specialties})),
    )


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
