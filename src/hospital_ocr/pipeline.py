from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hospital_ocr.catalogs import load_centers, load_places, load_specialties
from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.discovery import (
    discover_images,
    find_unmapped_images,
    select_evenly,
)
from hospital_ocr.exporting import export_results
from hospital_ocr.image_processor import (
    ImageProcessingOptions,
    ImageProcessor,
)
from hospital_ocr.models import (
    ConsolidationResult,
    PatientRecord,
)
from hospital_ocr.name_splitter import load_name_lexicons
from hospital_ocr.pipeline_types import OCR_MODES, OcrMode
from hospital_ocr.processing_metrics import (
    ProgressCallback,
    report_progress,
    write_processing_diagnostics,
)


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

    report_progress(
        progress_callback,
        0.0,
        "Preparando caché y procesamiento OCR",
    )
    processor = ImageProcessor(
        ImageProcessingOptions(
            interim_dir=config.interim_dir,
            cache_dir=config.cache_dir,
            preprocess=config.preprocess,
            ocr_mode=config.ocr_mode,
        ),
        specialties,
        places,
        name_lexicons,
    )
    extracted: list[PatientRecord] = []
    timings: list[dict[str, object]] = []
    errors: list[dict[str, str]] = [
        {
            "imagen": str(path.relative_to(config.images_dir)),
            "error": "Carpeta de centro no configurada; imagen omitida",
        }
        for path in unmapped
    ]
    processed = 0

    for image_index, source in enumerate(selected):
        outcome = processor.process(
            source,
            image_index=image_index,
            total_images=len(selected),
            progress_callback=progress_callback,
        )
        extracted.extend(outcome.records)
        timings.append(outcome.timing)
        if outcome.error is not None:
            errors.append(outcome.error)
        if outcome.processed:
            processed += 1

    report_progress(
        progress_callback,
        0.94,
        "Consolidando registros y duplicados",
    )
    write_processing_diagnostics(config.interim_dir, errors, timings)

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
    report_progress(
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
