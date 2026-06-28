from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hospital_ocr.catalogs import load_centers, load_specialties
from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.discovery import discover_images, select_evenly
from hospital_ocr.exporting import export_results
from hospital_ocr.models import PatientRecord
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


@dataclass(frozen=True)
class PipelineReport:
    discovered_images: int
    processed_images: int
    extracted_records: int
    consolidated_records: int
    review_records: int
    errors: tuple[dict[str, str], ...]
    output_path: Path


def run_pipeline(config: PipelineConfig) -> PipelineReport:
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"El archivo ya existe: {config.output_path}. "
            "Use --force para reemplazarlo."
        )
    centers = load_centers(config.centers_path)
    specialties = load_specialties(config.specialties_path)
    name_lexicons = load_name_lexicons(
        config.given_names_path,
        config.surnames_path,
    )
    discovered = discover_images(config.images_dir, centers)
    selected = select_evenly(discovered, config.limit)
    if not selected:
        raise ValueError(f"No se encontraron imágenes en {config.images_dir}")

    engine = PaddleOcrEngine(config.cache_dir)
    extracted: list[PatientRecord] = []
    errors: list[dict[str, str]] = []
    processed = 0

    for source in selected:
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

            lines = engine.recognize(processed_path)
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

    config.interim_dir.mkdir(parents=True, exist_ok=True)
    (config.interim_dir / "errores.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    consolidation = consolidate_records(extracted)
    consolidated = consolidation.patients
    export_results(
        consolidation,
        config.output_path,
        specialty_values=[item.specialty for item in specialties],
    )
    return PipelineReport(
        discovered_images=len(discovered),
        processed_images=processed,
        extracted_records=len(extracted),
        consolidated_records=len(consolidated),
        review_records=sum(record.needs_review for record in consolidated),
        errors=tuple(errors),
        output_path=config.output_path,
    )
