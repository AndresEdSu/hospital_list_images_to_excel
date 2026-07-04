from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hospital_ocr.grid_detector import detect_table_grid
from hospital_ocr.models import ImageSource, PatientRecord, Place, Specialty
from hospital_ocr.name_splitter import NameLexicons
from hospital_ocr.ocr_cache import load_cached_ocr, save_cached_ocr
from hospital_ocr.ocr_engine import PaddleOcrEngine, save_raw_ocr
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.pipeline_types import OcrMode
from hospital_ocr.preprocessing import preprocess_image
from hospital_ocr.processing_metrics import (
    ProgressCallback,
    TimingRecorder,
    image_progress,
    report_progress,
)
from hospital_ocr.recognition import recognize_image


@dataclass(frozen=True)
class ImageProcessingOptions:
    interim_dir: Path
    cache_dir: Path
    preprocess: bool
    ocr_mode: OcrMode


@dataclass(frozen=True)
class ImageProcessingOutcome:
    records: list[PatientRecord]
    timing: dict[str, object]
    error: dict[str, str] | None
    processed: bool


class ImageProcessor:
    def __init__(
        self,
        options: ImageProcessingOptions,
        specialties: list[Specialty],
        places: list[Place],
        name_lexicons: NameLexicons,
    ) -> None:
        self.options = options
        self.specialties = specialties
        self.places = places
        self.name_lexicons = name_lexicons
        self._engine: PaddleOcrEngine | None = None

    def process(
        self,
        source: ImageSource,
        image_index: int,
        total_images: int,
        progress_callback: ProgressCallback | None = None,
    ) -> ImageProcessingOutcome:
        image_label = (
            f"Imagen {image_index + 1} de {total_images}: {source.path.name}"
        )
        timing = TimingRecorder(source.path.name, self.options.ocr_mode)
        records: list[PatientRecord] = []
        error: dict[str, str] | None = None
        processed = False
        try:
            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 0.0),
                f"{image_label} — preprocesamiento",
            )
            timing.start_stage()
            if self.options.preprocess:
                processed_path = (
                    self.options.interim_dir
                    / "preprocessed"
                    / source.center_slug
                    / f"{source.path.stem}.jpg"
                )
                preprocess_image(source.path, processed_path)
            else:
                processed_path = source.path
            timing.finish_stage("preprocesamiento_segundos")

            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 0.14),
                f"{image_label} — detectando tabla",
            )
            timing.start_stage()
            grid_path = (
                self.options.interim_dir
                / "grids"
                / source.center_slug
                / f"{source.path.stem}.jpg"
            )
            grid = detect_table_grid(processed_path, grid_path)
            timing.finish_stage("deteccion_tabla_segundos")

            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 0.26),
                f"{image_label} — aplicando OCR",
            )
            rows_dir = (
                self.options.interim_dir
                / "handwriting_rows"
                / source.center_slug
                / source.path.stem
            )
            timing.start_stage()
            cached = load_cached_ocr(
                self.options.cache_dir,
                processed_path,
                self.options.ocr_mode,
            )
            timing.finish_stage("lectura_cache_segundos")
            if cached is not None:
                lines = cached.lines
                row_audit = cached.audit
                timing.set("cache_ocr", True)
                timing.set("inicializacion_motor_segundos", 0.0)
                timing.set("ocr_segundos", 0.0)
                report_progress(
                    progress_callback,
                    image_progress(image_index, total_images, 0.70),
                    f"{image_label} — OCR recuperado de caché",
                )
            else:
                if self._engine is None:
                    timing.start_stage()
                    self._engine = PaddleOcrEngine(self.options.cache_dir)
                    timing.finish_stage("inicializacion_motor_segundos")
                else:
                    timing.set("inicializacion_motor_segundos", 0.0)
                timing.start_stage()
                lines, row_audit = recognize_image(
                    self._engine,
                    processed_path,
                    grid,
                    self.options.ocr_mode,
                    rows_dir,
                )
                timing.finish_stage("ocr_segundos")
                timing.start_stage()
                try:
                    save_cached_ocr(
                        self.options.cache_dir,
                        processed_path,
                        self.options.ocr_mode,
                        lines,
                        row_audit,
                    )
                    timing.set("escritura_cache_correcta", True)
                except OSError:
                    timing.set("escritura_cache_correcta", False)
                timing.finish_stage("escritura_cache_segundos")

            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 0.78),
                f"{image_label} — interpretando campos",
            )
            if row_audit is not None:
                rows_dir.mkdir(parents=True, exist_ok=True)
                (rows_dir / "audit.json").write_text(
                    json.dumps(row_audit, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            raw_path = (
                self.options.interim_dir
                / "ocr"
                / source.center_slug
                / f"{source.path.stem}.json"
            )
            save_raw_ocr(raw_path, source.path, lines)
            timing.start_stage()
            records = parse_ocr_lines(
                lines,
                self.specialties,
                self.name_lexicons,
                source.center_name,
                source.path.name,
                self.places,
                grid,
            )
            timing.finish_stage("interpretacion_segundos")
            timing.set("registros_extraidos", len(records))

            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 0.96),
                f"{image_label} — finalizando",
            )
            processed = True
        except Exception as caught:
            error = {
                "imagen": source.path.name,
                "error": f"{type(caught).__name__}: {caught}",
            }
        finally:
            report_progress(
                progress_callback,
                image_progress(image_index, total_images, 1.0),
                f"{image_label} — completada",
            )

        return ImageProcessingOutcome(
            records=records,
            timing=timing.finish(),
            error=error,
            processed=processed,
        )
