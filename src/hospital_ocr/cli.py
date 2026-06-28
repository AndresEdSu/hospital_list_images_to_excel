from __future__ import annotations

import argparse
from pathlib import Path

from hospital_ocr.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convierte imágenes de listas hospitalarias en un Excel consolidado."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/input/images"))
    parser.add_argument("--centers", type=Path, default=Path("config/centros.csv"))
    parser.add_argument(
        "--specialties", type=Path, default=Path("config/especialidades.csv")
    )
    parser.add_argument(
        "--given-names", type=Path, default=Path("config/nombres_comunes.csv")
    )
    parser.add_argument(
        "--surnames", type=Path, default=Path("config/apellidos_comunes.csv")
    )
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/pacientes_consolidados.xlsx"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/paddlex")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa una muestra distribuida de N imágenes.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Envía las imágenes originales directamente al OCR.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite reemplazar un archivo de salida existente.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_pipeline(
        PipelineConfig(
            images_dir=args.images_dir,
            centers_path=args.centers,
            specialties_path=args.specialties,
            given_names_path=args.given_names,
            surnames_path=args.surnames,
            interim_dir=args.interim_dir,
            output_path=args.output,
            cache_dir=args.cache_dir,
            limit=args.limit,
            preprocess=not args.skip_preprocessing,
            overwrite=args.force,
        )
    )
    print(f"Imágenes encontradas: {report.discovered_images}")
    print(f"Imágenes procesadas: {report.processed_images}")
    print(f"Registros extraídos: {report.extracted_records}")
    print(f"Pacientes consolidados: {report.consolidated_records}")
    print(f"Registros para revisión: {report.review_records}")
    print(f"Errores: {len(report.errors)}")
    print(f"Excel: {report.output_path.resolve()}")
