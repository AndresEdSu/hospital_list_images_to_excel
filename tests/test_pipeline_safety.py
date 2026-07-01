from pathlib import Path

import pytest

from hospital_ocr.pipeline import PipelineConfig, run_pipeline


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "pacientes.xlsx"
    output.write_bytes(b"existing")
    config = PipelineConfig(
        images_dir=tmp_path / "images",
        centers_path=tmp_path / "centros.csv",
        specialties_path=tmp_path / "especialidades.csv",
        given_names_path=tmp_path / "nombres.csv",
        surnames_path=tmp_path / "apellidos.csv",
        interim_dir=tmp_path / "interim",
        output_path=output,
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(FileExistsError, match="--force"):
        run_pipeline(config)

    assert output.read_bytes() == b"existing"


def test_pipeline_rejects_unknown_ocr_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Modo OCR no válido"):
        PipelineConfig(
            images_dir=tmp_path / "images",
            centers_path=tmp_path / "centros.csv",
            specialties_path=tmp_path / "especialidades.csv",
            given_names_path=tmp_path / "nombres.csv",
            surnames_path=tmp_path / "apellidos.csv",
            interim_dir=tmp_path / "interim",
            output_path=tmp_path / "output.xlsx",
            cache_dir=tmp_path / "cache",
            ocr_mode="unknown",  # type: ignore[arg-type]
        )
