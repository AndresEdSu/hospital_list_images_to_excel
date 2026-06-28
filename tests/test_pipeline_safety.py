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
