import json
from pathlib import Path

from PIL import Image

from hospital_ocr.models import OcrLine
from hospital_ocr.pipeline import PipelineConfig, process_images


def _write_catalogs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "centers": tmp_path / "centros.csv",
        "specialties": tmp_path / "especialidades.csv",
        "names": tmp_path / "nombres.csv",
        "surnames": tmp_path / "apellidos.csv",
    }
    paths["centers"].write_text(
        "carpeta,centro\nhospital,Hospital de Prueba\n",
        encoding="utf-8",
    )
    paths["specialties"].write_text(
        "alias,especialidad,area\npediatria,Pediatría,\n",
        encoding="utf-8",
    )
    paths["names"].write_text(
        "termino,peso\nmaria,1\n",
        encoding="utf-8",
    )
    paths["surnames"].write_text(
        "termino,peso\nperez,1\n",
        encoding="utf-8",
    )
    return paths


def test_second_pipeline_run_uses_cached_ocr_without_loading_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalogs = _write_catalogs(tmp_path)
    images_dir = tmp_path / "images"
    image_path = images_dir / "hospital" / "lista.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (400, 200), "white").save(image_path)
    calls = {"initializations": 0, "recognitions": 0}

    class FakeEngine:
        def __init__(self, cache_dir: Path) -> None:
            calls["initializations"] += 1

        def recognize(self, path: Path) -> list[OcrLine]:
            calls["recognitions"] += 1
            return [
                OcrLine(
                    "María Pérez 20 años F",
                    0.95,
                    (20, 50, 300, 90),
                    400,
                    200,
                )
            ]

    monkeypatch.setattr(
        "hospital_ocr.image_processor.PaddleOcrEngine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "hospital_ocr.image_processor.detect_table_grid",
        lambda image, debug: None,
    )

    def config(interim: str) -> PipelineConfig:
        return PipelineConfig(
            images_dir=images_dir,
            centers_path=catalogs["centers"],
            specialties_path=catalogs["specialties"],
            given_names_path=catalogs["names"],
            surnames_path=catalogs["surnames"],
            interim_dir=tmp_path / interim,
            output_path=tmp_path / f"{interim}.xlsx",
            cache_dir=tmp_path / "cache",
            preprocess=False,
        )

    process_images(config("first"))
    process_images(config("second"))

    assert calls == {"initializations": 1, "recognitions": 1}
    timings = json.loads(
        (tmp_path / "second" / "tiempos.json").read_text(encoding="utf-8")
    )
    assert timings[0]["cache_ocr"] is True
    assert timings[0]["ocr_segundos"] == 0.0
