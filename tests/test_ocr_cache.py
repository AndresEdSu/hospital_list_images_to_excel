from pathlib import Path

from PIL import Image

from hospital_ocr.models import OcrLine
from hospital_ocr.ocr_cache import load_cached_ocr, save_cached_ocr


def _image(path: Path, color: str = "white") -> Path:
    Image.new("RGB", (40, 30), color).save(path)
    return path


def test_ocr_cache_round_trip_and_mode_separation(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "image.png")
    cache_dir = tmp_path / "cache"
    lines = [OcrLine("María Pérez", 0.91, (1, 2, 30, 20), 40, 30)]
    audit = {"refuerzo": {"celdas_seleccionadas": 2}}

    save_cached_ocr(cache_dir, image_path, "auto", lines, audit)

    cached = load_cached_ocr(cache_dir, image_path, "auto")
    assert cached is not None
    assert cached.lines == lines
    assert cached.audit == audit
    assert load_cached_ocr(cache_dir, image_path, "printed") is None


def test_ocr_cache_key_changes_with_image_content(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "image.png")
    cache_dir = tmp_path / "cache"
    save_cached_ocr(
        cache_dir,
        image_path,
        "auto",
        [OcrLine("original", 0.9, (1, 1, 20, 20), 40, 30)],
        None,
    )

    _image(image_path, "black")

    assert load_cached_ocr(cache_dir, image_path, "auto") is None
