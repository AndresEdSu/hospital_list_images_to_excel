from pathlib import Path

from hospital_ocr.catalogs import load_centers
from hospital_ocr.discovery import discover_images, select_evenly


def test_center_folder_determines_center(tmp_path: Path) -> None:
    centers_path = tmp_path / "centros.csv"
    centers_path.write_text(
        "carpeta,centro\nhospital_demo,Hospital de Prueba\n",
        encoding="utf-8",
    )
    image_dir = tmp_path / "images" / "hospital_demo"
    image_dir.mkdir(parents=True)
    (image_dir / "lista.jpg").write_bytes(b"not-an-image")
    (image_dir / "notas.txt").write_text("ignorar", encoding="utf-8")

    centers = load_centers(centers_path)
    images = discover_images(tmp_path / "images", centers)

    assert centers == {"hospital_demo": "Hospital de Prueba"}
    assert len(images) == 1
    assert images[0].center_name == "Hospital de Prueba"


def test_pilot_selection_includes_first_and_last(tmp_path: Path) -> None:
    centers = {"hospital_demo": "Hospital de Prueba"}
    image_dir = tmp_path / "images" / "hospital_demo"
    image_dir.mkdir(parents=True)
    for index in range(10):
        (image_dir / f"{index:02}.jpg").write_bytes(b"x")

    images = discover_images(tmp_path / "images", centers)
    selected = select_evenly(images, 3)

    assert [item.path.name for item in selected] == ["00.jpg", "04.jpg", "09.jpg"]
