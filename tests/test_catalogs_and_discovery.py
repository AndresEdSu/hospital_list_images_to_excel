from pathlib import Path

from hospital_ocr.catalogs import load_centers, load_places, write_center_catalog
from hospital_ocr.discovery import (
    discover_images,
    find_unmapped_images,
    select_evenly,
)
from hospital_ocr.matching import match_place


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


def test_write_center_catalog_for_custom_web_center(tmp_path: Path) -> None:
    centers_path = tmp_path / "centros.csv"

    write_center_catalog(
        centers_path,
        "otro_centro",
        "  Hospital Comunitario de Prueba  ",
    )

    assert load_centers(centers_path) == {
        "otro_centro": "Hospital Comunitario de Prueba"
    }


def test_place_catalog_normalizes_aliases(tmp_path: Path) -> None:
    places_path = tmp_path / "lugares.csv"
    places_path.write_text(
        "alias,lugar\nLa Guaira,La Guaira\nPETARE,Petare\n",
        encoding="utf-8",
    )

    places = load_places(places_path)

    assert [(place.alias, place.name) for place in places] == [
        ("la guaira", "La Guaira"),
        ("petare", "Petare"),
    ]


def test_project_place_catalog_includes_states_and_health_centers() -> None:
    places = load_places(Path("config/lugares.csv"))

    state = match_place("Edo. Tachira", places, contextual=True)
    center = match_place("Hospital Miguel Perez Carreno", places, contextual=True)

    assert state is not None
    assert state.name == "Tachira"
    assert center is not None
    assert center.name == "Hospital Dr. Miguel Perez Carreno"


def test_pilot_selection_includes_first_and_last(tmp_path: Path) -> None:
    centers = {"hospital_demo": "Hospital de Prueba"}
    image_dir = tmp_path / "images" / "hospital_demo"
    image_dir.mkdir(parents=True)
    for index in range(10):
        (image_dir / f"{index:02}.jpg").write_bytes(b"x")

    images = discover_images(tmp_path / "images", centers)
    selected = select_evenly(images, 3)

    assert [item.path.name for item in selected] == ["00.jpg", "04.jpg", "09.jpg"]


def test_unmapped_images_are_reported_for_cli_layout(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    configured = images_dir / "hospital_demo"
    unknown = images_dir / "nombre_incorrecto"
    configured.mkdir(parents=True)
    unknown.mkdir()
    (configured / "lista.jpg").write_bytes(b"x")
    (unknown / "otra.jpg").write_bytes(b"x")
    (images_dir / "suelta.png").write_bytes(b"x")

    unmapped = find_unmapped_images(
        images_dir,
        {"hospital_demo": "Hospital de Prueba"},
    )

    assert [path.relative_to(images_dir).as_posix() for path in unmapped] == [
        "nombre_incorrecto/otra.jpg",
        "suelta.png",
    ]
