from __future__ import annotations

import csv
from pathlib import Path

from hospital_ocr.models import Place, Specialty
from hospital_ocr.text import normalize_text


def load_centers(path: Path) -> dict[str, str]:
    centers: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            slug = (row.get("carpeta") or "").strip()
            name = (row.get("centro") or "").strip()
            if not slug or not name:
                raise ValueError(f"Centro incompleto en {path}")
            if slug in centers:
                raise ValueError(f"Centro duplicado: {slug}")
            centers[slug] = name
    if not centers:
        raise ValueError(f"No hay centros configurados en {path}")
    return centers


def write_center_catalog(path: Path, center_slug: str, center_name: str) -> None:
    """Create a minimal center catalog for one isolated web session."""
    slug = center_slug.strip()
    name = center_name.strip()
    if not slug or not name:
        raise ValueError("El centro de salud no puede estar vacío")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["carpeta", "centro"])
        writer.writeheader()
        writer.writerow({"carpeta": slug, "centro": name})


def load_specialties(path: Path) -> list[Specialty]:
    specialties: list[Specialty] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            alias = normalize_text(row.get("alias") or "")
            specialty = (row.get("especialidad") or "").strip()
            area = (row.get("area") or "").strip()
            if alias and specialty:
                specialties.append(Specialty(alias, specialty, area))
    return sorted(specialties, key=lambda item: len(item.alias), reverse=True)


def load_places(path: Path | None) -> list[Place]:
    if path is None or not path.exists():
        return []
    places: list[Place] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            alias = normalize_text(row.get("alias") or "")
            name = (row.get("lugar") or "").strip()
            if alias and name:
                places.append(Place(alias, name))
    return sorted(places, key=lambda item: len(item.alias), reverse=True)
