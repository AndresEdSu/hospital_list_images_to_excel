from __future__ import annotations

from pathlib import Path

from hospital_ocr.models import ImageSource


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def find_unmapped_images(
    images_dir: Path,
    centers: dict[str, str],
) -> list[Path]:
    if not images_dir.is_dir():
        return []
    configured_directories = {
        (images_dir / center_slug).resolve() for center_slug in centers
    }
    unmapped = []
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not any(parent in configured_directories for parent in path.parents):
            unmapped.append(path)
    return sorted(unmapped, key=lambda path: str(path).lower())


def discover_images(images_dir: Path, centers: dict[str, str]) -> list[ImageSource]:
    images: list[ImageSource] = []
    for center_slug, center_name in centers.items():
        center_dir = images_dir / center_slug
        if not center_dir.is_dir():
            continue
        for path in sorted(center_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(ImageSource(path, center_slug, center_name))
    return sorted(images, key=lambda item: str(item.path).lower())


def select_evenly(images: list[ImageSource], limit: int | None) -> list[ImageSource]:
    if limit is None or limit >= len(images):
        return images
    if limit <= 0:
        return []
    if limit == 1:
        return [images[0]]
    indexes = {
        round(index * (len(images) - 1) / (limit - 1)) for index in range(limit)
    }
    return [images[index] for index in sorted(indexes)]
