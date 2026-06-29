from __future__ import annotations

import re
import shutil
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class UploadedFile(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def create_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    session = root / uuid4().hex
    session.mkdir()
    return session


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_stem = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_stem).strip("_")
    return safe[:60] or "imagen"


def save_uploaded_images(
    uploaded_files: list[UploadedFile],
    images_root: Path,
    center_slug: str,
) -> list[Path]:
    destination = images_root / center_slug
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Extensión no permitida: {uploaded.name}")
        data = uploaded.getvalue()
        if not data:
            raise ValueError(f"El archivo está vacío: {uploaded.name}")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"El archivo supera 15 MB: {uploaded.name}")
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
                image_format = image.format
        except Exception as error:
            raise ValueError(f"No es una imagen válida: {uploaded.name}") from error
        if image_format not in ALLOWED_FORMATS:
            raise ValueError(f"Formato de imagen no permitido: {image_format}")

        filename = f"{index:03d}_{_safe_stem(uploaded.name)}{suffix}"
        path = destination / filename
        path.write_bytes(data)
        saved.append(path)
    return saved


def remove_session(session: Path, sessions_root: Path) -> None:
    resolved_root = sessions_root.resolve()
    resolved_session = session.resolve()
    if resolved_session == resolved_root or resolved_root not in resolved_session.parents:
        raise ValueError("La sesión está fuera del directorio permitido")
    if resolved_session.exists():
        shutil.rmtree(resolved_session)


def cleanup_old_sessions(
    sessions_root: Path,
    maximum_age_hours: int = 24,
) -> int:
    if not sessions_root.exists():
        return 0
    threshold = time.time() - (maximum_age_hours * 3600)
    removed = 0
    for child in sessions_root.iterdir():
        if child.is_dir() and child.stat().st_mtime < threshold:
            remove_session(child, sessions_root)
            removed += 1
    return removed
