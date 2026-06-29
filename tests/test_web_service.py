import os
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from hospital_ocr.web_service import (
    cleanup_old_sessions,
    create_session,
    remove_session,
    save_uploaded_images,
)


class FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_uploaded_image_is_validated_and_renamed(tmp_path: Path) -> None:
    paths = save_uploaded_images(
        [FakeUpload("../../Lista médica.jpg", image_bytes())],
        tmp_path / "images",
        "hospital_demo",
    )

    assert len(paths) == 1
    assert paths[0].is_file()
    assert paths[0].parent == tmp_path / "images" / "hospital_demo"
    assert ".." not in paths[0].name


def test_non_image_upload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="imagen válida"):
        save_uploaded_images(
            [FakeUpload("lista.jpg", b"not-an-image")],
            tmp_path / "images",
            "hospital_demo",
        )


def test_cleanup_only_removes_old_sessions(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    old = create_session(root)
    recent = create_session(root)
    old_time = time.time() - (48 * 3600)
    os.utime(old, (old_time, old_time))

    removed = cleanup_old_sessions(root, maximum_age_hours=24)

    assert removed == 1
    assert not old.exists()
    assert recent.exists()
    remove_session(recent, root)
