from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hospital_ocr.models import OcrLine
from hospital_ocr.pipeline_types import OcrMode


OCR_RESULT_CACHE_VERSION = "refinement-policy-v6"


@dataclass(frozen=True)
class CachedOcr:
    lines: list[OcrLine]
    audit: dict[str, Any] | None


def _cache_key(image_path: Path, mode: OcrMode) -> str:
    digest = hashlib.sha256()
    digest.update(OCR_RESULT_CACHE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(mode.encode("ascii"))
    digest.update(b"\0")
    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, image_path: Path, mode: OcrMode) -> Path:
    return cache_dir / "ocr_results" / f"{_cache_key(image_path, mode)}.json"


def load_cached_ocr(
    cache_dir: Path,
    image_path: Path,
    mode: OcrMode,
) -> CachedOcr | None:
    path = _cache_path(cache_dir, image_path, mode)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != OCR_RESULT_CACHE_VERSION:
            return None
        lines = [
            OcrLine(
                text=str(item["text"]),
                score=float(item["score"]),
                box=tuple(int(value) for value in item["box"]),
                image_width=int(item["image_width"]),
                image_height=int(item["image_height"]),
            )
            for item in payload.get("lines", [])
        ]
        audit = payload.get("audit")
        return CachedOcr(
            lines=lines,
            audit=audit if isinstance(audit, dict) else None,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_cached_ocr(
    cache_dir: Path,
    image_path: Path,
    mode: OcrMode,
    lines: list[OcrLine],
    audit: dict[str, Any] | None,
) -> None:
    path = _cache_path(cache_dir, image_path, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": OCR_RESULT_CACHE_VERSION,
        "mode": mode,
        "lines": [
            {
                "text": line.text,
                "score": line.score,
                "box": list(line.box),
                "image_width": line.image_width,
                "image_height": line.image_height,
            }
            for line in lines
        ],
        "audit": audit,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)
