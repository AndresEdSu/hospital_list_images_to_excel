from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


ProgressCallback = Callable[[float, float, str], None]

IMAGE_PROGRESS_START = 0.04
IMAGE_PROGRESS_END = 0.92


def image_progress(
    image_index: int,
    total_images: int,
    fraction: float,
) -> float:
    if total_images <= 0:
        return IMAGE_PROGRESS_START
    bounded_fraction = min(1.0, max(0.0, fraction))
    completed_images = image_index + bounded_fraction
    image_share = completed_images / total_images
    return IMAGE_PROGRESS_START + image_share * (
        IMAGE_PROGRESS_END - IMAGE_PROGRESS_START
    )


def report_progress(
    callback: ProgressCallback | None,
    value: float,
    message: str,
) -> None:
    if callback:
        callback(min(1.0, max(0.0, value)), 1.0, message)


@dataclass
class TimingRecorder:
    image_name: str
    mode: str
    values: dict[str, Any] = field(init=False)
    _started: float = field(default_factory=perf_counter, init=False)
    _stage_started: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.values = {
            "imagen": self.image_name,
            "modo": self.mode,
            "cache_ocr": False,
        }

    def start_stage(self) -> None:
        self._stage_started = perf_counter()

    def finish_stage(self, key: str) -> None:
        self.values[key] = round(
            perf_counter() - self._stage_started,
            4,
        )

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def finish(self) -> dict[str, Any]:
        self.values["total_segundos"] = round(
            perf_counter() - self._started,
            4,
        )
        return dict(self.values)


def write_processing_diagnostics(
    interim_dir: Path,
    errors: list[dict[str, str]],
    timings: list[dict[str, Any]],
) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / "errores.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (interim_dir / "tiempos.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
