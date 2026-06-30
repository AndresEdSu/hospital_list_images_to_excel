from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from hospital_ocr.handwriting import TextRow
from hospital_ocr.models import OcrLine


class PaddleOcrEngine:
    def __init__(self, cache_dir: Path) -> None:
        resolved_cache = cache_dir.resolve()
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(resolved_cache))
        os.environ.setdefault("HF_HOME", str(resolved_cache / "huggingface"))

        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            lang="es",
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
            text_rec_score_thresh=0.0,
        )

    def recognize(self, image_path: Path) -> list[OcrLine]:
        results = list(self._engine.predict(str(image_path)))
        if not results:
            return []
        result = results[0]
        texts = list(result.get("rec_texts", []))
        scores = list(result.get("rec_scores", []))
        boxes = list(result.get("rec_boxes", []))
        with Image.open(image_path) as image:
            width, height = image.size

        lines: list[OcrLine] = []
        for text, score, box in zip(texts, scores, boxes, strict=False):
            values = [int(round(float(value))) for value in box]
            if len(values) != 4:
                continue
            lines.append(
                OcrLine(
                    text=str(text).strip(),
                    score=float(score),
                    box=(values[0], values[1], values[2], values[3]),
                    image_width=width,
                    image_height=height,
                )
            )
        return lines

    def recognize_rows(
        self,
        image_path: Path,
        rows: list[TextRow],
        artifacts_dir: Path,
    ) -> list[OcrLine]:
        """Run a second OCR pass on enlarged, enhanced text-row crops."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mapped_lines: list[OcrLine] = []
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            image_width, image_height = image.size

            for index, row in enumerate(rows, start=1):
                left, top, right, bottom = row.box
                vertical_padding = max(2, round((bottom - top) * 0.03))
                top = max(0, top - vertical_padding)
                bottom = min(image_height, bottom + vertical_padding)
                crop = image.crop((left, top, right, bottom))
                crop.save(artifacts_dir / f"row_{index:03d}_source.jpg", quality=92)

                scale = 3
                overlap = round(crop.width * 0.12)
                seam = crop.width // 2
                windows = (
                    ("left", 0, min(crop.width, seam + overlap)),
                    ("right", max(0, seam - overlap), crop.width),
                )
                for window_name, window_left, window_right in windows:
                    window = crop.crop(
                        (window_left, 0, window_right, crop.height)
                    )
                    window = ImageEnhance.Contrast(window).enhance(1.12)
                    window = window.resize(
                        (window.width * scale, window.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                    window_path = (
                        artifacts_dir
                        / f"row_{index:03d}_{window_name}_enhanced.png"
                    )
                    window.save(window_path)

                    for line in self.recognize(window_path):
                        mapped = OcrLine(
                            text=line.text,
                            score=line.score,
                            box=(
                                round(line.box[0] / scale)
                                + left
                                + window_left,
                                round(line.box[1] / scale) + top,
                                round(line.box[2] / scale)
                                + left
                                + window_left,
                                round(line.box[3] / scale) + top,
                            ),
                            image_width=image_width,
                            image_height=image_height,
                        )
                        local_center = mapped.center_x - left
                        owns_center = (
                            window_name == "left" and local_center < seam
                        ) or (
                            window_name == "right" and local_center >= seam
                        )
                        if owns_center:
                            mapped_lines.append(mapped)
        return mapped_lines


def save_raw_ocr(path: Path, image_path: Path, lines: list[OcrLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "imagen": image_path.name,
        "lineas": [
            {
                "texto": line.text,
                "confianza": round(line.score, 6),
                "caja": list(line.box),
            }
            for line in lines
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
