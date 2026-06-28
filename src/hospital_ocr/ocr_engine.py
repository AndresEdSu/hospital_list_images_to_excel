from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

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
