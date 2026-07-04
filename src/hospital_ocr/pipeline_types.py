from typing import Literal


OcrMode = Literal["auto", "handwritten", "printed"]
OCR_MODES: tuple[OcrMode, ...] = ("auto", "handwritten", "printed")
