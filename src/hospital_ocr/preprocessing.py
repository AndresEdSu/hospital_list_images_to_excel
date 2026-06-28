from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def preprocess_image(
    source: Path,
    destination: Path,
    minimum_long_side: int = 1600,
) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        long_side = max(image.size)
        if long_side < minimum_long_side:
            scale = minimum_long_side / long_side
            target = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(target, Image.Resampling.LANCZOS)
        image = ImageOps.autocontrast(image, cutoff=1)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        image.save(destination, format="JPEG", quality=95, optimize=True)
        return image.size
