from pathlib import Path

from PIL import Image, ImageDraw

from hospital_ocr.preprocessing import _estimate_horizontal_skew, preprocess_image


def test_preprocess_image_normalizes_and_writes_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "shadowed.png"
    destination = tmp_path / "processed" / "shadowed.jpg"
    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    for x in range(image.width):
        shade = 255 - round(80 * x / image.width)
        draw.line((x, 0, x, image.height), fill=(shade, shade, shade))
    draw.text((100, 180), "Nombre y Apellido", fill="black")
    image.save(source)

    size = preprocess_image(source, destination, minimum_long_side=1000)

    assert destination.exists()
    assert max(size) == 1000
    with Image.open(destination) as processed:
        assert processed.format == "JPEG"
        assert processed.mode == "RGB"


def test_preprocess_deskews_clear_horizontal_slant(tmp_path: Path) -> None:
    source = tmp_path / "slanted.png"
    destination = tmp_path / "processed" / "slanted.jpg"
    image = Image.new("RGB", (800, 520), "white")
    draw = ImageDraw.Draw(image)
    for index in range(7):
        y = 100 + index * 55
        draw.line((80, y, 720, y), fill="black", width=3)
        draw.text((100, y + 10), f"Paciente {index}", fill="black")
    slanted = image.rotate(
        -6,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )
    slanted.save(source)

    before = _estimate_horizontal_skew(slanted)
    preprocess_image(source, destination, minimum_long_side=900)

    assert before is not None
    assert abs(before) >= 4.0
    with Image.open(destination) as processed:
        after = _estimate_horizontal_skew(processed.convert("RGB"))
    assert after is None or abs(after) < 3.0
