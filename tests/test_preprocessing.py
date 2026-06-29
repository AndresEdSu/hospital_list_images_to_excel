from pathlib import Path

from PIL import Image, ImageDraw

from hospital_ocr.preprocessing import preprocess_image


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
