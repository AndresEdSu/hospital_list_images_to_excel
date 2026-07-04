from hospital_ocr.models import OcrLine, PatientRecord
from hospital_ocr.name_splitter import NameLexicons


LEXICONS = NameLexicons(
    given_names={"maria": 1.0, "luis": 1.0},
    surnames={"perez": 1.0, "gomez": 1.0},
)


def line(text: str, y: int, x: int = 50, score: float = 0.95) -> OcrLine:
    return OcrLine(
        text=text,
        score=score,
        box=(x, y, x + 500, y + 40),
        image_width=1000,
        image_height=1200,
    )


def table_line(
    text: str,
    y: int,
    x1: int,
    x2: int,
    score: float = 0.95,
) -> OcrLine:
    return OcrLine(
        text=text,
        score=score,
        box=(x1, y, x2, y + 28),
        image_width=1000,
        image_height=1200,
    )


def record(**overrides: object) -> PatientRecord:
    values = {
        "full_name": "María Pérez",
        "first_name": "María",
        "last_name": "Pérez",
        "name_split_confidence": 1.0,
        "detected_name_order": "Nombre-Apellido",
        "center": "Hospital de Prueba",
        "age": 8,
        "age_unit": "años",
        "sex": "F",
        "origin": "Petare",
        "specialty": "Pediatría",
        "area": "UCI",
        "source_image": "lista.jpg",
        "confidence": 0.95,
        "needs_review": False,
        "raw_line": "María Pérez 8a F Petare",
    }
    values.update(overrides)
    return PatientRecord(**values)
