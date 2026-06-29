from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.models import OcrLine, PatientRecord, Specialty
from hospital_ocr.name_splitter import NameLexicons
from hospital_ocr.parsing import detect_specialty, parse_ocr_lines


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


def test_parser_extracts_required_fields() -> None:
    specialties = [Specialty("pediatria uci", "Pediatría", "UCI")]
    records = parse_ocr_lines(
        [
            line("Pediatría - UCI", 20),
            line("María Pérez 8a F Petare", 100),
        ],
        specialties,
        LEXICONS,
        "Hospital de Prueba",
        "lista.jpg",
    )

    assert len(records) == 1
    patient = records[0]
    assert patient.first_name == "María"
    assert patient.last_name == "Pérez"
    assert patient.age == 8
    assert patient.age_unit == "años"
    assert patient.sex == "F"
    assert patient.origin == "Petare"
    assert patient.specialty == "Pediatría"
    assert patient.area == "UCI"
    assert patient.needs_review is False


def test_parser_marks_missing_values_for_review() -> None:
    specialties = [Specialty("trauma", "Traumatología", "")]
    records = parse_ocr_lines(
        [line("Trauma", 20), line("Luis Gómez 42", 100)],
        specialties,
        LEXICONS,
        "Hospital de Prueba",
        "lista.jpg",
    )

    assert len(records) == 1
    assert records[0].needs_review is True
    assert "Sexo no reconocido" in records[0].notes
    assert "Procedencia no reconocida" in records[0].notes


def test_specialty_fuzzy_match_ignores_short_alias_inside_names() -> None:
    specialties = [
        Specialty("quirofano", "Cirugía", "Quirófano"),
        Specialty("uci", "Cuidados intensivos", "UCI"),
    ]

    assert detect_specialty("Quiropano 26 106120", specialties) == (
        "Cirugía",
        "Quirófano",
    )
    assert detect_specialty("Lucía González 42", specialties) is None


def test_table_parser_extracts_columns_and_keeps_plan_as_observation() -> None:
    records = parse_ocr_lines(
        [
            table_line("Nombre y Apellido", 20, 100, 280),
            table_line("C.I.", 20, 300, 370),
            table_line("Edad", 20, 390, 440),
            table_line("Sexo", 20, 450, 500),
            table_line("Procedencia", 20, 570, 690),
            table_line("Plan", 20, 730, 820),
            table_line("1", 100, 20, 40),
            table_line("María Pérez", 100, 100, 270),
            table_line("12.345.678", 100, 290, 380),
            table_line("38", 100, 390, 430),
            table_line("F", 100, 450, 475),
            table_line("Petare", 100, 580, 660),
            table_line("Trauma", 100, 740, 830),
            table_line("2", 140, 20, 40),
            table_line("Luis Gómez", 140, 100, 260),
            table_line("M", 140, 450, 475),
            table_line("Guarenas", 140, 580, 680),
        ],
        [Specialty("trauma", "Traumatología", "")],
        LEXICONS,
        "Hospital de Prueba",
        "tabla.jpg",
    )

    assert len(records) == 2
    assert records[0].full_name == "María Pérez"
    assert records[0].document_id == "12345678"
    assert records[0].age == 38
    assert records[0].sex == "F"
    assert records[0].origin == "Petare"
    assert records[0].specialty == ""
    assert records[0].clinical_notes == "Plan: Trauma"
    assert records[1].age is None


def test_consolidation_merges_compatible_duplicates_and_keeps_evidence() -> None:
    first = record()
    second = record(
        source_image="otra.jpg",
        confidence=0.85,
    )

    result = consolidate_records([first, second])

    assert len(result.patients) == 1
    assert result.patients[0].occurrences == 2
    assert result.patients[0].source_images == ["lista.jpg", "otra.jpg"]
    assert result.patients[0].duplicate_status == "Duplicado consolidado"
    assert "2 apariciones consolidadas" in result.patients[0].duplicate_detail
    assert len(result.evidence) == 2
    assert result.evidence[0].patient_id == result.evidence[1].patient_id


def test_consolidation_uses_document_id_as_strong_identity() -> None:
    first = record(document_id="V-12.345.678")
    second = record(
        full_name="Maria Peres",
        age=9,
        document_id="12345678",
        source_image="otra.jpg",
    )

    result = consolidate_records([first, second])

    assert len(result.patients) == 1
    assert result.patients[0].occurrences == 2
    assert "misma cédula" in result.patients[0].notes_text


def test_different_document_ids_prevent_duplicate_match() -> None:
    first = record(document_id="11111111")
    second = record(document_id="22222222", source_image="otra.jpg")

    result = consolidate_records([first, second])

    assert len(result.patients) == 2
    assert all(item.duplicate_status == "Único" for item in result.patients)


def test_consolidation_does_not_merge_identity_conflicts() -> None:
    first = record()
    second = record(source_image="otra.jpg", origin="Guarenas")

    result = consolidate_records([first, second])

    assert len(result.patients) == 2
    assert all(patient.needs_review for patient in result.patients)
    assert all(
        "Posible duplicado no fusionado" in patient.notes
        for patient in result.patients
    )
    assert all(
        patient.duplicate_status == "Posible duplicado"
        for patient in result.patients
    )
    assert result.patients[0].patient_id in result.patients[1].duplicate_detail
    assert result.patients[1].patient_id in result.patients[0].duplicate_detail


def test_missing_age_requires_exact_name_to_be_duplicate_candidate() -> None:
    first = record(full_name="Mariela Fernández", age=None, age_unit="")
    second = record(
        full_name="Mariela Fernandes",
        age=None,
        age_unit="",
        source_image="otra.jpg",
    )

    result = consolidate_records([first, second])

    assert len(result.patients) == 2
    assert all(patient.duplicate_status == "Único" for patient in result.patients)
