from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.matching import match_place
from hospital_ocr.models import (
    GridBoundary,
    OcrLine,
    PatientRecord,
    Place,
    Specialty,
    TableGrid,
)
from hospital_ocr.name_splitter import NameLexicons
from hospital_ocr.parsing import detect_specialty, parse_ocr_lines
from hospital_ocr.table_parser import looks_like_table


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
        [Place("petare", "Petare")],
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


def test_place_match_tolerates_ocr_space_changes_and_prefers_specific_alias() -> None:
    places = [
        Place("catia", "Catia"),
        Place("catia la mar", "Catia La Mar"),
    ]

    joined = match_place("catia lamar", places)
    split = match_place("cati a la mar", places)
    short = match_place("catia", places)

    assert joined is not None and joined.name == "Catia La Mar"
    assert split is not None and split.name == "Catia La Mar"
    assert short is not None and short.name == "Catia"


def test_specialty_match_tolerates_removed_space() -> None:
    specialties = [
        Specialty("medicina", "Medicina general", ""),
        Specialty("medicina interna", "Medicina interna", ""),
    ]

    assert detect_specialty("Medicinainterna", specialties) == (
        "Medicina interna",
        "",
    )


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
    assert "índice descartado" in records[0].field_evidence["nombre"]
    assert records[1].age is None


def test_table_parser_uses_headers_when_columns_are_reordered() -> None:
    records = parse_ocr_lines(
        [
            table_line("Procedencia", 20, 70, 190),
            table_line("Sexo", 20, 230, 290),
            table_line("Paciente", 20, 340, 500),
            table_line("Edad", 20, 570, 630),
            table_line("C.I.", 20, 690, 760),
            table_line("Servicio", 20, 820, 930),
            table_line("Petare", 100, 80, 170),
            table_line("F", 100, 245, 270),
            table_line("María Pérez", 100, 350, 500),
            table_line("38", 100, 580, 620),
            table_line("12.345.678", 100, 680, 770),
            table_line("Pediatría", 100, 825, 925),
            table_line("Guarenas", 145, 75, 185),
            table_line("M", 145, 245, 270),
            table_line("Luis Gómez", 145, 350, 490),
            table_line("42", 145, 580, 620),
            table_line("9.876.543", 145, 690, 765),
            table_line("Trauma", 145, 835, 910),
        ],
        [
            Specialty("pediatria", "Pediatría", ""),
            Specialty("trauma", "Traumatología", ""),
        ],
        LEXICONS,
        "Hospital de Prueba",
        "columnas_reordenadas.jpg",
        [Place("petare", "Petare"), Place("guarenas", "Guarenas")],
    )

    assert [record.full_name for record in records] == [
        "María Pérez",
        "Luis Gómez",
    ]
    assert records[0].document_id == "12345678"
    assert records[0].age == 38
    assert records[0].sex == "F"
    assert records[0].origin == "Petare"
    assert records[0].specialty == "Pediatría"
    assert records[0].document_confidence > 0.8
    assert "encabezado" in records[0].field_evidence["cédula"]
    assert records[1].specialty == "Traumatología"


def test_ignored_bed_column_limits_origin_without_exporting_bed_values() -> None:
    records = parse_ocr_lines(
        [
            table_line("Procedencia", 20, 850, 980),
            table_line("Cama", 20, 750, 820),
            table_line("Servicio", 25, 580, 700),
            table_line("Diagnóstico", 35, 500, 570),
            table_line("Afiliación", 45, 400, 480),
            table_line("Edad", 55, 300, 360),
            table_line("Paciente", 75, 80, 250),
            table_line("María Pérez", 140, 90, 250),
            table_line("38", 140, 310, 350),
            table_line("Trauma", 140, 590, 680),
            table_line("620-A", 140, 760, 815),
            table_line("Luis Gómez", 185, 90, 240),
            table_line("42", 185, 310, 350),
            table_line("Trauma", 185, 590, 680),
            table_line("706", 185, 765, 810),
            table_line("UTIA", 185, 870, 930),
        ],
        [Specialty("trauma", "Traumatología", "")],
        LEXICONS,
        "Hospital de Prueba",
        "cama_intermedia.jpg",
    )

    assert len(records) == 2
    assert records[0].origin == ""
    assert records[1].origin == "UTIA"
    assert "620-A" not in records[0].observations_text
    assert "706" not in records[1].observations_text


def test_unknown_intermediate_header_becomes_neutral_column() -> None:
    records = parse_ocr_lines(
        [
            table_line("Procedencia", 20, 850, 980),
            table_line("Código interno", 30, 720, 840),
            table_line("Servicio", 35, 580, 700),
            table_line("Edad", 55, 300, 360),
            table_line("Paciente", 75, 80, 250),
            table_line("María Pérez", 140, 90, 250),
            table_line("38", 140, 310, 350),
            table_line("Trauma", 140, 590, 680),
            table_line("ZX-91", 140, 750, 820),
            table_line("Luis Gómez", 185, 90, 240),
            table_line("42", 185, 310, 350),
            table_line("Trauma", 185, 590, 680),
            table_line("AB-77", 185, 750, 820),
            table_line("Petare", 185, 870, 940),
        ],
        [Specialty("trauma", "Traumatología", "")],
        LEXICONS,
        "Hospital de Prueba",
        "columna_desconocida.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 2
    assert records[0].origin == ""
    assert records[1].origin == "Petare"
    assert "ZX-91" not in records[0].observations_text
    assert "AB-77" not in records[1].observations_text


def test_headerless_table_without_indexes_age_or_sex_is_detected() -> None:
    lines = [
        table_line("María Pérez", 100, 100, 270),
        table_line("Luis Gómez", 145, 100, 260),
        table_line("María Gómez", 190, 100, 270),
        table_line("Luis Pérez", 235, 100, 260),
        table_line("María Pérez", 280, 100, 270),
        table_line("Luis Gómez", 325, 100, 260),
    ]

    assert looks_like_table(lines) is True

    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "lista_sin_encabezados.jpg",
    )

    assert len(records) == 6
    assert all(record.age is None for record in records)
    assert all(record.sex == "" for record in records)


def test_headerless_reordered_columns_are_classified_by_content() -> None:
    names = [
        "María Pérez",
        "Luis Gómez",
        "Ana Rivera",
        "José Torres",
        "Carla Medina",
        "Pedro Rojas",
    ]
    ages = [30, 42, 55, 28, 63, 47]
    lines: list[OcrLine] = []
    for index, name in enumerate(names):
        y = 100 + index * 45
        lines.extend(
            [
                table_line(f"12.345.{670 + index}", y, 80, 190),
                table_line(str(ages[index]), y, 280, 330),
                table_line("F" if index % 2 == 0 else "M", y, 390, 420),
                table_line(name, y, 520, 690),
                table_line("S1 GNB", y, 700, 760),
                table_line("Trauma", y, 780, 860),
                table_line(
                    "Petare" if index % 2 == 0 else "Guarenas",
                    y,
                    890,
                    980,
                ),
            ]
        )

    records = parse_ocr_lines(
        lines,
        [Specialty("trauma", "Traumatología", "")],
        LEXICONS,
        "Hospital de Prueba",
        "columnas_reordenadas_sin_encabezado.jpg",
        [Place("petare", "Petare"), Place("guarenas", "Guarenas")],
    )

    assert len(records) == 6
    assert records[0].full_name == "María Pérez"
    assert records[0].document_id == "12345670"
    assert records[0].age == 30
    assert records[0].sex == "F"
    assert records[0].origin == "Petare"
    assert records[0].specialty == "Traumatología"
    assert records[1].full_name == "Luis Gómez"
    assert records[1].origin == "Guarenas"


def test_slanted_grid_keeps_distant_cells_in_the_same_row() -> None:
    horizontal = tuple(
        GridBoundary(0.05, 50 + index * 50, 1.0)
        for index in range(7)
    )
    vertical = tuple(
        GridBoundary(0.0, position, 1.0)
        for position in (50, 350, 700, 1000)
    )
    grid = TableGrid(horizontal, vertical, 1.0)
    names = [
        "María Pérez",
        "Luis Gómez",
        "Ana Rivera",
        "José Torres",
        "Carla Medina",
    ]
    lines: list[OcrLine] = []
    for index, name in enumerate(names):
        name_y = 70 + index * 50
        lines.append(table_line(name, name_y, 100, 260))
        if index == 1:
            lines.append(table_line("Petare", name_y + 40, 850, 940))

    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "tabla_inclinada.jpg",
        [Place("petare", "Petare")],
        grid,
    )

    assert len(records) == 5
    assert records[0].origin == ""
    assert records[1].full_name == "Luis Gómez"
    assert records[1].origin == "Petare"
    assert records[2].origin == ""


def test_short_name_list_is_not_assumed_to_be_table() -> None:
    lines = [
        table_line("María Pérez", 100, 100, 270),
        table_line("Luis Gómez", 145, 100, 260),
        table_line("María Gómez", 190, 100, 270),
        table_line("Luis Pérez", 235, 100, 260),
    ]

    assert looks_like_table(lines) is False


def test_free_list_extracts_document_and_fuzzy_place() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría UCI", 20),
            line("María Pérez V-12.345.678 8 años F Petarre", 100),
        ],
        [Specialty("pediatria uci", "Pediatría", "UCI")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_libre.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 1
    assert records[0].full_name == "María Pérez"
    assert records[0].document_id == "V-12345678"
    assert records[0].age == 8
    assert records[0].sex == "F"
    assert records[0].origin == "Petare"
    assert records[0].specialty == "Pediatría"
    assert records[0].document_confidence > 0.8
    assert records[0].origin_confidence > 0.8


def test_inline_list_uses_name_plus_any_field_and_keeps_incomplete_rows() -> None:
    records = parse_ocr_lines(
        [
            line("27Jun", 10),
            line("Emergencia Pediátrica", 35),
            line("Hospital de Prueba", 60),
            line("1. María Pérez, 8 años, Petare", 120),
            line("2. Luis Gómez, Guarenas", 165),
            line("3. Ana Rivera, F", 210),
            line("4. Carla Medina", 255),
        ],
        [Specialty("emergencia pediatrica", "Pediatría", "Emergencia")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_en_linea.jpg",
        [Place("petare", "Petare"), Place("guarenas", "Guarenas")],
    )

    assert [record.full_name for record in records] == [
        "María Pérez",
        "Luis Gómez",
        "Ana Rivera",
        "Carla Medina",
    ]
    assert records[0].age == 8
    assert records[0].origin == "Petare"
    assert records[1].age is None
    assert records[1].origin == "Guarenas"
    assert records[2].sex == "F"
    assert records[3].age is None


def test_inline_handwritten_row_accepts_document_when_age_is_missing() -> None:
    records = parse_ocr_lines(
        [
            line("1 Duarte Andres es 33432291 texto OCR adicional", 100),
            line("2 Maria Perez 12345678 ruido manuscrito adicional largo", 145),
            line("3 Luis Gomez 87654321 otro fragmento manuscrito largo", 190),
        ],
        [],
        LEXICONS,
        "Hospital de Prueba",
        "lista_manuscrita.jpg",
        [],
    )

    assert len(records) == 3
    assert records[1].document_id == "12345678"
    assert records[1].age is None


def test_free_list_does_not_treat_place_surname_as_origin() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría", 20),
            line("María Valencia 8 años F", 100),
        ],
        [Specialty("pediatria", "Pediatría", "")],
        LEXICONS,
        "Hospital de Prueba",
        "apellido_geografico.jpg",
        [Place("valencia", "Valencia")],
    )

    assert len(records) == 1
    assert records[0].full_name == "María Valencia"
    assert records[0].origin == ""


def test_free_list_discards_index_before_extracting_name_and_age() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría", 20),
            line("17. María Pérez 38 años F Petare", 100),
        ],
        [Specialty("pediatria", "Pediatría", "")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_numerada.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 1
    assert records[0].full_name == "María Pérez"
    assert records[0].age == 38
    assert records[0].origin == "Petare"
    assert "índice descartado" in records[0].field_evidence["nombre"]


def test_free_list_does_not_use_lonely_index_as_age() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría", 20),
            line("18) María Pérez F Petare", 100),
        ],
        [Specialty("pediatria", "Pediatría", "")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_sin_edad.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 1
    assert records[0].full_name == "María Pérez"
    assert records[0].age is None
    assert records[0].sex == "F"


def test_free_list_requires_catalog_match_for_origin() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría", 20),
            line("María Pérez 8 años F Texto Libre", 100),
        ],
        [Specialty("pediatria", "Pediatría", "")],
        LEXICONS,
        "Hospital de Prueba",
        "procedencia_incierta.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 1
    assert records[0].origin == ""
    assert "Procedencia no reconocida" in records[0].notes


def test_explicit_origin_column_preserves_unknown_value_for_review() -> None:
    records = parse_ocr_lines(
        [
            table_line("Paciente", 20, 100, 300),
            table_line("Edad", 20, 420, 500),
            table_line("Procedencia", 20, 650, 850),
            table_line("María Pérez", 100, 110, 290),
            table_line("38", 100, 430, 480),
            table_line("Sector no catalogado", 100, 660, 840),
            table_line("Luis Gómez", 145, 110, 280),
            table_line("42", 145, 430, 480),
            table_line("Otra localidad", 145, 660, 830),
        ],
        [],
        LEXICONS,
        "Hospital de Prueba",
        "procedencia_explicita.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 2
    assert records[0].origin == "Sector no catalogado"
    assert "Procedencia no validada en catálogo" in records[0].notes


def test_age_at_start_is_allowed_when_unit_is_explicit() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatría", 20),
            line("38 años María Pérez F Petare", 100),
        ],
        [Specialty("pediatria", "Pediatría", "")],
        LEXICONS,
        "Hospital de Prueba",
        "edad_inicial.jpg",
        [Place("petare", "Petare")],
    )

    assert len(records) == 1
    assert records[0].full_name == "María Pérez"
    assert records[0].age == 38


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
