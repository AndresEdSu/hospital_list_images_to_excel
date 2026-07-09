from hospital_ocr.models import Place, Specialty
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.table_parser import looks_like_table
from tests.parsing_helpers import LEXICONS, line, table_line


def test_short_name_list_is_not_assumed_to_be_table() -> None:
    lines = [
        table_line("María Pérez", 100, 100, 270),
        table_line("Luis Gómez", 145, 100, 260),
        table_line("María Gómez", 190, 100, 270),
        table_line("Luis Pérez", 235, 100, 260),
    ]

    assert looks_like_table(lines) is False


def test_headerless_table_keeps_shifted_rows_from_repeated_sections() -> None:
    records = parse_ocr_lines(
        [
            table_line("Pediatria Piso 6", 20, 20, 300),
            table_line("Maria Perez 8", 90, 40, 260),
            table_line("Luis Gomez 9", 140, 40, 260),
            table_line("Ana Rivera 10", 190, 40, 275),
            table_line("Carla Medina 11", 240, 40, 290),
            table_line("Rosa Torres 12", 290, 40, 280),
            table_line("Pediatria Piso 6", 370, 90, 350),
            table_line("Luis Gomez 9", 440, 110, 330),
            table_line("Ana Rivera 10", 490, 110, 345),
            table_line("Rosa Torres 12", 540, 110, 350),
            table_line("Elena Vargas 13", 590, 110, 360),
        ],
        [Specialty("pediatria piso 6", "Pediatria", "Piso 6")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_secciones_repetidas.jpg",
        [],
    )

    assert [record.full_name for record in records] == [
        "Maria Perez",
        "Luis Gomez",
        "Ana Rivera",
        "Carla Medina",
        "Rosa Torres",
        "Luis Gomez",
        "Ana Rivera",
        "Rosa Torres",
        "Elena Vargas",
    ]
    assert [record.age for record in records] == [
        8,
        9,
        10,
        11,
        12,
        9,
        10,
        12,
        13,
    ]


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


def test_free_list_extracts_comma_document_and_multiple_origins() -> None:
    records = parse_ocr_lines(
        [
            line("Pediatria", 20),
            line("Maria Perez 10,711,859 60 anos F Caribe - La Guaira", 100),
        ],
        [Specialty("pediatria", "Pediatria", "")],
        LEXICONS,
        "Hospital de Prueba",
        "lista_procedencias_multiples.jpg",
        [
            Place("caribe", "Caribe"),
            Place("la guaira", "La Guaira"),
        ],
    )

    assert len(records) == 1
    assert records[0].document_id == "10711859"
    assert records[0].origin == "Caribe - La Guaira"


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
