from hospital_ocr.models import GridBoundary, OcrLine, Place, Specialty, TableGrid
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.table_parser import looks_like_table
from tests.parsing_helpers import LEXICONS, table_line


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


def test_partial_vertical_grid_keeps_names_outside_detected_columns() -> None:
    grid = TableGrid(
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (80, 130, 180, 230, 280, 330, 380)
        ),
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (350, 500, 620, 700, 850, 1000)
        ),
        0.92,
    )
    names = [
        "Monica Vielma",
        "Elizabeth Delgado",
        "Yaritza Garcia",
        "Vanessa Gonzalez",
        "Grismele Navarro",
        "Josefina Rojas",
    ]
    origins = ["Guaira", "Caracas", "Guaira", "Caracas", "Guaira", "Caracas"]
    lines: list[OcrLine] = []
    for index, (name, origin) in enumerate(zip(names, origins, strict=True)):
        y = 90 + index * 50
        lines.extend(
            [
                table_line(name, y, 100, 380),
                table_line(f"12345{index + 10}", y, 390, 480),
                table_line(origin, y, 720, 820),
            ]
        )

    places = [Place("guaira", "Guaira"), Place("caracas", "Caracas")]
    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "cuadricula_vertical_parcial.jpg",
        places,
        grid,
    )

    assert [record.full_name for record in records] == names
    assert [record.origin for record in records] == origins


def test_headerless_partial_grid_recovers_documents_and_ages() -> None:
    grid = TableGrid(
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (80, 130, 180, 230, 280, 330, 380)
        ),
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (350, 500, 620, 700, 850, 1000)
        ),
        0.92,
    )
    names = [
        "Monica Vielma",
        "Elizabeth Delgado",
        "Yaritza Garcia",
        "Vanessa Gonzalez",
        "Grismele Navarro",
        "Josefina Rojas",
    ]
    documents = [
        "18404003",
        "32446991",
        "12950267",
        "15164293",
        "14953283",
        "26590820",
    ]
    ages = [38, 20, 53, 24, 43, 26]
    noisy_indexes = ["32", "34", "15", "18", "9", "60"]
    lines: list[OcrLine] = []
    for index, (name, document, age, row_index) in enumerate(
        zip(names, documents, ages, noisy_indexes, strict=True)
    ):
        y = 90 + index * 50
        lines.append(table_line(row_index, y, 5, 50))
        if index < 3:
            lines.extend(
                [
                    table_line(f"{name} {document}", y, 100, 580),
                    table_line(str(age), y, 630, 670),
                ]
            )
        else:
            lines.extend(
                [
                    table_line(name, y, 100, 330),
                    table_line(f"{document} {age}", y, 390, 670),
                ]
            )
        lines.append(table_line("Guaira", y, 720, 820))

    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "campos_fusionados_sin_encabezado.jpg",
        [Place("guaira", "Guaira")],
        grid,
    )

    assert [record.full_name for record in records] == names
    assert [record.document_id for record in records] == documents
    assert [record.age for record in records] == ages


def test_cropped_top_table_keeps_first_partial_row() -> None:
    grid = TableGrid(
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (40, 80, 120, 160, 200, 240)
        ),
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (50, 350, 700, 1000)
        ),
        0.92,
    )
    names = [
        "Monica Vielma",
        "Elizabeth Delgado",
        "Yaritza Garcia",
        "Vanessa Gonzalez",
        "Grismele Navarro",
        "Josefina Rojas",
    ]
    documents = [
        "18404003",
        "32446991",
        "12950267",
        "15164293",
        "14953283",
        "26590820",
    ]
    ages = [38, 20, 53, 24, 43, 26]
    lines: list[OcrLine] = []
    for index, (name, document, age) in enumerate(
        zip(names, documents, ages, strict=True)
    ):
        y = 2 if index == 0 else 50 + (index - 1) * 40
        lines.extend(
            [
                table_line(name, y, 100, 320),
                table_line(f"{document} {age}", y, 400, 650),
                table_line("Guaira", y, 750, 900),
            ]
        )

    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "tabla_recortada_arriba.jpg",
        [Place("guaira", "Guaira")],
        grid,
    )

    assert [record.full_name for record in records] == names
    assert records[0].document_id == "18404003"
    assert records[0].age == 38
    assert records[0].origin == "Guaira"


def test_partial_grid_header_infers_missing_document_age_and_sex() -> None:
    grid = TableGrid(
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (0, 40, 80, 120, 160)
        ),
        tuple(
            GridBoundary(0.0, position, 1.0)
            for position in (0, 100, 400, 550, 650, 750, 1000)
        ),
        1.0,
    )
    lines = [
        table_line("Nombre y Apellido", 20, 120, 370),
        table_line("Procedencia", 20, 780, 950),
    ]
    for index, (name, document, age, sex) in enumerate(
        [
            ("MarÃ­a PÃ©rez", "12345678", "14", "F"),
            ("Luis GÃ³mez", "87654321", "20", "M"),
            ("Ana Rivera", "11223344", "15", "F"),
        ]
    ):
        y = 35 + index * 40
        lines.extend(
            [
                table_line(name, y, 120, 370),
                table_line(document, y, 420, 530),
                table_line(age, y, 570, 620),
                table_line(sex, y, 680, 720),
                table_line("Petare", y, 780, 930),
            ]
        )

    records = parse_ocr_lines(
        lines,
        [],
        LEXICONS,
        "Hospital de Prueba",
        "encabezado_parcial.jpg",
        [Place("petare", "Petare")],
        grid,
    )

    assert len(records) == 3
    assert records[0].document_id == "12345678"
    assert records[0].age == 14
    assert records[0].sex == "F"
    assert records[0].origin == "Petare"
