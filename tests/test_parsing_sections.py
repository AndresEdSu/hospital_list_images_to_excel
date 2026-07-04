from hospital_ocr.models import GridBoundary, OcrLine, Place, Specialty, TableGrid
from hospital_ocr.parsing import parse_ocr_lines
from tests.parsing_helpers import LEXICONS, table_line


def test_headerless_sections_supply_context_and_parse_inline_fields() -> None:
    lines = [
        table_line("Pediatría Piso 6", 20, 40, 300),
        table_line("María Pérez", 100, 50, 280),
        table_line("1Be F", 100, 390, 500),
        table_line("La Guaira", 100, 620, 820),
        table_line("Luis Gómez", 150, 50, 280),
        table_line("loe H", 150, 390, 500),
        table_line("La Guaira", 150, 620, 820),
        table_line("María Gómez", 200, 50, 280),
        table_line("Ga (F)", 200, 390, 500),
        table_line("La Guaira", 200, 620, 820),
        table_line("Luis Pérez", 400, 50, 280),
        table_line("54e M", 400, 390, 500),
        table_line("La Guaira", 400, 620, 820),
        table_line("María Gómez", 450, 50, 280),
        table_line("27e F", 450, 390, 500),
        table_line("La Guaira", 450, 620, 820),
        table_line("Pediatría", 570, 40, 300),
        table_line("Piso1", 570, 430, 530),
        table_line("Emergencia", 592, 430, 600),
        table_line("Luis Pérez", 650, 50, 280),
        table_line("lla (M)", 650, 390, 500),
        table_line("La Guaira", 650, 620, 820),
        table_line("María Pérez", 700, 50, 280),
        table_line("Ize F", 700, 390, 500),
        table_line("La Guaira", 700, 620, 820),
        table_line("Luis Gómez", 750, 50, 280),
        table_line("s3e M", 750, 390, 500),
        table_line("La Guaira", 750, 620, 820),
    ]

    records = parse_ocr_lines(
        lines,
        [
            Specialty("pediatria", "Pediatría", ""),
            Specialty("pediatria uci", "Pediatría", "UCI"),
        ],
        LEXICONS,
        "Hospital de Prueba",
        "secciones_sin_encabezado.jpg",
        [Place("la guaira", "La Guaira")],
    )

    assert len(records) == 8
    assert [record.age for record in records] == [
        13,
        10,
        6,
        54,
        27,
        11,
        12,
        53,
    ]
    assert [record.sex for record in records] == [
        "F",
        "M",
        "F",
        "M",
        "F",
        "M",
        "F",
        "M",
    ]
    assert [record.specialty for record in records] == [
        "Pediatría",
        "Pediatría",
        "Pediatría",
        "",
        "",
        "Pediatría",
        "Pediatría",
        "Pediatría",
    ]
    assert [record.area for record in records[:3]] == ["Piso 6"] * 3
    assert records[3].area == ""
    assert records[4].area == ""
    assert [record.area for record in records[5:]] == [
        "Piso 1 - Emergencia"
    ] * 3
    assert "1Be F" in records[0].raw_line
    assert all(
        "encabezado de sección" in record.field_evidence["especialidad"]
        for record in [*records[:3], *records[5:]]
    )


def test_grid_specialty_cells_apply_only_to_their_patient_row() -> None:
    grid = TableGrid(
        horizontal=tuple(
            GridBoundary(0, position, 1)
            for position in range(0, 351, 50)
        ),
        vertical=(
            GridBoundary(0, 0, 1),
            GridBoundary(0, 300, 1),
            GridBoundary(0, 450, 1),
            GridBoundary(0, 550, 1),
            GridBoundary(0, 850, 1),
        ),
        confidence=1,
    )
    names = [
        "María Pérez",
        "Luis Gómez",
        "María Gómez",
        "Luis Pérez",
        "María Pérez",
        "Luis Gómez",
    ]
    lines: list[OcrLine] = []
    for index, name in enumerate(names):
        y = 10 + index * 50
        lines.extend(
            [
                table_line(name, y, 40, 260),
                table_line(str(30 + index), y, 330, 410),
                table_line("F" if index % 2 == 0 else "M", y, 480, 520),
            ]
        )
    lines.extend(
        [
            table_line("Pediatría", 10, 600, 760),
            table_line("M.I.", 110, 600, 700),
        ]
    )

    records = parse_ocr_lines(
        lines,
        [
            Specialty("pediatria", "Pediatría", ""),
            Specialty("mi", "Medicina interna", ""),
        ],
        LEXICONS,
        "Hospital de Prueba",
        "especialidad_por_fila.jpg",
        [],
        grid,
    )

    assert len(records) == 6
    assert [record.specialty for record in records] == [
        "Pediatría",
        "",
        "Medicina interna",
        "",
        "",
        "",
    ]
