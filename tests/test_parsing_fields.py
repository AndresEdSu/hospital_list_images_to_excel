from hospital_ocr.matching import match_place, match_places
from hospital_ocr.models import Place, Specialty
from hospital_ocr.parsing import detect_specialty, parse_ocr_lines
from tests.parsing_helpers import LEXICONS, line, table_line


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


def test_internal_medicine_initials_require_the_whole_cell() -> None:
    specialties = [Specialty("mi", "Medicina interna", "")]

    assert detect_specialty("M.I.", specialties) == ("Medicina interna", "")
    assert detect_specialty("MI", specialties) == ("Medicina interna", "")
    assert detect_specialty("Traslado a mi cuidado", specialties) is None


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


def test_place_matches_extracts_multiple_catalog_origins() -> None:
    places = [
        Place("caribe", "Caribe"),
        Place("la guaira", "La Guaira"),
        Place("caraballeda", "Caraballeda"),
    ]

    hyphen_matches = match_places("Caribe - La Guaira", places)
    comma_matches = match_places("Caribe, Caraballeda", places)
    adjacent_matches = match_places("Caraballeda Caribe", places)

    assert [match.name for match in hyphen_matches] == [
        "Caribe",
        "La Guaira",
    ]
    assert [match.name for match in comma_matches] == [
        "Caribe",
        "Caraballeda",
    ]
    assert [match.name for match in adjacent_matches] == [
        "Caraballeda",
        "Caribe",
    ]


def test_contextual_place_match_requires_a_clear_catalog_winner() -> None:
    places = [
        Place("guaira", "La Guaira"),
        Place("guatire", "Guatire"),
    ]

    assert match_place("Guaina", places) is None
    contextual = match_place("Guaina", places, contextual=True)

    assert contextual is not None
    assert contextual.name == "La Guaira"
    assert contextual.contextual
    assert contextual.score - contextual.runner_up_score >= 0.06

    ambiguous = match_place(
        "Guaixa",
        [Place("guaira", "La Guaira"), Place("guaina", "Otro lugar")],
        contextual=True,
    )
    assert ambiguous is None


def test_specialty_match_tolerates_removed_space() -> None:
    specialties = [
        Specialty("medicina", "Medicina general", ""),
        Specialty("medicina interna", "Medicina interna", ""),
    ]

    assert detect_specialty("Medicinainterna", specialties) == (
        "Medicina interna",
        "",
    )


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


def test_explicit_origin_column_uses_contextual_catalog_match() -> None:
    records = parse_ocr_lines(
        [
            table_line("Paciente", 20, 100, 300),
            table_line("Procedencia", 20, 600, 850),
            table_line("María Pérez", 100, 110, 290),
            table_line("Guaina", 100, 620, 800),
            table_line("Luis Gómez", 145, 110, 280),
            table_line("Guatire", 145, 620, 800),
        ],
        [],
        LEXICONS,
        "Hospital de Prueba",
        "procedencia_contextual.jpg",
        [Place("guaira", "La Guaira"), Place("guatire", "Guatire")],
    )

    assert len(records) == 2
    assert records[0].origin == "La Guaira"
    assert "Guaina" in records[0].raw_line
    assert (
        "Procedencia normalizada por coincidencia contextual"
        in records[0].notes
    )
    assert "contextual" in records[0].field_evidence["procedencia"]
    assert records[1].origin == "Guatire"


def test_explicit_sex_column_normalizes_confusions_and_rejects_conflict() -> None:
    records = parse_ocr_lines(
        [
            table_line("Paciente", 20, 100, 300),
            table_line("Sexo", 20, 450, 520),
            table_line("María Pérez", 100, 110, 290),
            table_line("T", 100, 460, 500),
            table_line("Luis Gómez", 145, 110, 280),
            table_line("N", 145, 460, 500),
            table_line("María Gómez", 190, 110, 290),
            table_line("F", 190, 455, 480),
            table_line("M", 190, 485, 510),
        ],
        [],
        LEXICONS,
        "Hospital de Prueba",
        "sexo_contextual.jpg",
        [],
    )

    assert len(records) == 3
    assert records[0].sex == "F"
    assert "T" in records[0].raw_line
    assert "Sexo normalizado desde OCR: T" in records[0].notes
    assert "normalizado desde T" in records[0].field_evidence["sexo"]
    assert records[1].sex == "M"
    assert "Sexo normalizado desde OCR: N" in records[1].notes
    assert records[2].sex == ""
    assert "Sexo ambiguo entre valores incompatibles: F/M" in records[2].notes


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
