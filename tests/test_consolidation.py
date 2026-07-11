from hospital_ocr.consolidation import consolidate_records
from hospital_ocr.name_splitter import NameLexicons
from tests.parsing_helpers import record


def _assert_unique(records) -> None:
    assert all(
        patient.duplicate_status != "Posible duplicado"
        for patient in records
    )
    assert all(not patient.duplicate_detail for patient in records)


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


def test_consolidation_preserves_first_appearance_order() -> None:
    first = record(
        full_name="Zuleima Rojas",
        age=30,
        confidence=0.40,
        source_image="primera.jpg",
    )
    second = record(
        full_name="Ana Torres",
        age=42,
        confidence=0.80,
        source_image="segunda.jpg",
    )
    duplicate = record(
        full_name="Zuleima Rojas",
        age=30,
        confidence=0.99,
        source_image="tercera.jpg",
    )

    result = consolidate_records([first, second, duplicate])

    assert [patient.full_name for patient in result.patients] == [
        "Zuleima Rojas",
        "Ana Torres",
    ]
    assert result.patients[0].source_images == [
        "primera.jpg",
        "tercera.jpg",
    ]
    assert [
        evidence.record.source_image for evidence in result.evidence
    ] == ["primera.jpg", "segunda.jpg", "tercera.jpg"]


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
    assert "misma" in result.patients[0].notes_text


def test_different_document_ids_prevent_duplicate_match() -> None:
    first = record(document_id="11111111")
    second = record(document_id="22222222", source_image="otra.jpg")

    result = consolidate_records([first, second])

    assert len(result.patients) == 2
    _assert_unique(result.patients)


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


def test_ocr_name_correction_marks_possible_duplicate_without_merging() -> None:
    lexicons = NameLexicons(
        given_names={"samuel": 1.0},
        surnames={"vega": 1.0},
    )
    first = record(
        full_name="Samuel Vega",
        first_name="Samuel",
        last_name="Vega",
        age=11,
    )
    second = record(
        full_name="Samwel Vega",
        first_name="Samwel",
        last_name="Vega",
        age=11,
        source_image="otra.jpg",
    )

    result = consolidate_records(
        [first, second],
        name_lexicons=lexicons,
    )

    assert len(result.patients) == 2
    assert all(patient.occurrences == 1 for patient in result.patients)
    assert all(
        patient.duplicate_status == "Posible duplicado"
        for patient in result.patients
    )
    assert all(
        "nombre normalizado" in patient.duplicate_detail
        for patient in result.patients
    )


def test_ocr_name_correction_does_not_mark_missing_age_candidates() -> None:
    lexicons = NameLexicons(
        given_names={"samuel": 1.0},
        surnames={"vega": 1.0},
    )
    first = record(
        full_name="Samuel Vega",
        first_name="Samuel",
        last_name="Vega",
        age=None,
        age_unit="",
    )
    second = record(
        full_name="Samwel Vega",
        first_name="Samwel",
        last_name="Vega",
        age=None,
        age_unit="",
        source_image="otra.jpg",
    )

    result = consolidate_records(
        [first, second],
        name_lexicons=lexicons,
    )

    assert len(result.patients) == 2
    _assert_unique(result.patients)


def test_ocr_name_correction_uses_surname_catalog_too() -> None:
    lexicons = NameLexicons(
        given_names={"samuel": 1.0},
        surnames={"silva": 1.0},
    )
    first = record(
        full_name="Samuel Silva",
        first_name="Samuel",
        last_name="Silva",
        age=11,
    )
    second = record(
        full_name="Samuel Silua",
        first_name="Samuel",
        last_name="Silua",
        age=11,
        source_image="otra.jpg",
    )

    result = consolidate_records(
        [first, second],
        name_lexicons=lexicons,
    )

    assert len(result.patients) == 2
    assert all(
        patient.duplicate_status == "Posible duplicado"
        for patient in result.patients
    )
    assert all(
        "nombre normalizado" in patient.duplicate_detail
        for patient in result.patients
    )


def test_missing_age_requires_exact_name_to_be_duplicate_candidate() -> None:
    first = record(full_name="Mariela Fernandez", age=None, age_unit="")
    second = record(
        full_name="Mariela Fernandes",
        age=None,
        age_unit="",
        source_image="otra.jpg",
    )

    result = consolidate_records([first, second])

    assert len(result.patients) == 2
    _assert_unique(result.patients)
