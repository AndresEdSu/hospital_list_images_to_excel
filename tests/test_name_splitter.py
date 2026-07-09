from hospital_ocr.name_splitter import (
    NameLexicons,
    normalize_identity_text,
    split_full_name,
)


LEXICONS = NameLexicons(
    given_names={
        "jose": 1.0,
        "maria": 1.0,
        "margarita": 1.0,
        "luis": 1.0,
        "sebastian": 1.0,
        "valentina": 1.0,
    },
    surnames={
        "andrade": 1.0,
        "garcia": 1.0,
        "martinez": 1.0,
        "morales": 1.0,
        "perez": 1.0,
        "rodriguez": 1.0,
    },
)


def test_names_first_order_is_detected() -> None:
    result = split_full_name("Maria Jose Perez Garcia", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Maria Jose"
    assert result.last_name == "Perez Garcia"
    assert result.detected_order == "Nombre-Apellido"


def test_surnames_first_order_is_detected() -> None:
    result = split_full_name("Perez Garcia Maria Jose", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Maria Jose"
    assert result.last_name == "Perez Garcia"
    assert result.detected_order == "Apellido-Nombre"


def test_unknown_tokens_are_left_indeterminate() -> None:
    result = split_full_name("Xyz Abcd", LEXICONS)

    assert result.reliable is False
    assert result.first_name == ""
    assert result.last_name == ""
    assert result.detected_order == "Indeterminado"


def test_one_known_surname_can_determine_two_token_order() -> None:
    result = split_full_name("Lunara Perez", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Lunara"
    assert result.last_name == "Perez"
    assert result.confidence == 0.85


def test_small_ocr_typo_in_name_uses_fuzzy_catalog_match() -> None:
    result = split_full_name("Marria Perez", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Marria"
    assert result.last_name == "Perez"


def test_glued_given_names_are_split_with_name_catalog() -> None:
    assert (
        normalize_identity_text(
            "DAMARAVALENTINA YURIMARGARITA",
            LEXICONS,
            role="given",
        )
        == "DAMARA VALENTINA YURI MARGARITA"
    )


def test_glued_given_name_uses_small_ocr_confusion_inside_compound() -> None:
    assert (
        normalize_identity_text("SEBASTIANIOSE", LEXICONS, role="given")
        == "SEBASTIAN JOSE"
    )


def test_glued_surnames_are_split_with_surname_catalog() -> None:
    assert (
        normalize_identity_text(
            "RODRIGUEZBELIZARIO PULIDOMORALES",
            LEXICONS,
            role="surname",
        )
        == "RODRIGUEZ BELIZARIO PULIDO MORALES"
    )


def test_glued_surname_connector_is_split() -> None:
    assert (
        normalize_identity_text("ANDRADEDEVIEIRA", LEXICONS, role="surname")
        == "ANDRADE DE VIEIRA"
    )


def test_known_single_term_is_not_split() -> None:
    assert (
        normalize_identity_text("MARTINEZ", LEXICONS, role="surname")
        == "MARTINEZ"
    )


def test_short_known_suffix_does_not_split_unknown_given_name() -> None:
    assert (
        normalize_identity_text("TATIANA", LEXICONS, role="given")
        == "TATIANA"
    )
