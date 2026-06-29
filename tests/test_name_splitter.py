from hospital_ocr.name_splitter import NameLexicons, split_full_name


LEXICONS = NameLexicons(
    given_names={
        "jose": 1.0,
        "maria": 1.0,
        "luis": 1.0,
    },
    surnames={
        "garcia": 1.0,
        "perez": 1.0,
        "rodriguez": 1.0,
    },
)


def test_names_first_order_is_detected() -> None:
    result = split_full_name("María José Pérez García", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "María José"
    assert result.last_name == "Pérez García"
    assert result.detected_order == "Nombre-Apellido"


def test_surnames_first_order_is_detected() -> None:
    result = split_full_name("Pérez García María José", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "María José"
    assert result.last_name == "Pérez García"
    assert result.detected_order == "Apellido-Nombre"


def test_unknown_tokens_are_left_indeterminate() -> None:
    result = split_full_name("Xyz Abcd", LEXICONS)

    assert result.reliable is False
    assert result.first_name == ""
    assert result.last_name == ""
    assert result.detected_order == "Indeterminado"


def test_one_known_surname_can_determine_two_token_order() -> None:
    result = split_full_name("Lunara Pérez", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Lunara"
    assert result.last_name == "Pérez"
    assert result.confidence == 0.85


def test_small_ocr_typo_in_name_uses_fuzzy_catalog_match() -> None:
    result = split_full_name("Marria Pérez", LEXICONS)

    assert result.reliable is True
    assert result.first_name == "Marria"
    assert result.last_name == "Pérez"
