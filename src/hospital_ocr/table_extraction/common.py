from __future__ import annotations

import re
from difflib import SequenceMatcher

from hospital_ocr.models import OcrLine
from hospital_ocr.text import clean_display_text, normalize_text


NAME_WORD_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}"
)
DOCUMENT_RE = re.compile(
    r"(?<!\d)(?:[VEve]\s*[-.]?\s*)?\d(?:[.,\-·]?\d){5,10}(?!\d)"
)
TIME_RE = re.compile(
    r"\b\d{1,2}\s*[:.]\s*\d{2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")
SEMANTIC_AGE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>[0-9OQILBSGZEU]{1,3})\s*"
    r"(?P<unit>años?|anos?|a|e|mes(?:es)?|d[ií]as?)"
    r"(?=$|[^A-Za-z]|[MFH](?:\W|$))",
    re.IGNORECASE,
)
HEADER_WORDS = {
    "nombre",
    "apellido",
    "cedula",
    "edad",
    "sexo",
    "telefono",
    "procedencia",
    "plan",
    "paciente",
    "identificacion",
    "documento",
    "genero",
    "origen",
    "localidad",
    "sector",
    "direccion",
    "especialidad",
    "servicio",
    "area",
    "conducta",
    "observaciones",
    "cama",
    "habitacion",
    "cubiculo",
    "afiliacion",
    "diagnostico",
    "historia",
    "numero",
}
HEADER_ALIASES = {
    "name": (
        "nombre y apellido",
        "nombres y apellidos",
        "nombre apellido",
        "apellidos y nombres",
        "nombre completo",
        "paciente",
    ),
    "given_names": (
        "nombres",
        "nombre",
    ),
    "surnames": (
        "apellidos",
        "apellido",
    ),
    "document": (
        "cedula de identidad",
        "identificacion",
        "documento",
        "cedula",
        "c i",
        "ci",
    ),
    "age": ("edad", "unidad"),
    "sex": ("sexo", "genero"),
    "origin": (
        "lugar de procedencia",
        "procedencia",
        "localidad",
        "direccion",
        "origen",
        "sector",
    ),
    "specialty": ("especialidad", "servicio", "area"),
    "plan": ("observaciones", "tratamiento", "conducta", "plan"),
    "ignored_bed": ("habitacion", "cubiculo", "cama"),
    "ignored_affiliation": ("afiliacion",),
    "ignored_diagnosis": ("diagnostico",),
    "ignored_phone": ("telefono", "celular"),
    "ignored_history": (
        "historia clinica",
        "numero de historia",
        "n historia",
    ),
}
NON_NAME_WORDS = {
    "am",
    "ano",
    "anos",
    "pm",
    "pn",
    "ci",
    "cama",
    "dato",
    "datos",
    "sala",
    "edad",
    "sexo",
    "telefono",
    "procedencia",
    "plan",
    "nombre",
    "apellido",
    "nocturno",
    "numero",
    "sin",
}


def text_height(line: OcrLine) -> int:
    return max(1, line.box[3] - line.box[1])


def semantic_age_tokens(
    text: str,
) -> list[tuple[tuple[int, int], int, str]]:
    tokens: list[tuple[tuple[int, int], int, str]] = []
    translation = str.maketrans(
        {
            "O": "0",
            "Q": "0",
            "I": "1",
            "L": "1",
            "B": "3",
            "S": "5",
            "G": "6",
            "Z": "2",
            "E": "2",
            "U": "4",
        }
    )
    for match in SEMANTIC_AGE_RE.finditer(text):
        if normalize_text(match.group()) in {"la", "le", "se"}:
            continue
        compact = match.group("number").upper().translate(translation)
        if not compact.isdigit():
            continue
        age = int(compact)
        if 0 <= age <= 115:
            tokens.append((match.span(), age, match.group("unit")))
    return tokens


def remove_semantic_age_tokens(text: str) -> str:
    characters = list(text)
    for (start, end), _, _ in semantic_age_tokens(text):
        for index in range(start, end):
            characters[index] = " "
    return "".join(characters)


def name_from_text(
    text: str,
    *,
    allow_short_single: bool = False,
) -> str:
    cleaned = DATE_RE.sub(" ", text)
    cleaned = TIME_RE.sub(" ", cleaned)
    cleaned = DOCUMENT_RE.sub(" ", cleaned)
    cleaned = remove_semantic_age_tokens(cleaned)
    cleaned = re.sub(r"^\s*\d{1,3}\s*[.):\-]?\s*", " ", cleaned)
    cleaned = re.sub(r"\b(?:a|p)\s*\.?\s*m\.?\b", " ", cleaned, flags=re.I)
    words = [
        word
        for word in NAME_WORD_RE.findall(cleaned)
        if normalize_text(word) not in NON_NAME_WORDS
    ]
    if not 1 <= len(words) <= 6:
        return ""
    if len(words) == 1 and len(words[0]) < 6 and not allow_short_single:
        return ""
    return clean_display_text(" ".join(words))


def is_header_or_metadata(line: OcrLine) -> bool:
    normalized = normalize_text(line.text)
    words = set(normalized.split())
    if DATE_RE.search(line.text) or TIME_RE.fullmatch(line.text.strip()):
        return True
    if words & {"formato", "hospital", "nocturno"}:
        return True
    if words and words <= HEADER_WORDS:
        return True
    fuzzy_header_matches = sum(
        any(
            SequenceMatcher(None, word, header).ratio() >= 0.80
            for header in HEADER_WORDS
        )
        for word in words
    )
    return fuzzy_header_matches >= 2


def document_digits(text: str) -> list[str]:
    return [
        re.sub(r"\D", "", match.group())
        for match in DOCUMENT_RE.finditer(text)
    ]


def ocr_number(value: str) -> int | None:
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    compact = re.sub(r"^[CE]", "", compact)
    translation = str.maketrans(
        {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "3", "S": "5"}
    )
    compact = compact.translate(translation)
    if not compact.isdigit() or not 1 <= len(compact) <= 3:
        return None
    number = int(compact)
    return number if 0 <= number <= 115 else None
