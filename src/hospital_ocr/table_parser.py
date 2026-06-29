from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from hospital_ocr.models import OcrLine, PatientRecord
from hospital_ocr.name_splitter import NameLexicons, split_full_name
from hospital_ocr.text import clean_display_text, normalize_text


NAME_WORD_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}"
)
DOCUMENT_RE = re.compile(
    r"(?<!\d)(?:[VEve]\s*[-.]?\s*)?\d(?:[.\-·]?\d){5,10}(?!\d)"
)
TIME_RE = re.compile(
    r"\b\d{1,2}\s*[:.]\s*\d{2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")
HEADER_WORDS = {
    "nombre",
    "apellido",
    "cedula",
    "edad",
    "sexo",
    "telefono",
    "procedencia",
    "plan",
}
NON_NAME_WORDS = {
    "am",
    "pm",
    "pn",
    "ci",
    "cama",
    "sala",
    "edad",
    "sexo",
    "telefono",
    "procedencia",
    "plan",
    "nombre",
    "apellido",
    "nocturno",
}


@dataclass(frozen=True)
class _RowAnchor:
    line: OcrLine
    name: str


def _text_height(line: OcrLine) -> int:
    return max(1, line.box[3] - line.box[1])


def _has_table_header(lines: list[OcrLine]) -> bool:
    normalized = normalize_text(" ".join(line.text for line in lines))
    has_name = "nombre" in normalized and "apellido" in normalized
    supporting = sum(
        word in normalized
        for word in ("edad", "sexo", "procedencia", "plan", "telefono")
    )
    return has_name and supporting >= 1


def looks_like_table(lines: list[OcrLine]) -> bool:
    if not lines:
        return False
    if _has_table_header(lines):
        return True

    width = lines[0].image_width
    left_row_numbers = sum(
        bool(re.fullmatch(r"\s*\d{1,3}\.?\s*", line.text))
        and line.center_x < width * 0.16
        for line in lines
    )
    sex_markers = [
        line.center_x
        for line in lines
        if re.fullmatch(r"\s*[MFH]\s*", line.text, re.IGNORECASE)
        and width * 0.35 < line.center_x < width * 0.55
    ]
    aligned_sex_markers = (
        len(sex_markers) >= 4
        and max(sex_markers) - min(sex_markers) <= width * 0.08
    )
    name_candidates = sum(
        bool(_name_from_text(line.text))
        and line.box[0] < width * 0.34
        and line.box[2] > width * 0.08
        and line.center_x < width * 0.39
        for line in lines
    )
    document_markers = sum(
        bool(_document_digits(line.text))
        and line.box[2] >= width * 0.24
        and line.box[0] <= width * 0.44
        for line in lines
    )
    numbered_table = left_row_numbers >= 6 and aligned_sex_markers
    unnumbered_table = (
        name_candidates >= 6
        and document_markers >= 4
        and aligned_sex_markers
    )
    return numbered_table or unnumbered_table


def _name_from_text(text: str) -> str:
    cleaned = DATE_RE.sub(" ", text)
    cleaned = TIME_RE.sub(" ", cleaned)
    cleaned = DOCUMENT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"^\s*\d{1,3}\s*[.):\-]?\s*", " ", cleaned)
    cleaned = re.sub(r"\b(?:a|p)\s*\.?\s*m\.?\b", " ", cleaned, flags=re.I)
    words = [
        word
        for word in NAME_WORD_RE.findall(cleaned)
        if normalize_text(word) not in NON_NAME_WORDS
    ]
    if not 1 <= len(words) <= 6:
        return ""
    if len(words) == 1 and len(words[0]) < 6:
        return ""
    return clean_display_text(" ".join(words))


def _is_header_or_metadata(line: OcrLine) -> bool:
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


def _header_cutoff(lines: list[OcrLine]) -> float | None:
    header_lines = []
    for line in lines:
        words = set(normalize_text(line.text).split())
        if words & HEADER_WORDS:
            header_lines.append(line)
    if not header_lines:
        return None
    return max(line.center_y for line in header_lines) + 8


def _find_row_anchors(lines: list[OcrLine]) -> list[_RowAnchor]:
    width = lines[0].image_width
    candidates: list[_RowAnchor] = []
    for line in lines:
        reaches_name_column = (
            line.box[0] < width * 0.34
            and line.box[2] > width * 0.08
            and line.center_x < width * 0.39
        )
        if not reaches_name_column or _is_header_or_metadata(line):
            continue
        name = _name_from_text(line.text)
        if name:
            candidates.append(_RowAnchor(line, name))

    if not candidates:
        return []
    typical_height = median(_text_height(item.line) for item in candidates)
    same_row_tolerance = max(4.0, typical_height * 0.25)
    clusters: list[list[_RowAnchor]] = []
    for candidate in sorted(candidates, key=lambda item: item.line.center_y):
        if (
            clusters
            and abs(
                candidate.line.center_y
                - median(item.line.center_y for item in clusters[-1])
            )
            <= same_row_tolerance
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    anchors: list[_RowAnchor] = []
    for cluster in clusters:
        selected = max(
            cluster,
            key=lambda item: (len(item.name.split()), item.line.score),
        )
        combined_name = " ".join(
            dict.fromkeys(
                item.name
                for item in sorted(cluster, key=lambda item: item.line.center_x)
            )
        )
        anchors.append(_RowAnchor(selected.line, combined_name))
    return anchors


def _row_groups(
    lines: list[OcrLine],
    anchors: list[_RowAnchor],
) -> list[tuple[_RowAnchor, list[OcrLine]]]:
    centers = [anchor.line.center_y for anchor in anchors]
    groups: list[tuple[_RowAnchor, list[OcrLine]]] = []
    for index, anchor in enumerate(anchors):
        lower = (
            float("-inf")
            if index == 0
            else (centers[index - 1] + centers[index]) / 2
        )
        upper = (
            float("inf")
            if index == len(anchors) - 1
            else (centers[index] + centers[index + 1]) / 2
        )
        row_lines = [
            line for line in lines if lower < line.center_y <= upper
        ]
        groups.append((anchor, row_lines))
    return groups


def _document_digits(text: str) -> list[str]:
    return [
        re.sub(r"\D", "", match.group())
        for match in DOCUMENT_RE.finditer(text)
    ]


def _split_document_and_age(digits: str) -> tuple[str, int | None]:
    if 6 <= len(digits) <= 9:
        return digits, None
    if 10 <= len(digits) <= 11:
        for suffix_length in (2, 1, 3):
            document = digits[:-suffix_length]
            age_text = digits[-suffix_length:]
            age = int(age_text)
            if 6 <= len(document) <= 9 and 0 <= age <= 115:
                return document, age
    return "", None


def _extract_document_and_compound_age(
    lines: list[OcrLine],
    width: int,
) -> tuple[str, int | None]:
    candidates: list[tuple[float, str]] = []
    for line in lines:
        overlaps_document = (
            line.box[2] >= width * 0.24
            and line.box[0] <= width * 0.44
        )
        if not overlaps_document:
            continue
        for digits in _document_digits(line.text):
            document, age = _split_document_and_age(digits)
            if document:
                distance = abs(line.center_x / width - 0.33)
                candidates.append((distance, f"{document}|{age or ''}"))
    if not candidates:
        return "", None
    _, value = min(candidates, key=lambda item: item[0])
    document, age_text = value.split("|", 1)
    return document, int(age_text) if age_text else None


def _ocr_number(value: str) -> int | None:
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    compact = re.sub(r"^[CE]", "", compact)
    translation = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "B": "3", "S": "5"})
    compact = compact.translate(translation)
    if not compact.isdigit() or not 1 <= len(compact) <= 3:
        return None
    number = int(compact)
    return number if 0 <= number <= 115 else None


def _extract_age(
    lines: list[OcrLine],
    width: int,
    compound_age: int | None,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for line in lines:
        overlaps_age = (
            line.box[2] >= width * 0.36
            and line.box[0] <= width * 0.46
            and line.center_x >= width * 0.30
        )
        if not overlaps_age:
            continue
        if DATE_RE.search(line.text) or TIME_RE.search(line.text):
            continue
        text_without_document = DOCUMENT_RE.sub(" ", line.text)
        for token in re.findall(r"[A-Za-z]?\d{1,3}|[A-Za-z]\d", text_without_document):
            age = _ocr_number(token)
            if age is not None:
                distance = abs(line.center_x / width - 0.40)
                candidates.append((distance, age))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return compound_age


def _extract_sex(lines: list[OcrLine], width: int) -> str:
    candidates: list[tuple[float, str]] = []
    for line in lines:
        if not (
            line.box[2] >= width * 0.41
            and line.box[0] <= width * 0.51
        ):
            continue
        marker = re.sub(r"[^A-Za-z]", "", line.text).upper()
        if marker in {"M", "H", "N"}:
            sex = "M"
        elif marker == "F":
            sex = "F"
        else:
            continue
        candidates.append((abs(line.center_x / width - 0.455), sex))
    return min(candidates, default=(0.0, ""), key=lambda item: item[0])[1]


def _column_text(
    lines: list[OcrLine],
    width: int,
    start: float,
    end: float | None,
) -> str:
    selected = []
    for line in lines:
        normalized_start = line.box[0] / width
        normalized_center = line.center_x / width
        in_column = normalized_start >= start or normalized_center >= start
        if end is not None:
            in_column = in_column and normalized_center < end
        if in_column and not _is_header_or_metadata(line):
            selected.append(line)
    return clean_display_text(
        " ".join(line.text for line in sorted(selected, key=lambda item: item.center_x))
    )


def parse_table_lines(
    lines: list[OcrLine],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
) -> list[PatientRecord] | None:
    if not looks_like_table(lines):
        return None
    anchors = _find_row_anchors(lines)
    if len(anchors) < 2:
        return []

    width = lines[0].image_width
    header_cutoff = _header_cutoff(lines)
    records: list[PatientRecord] = []
    for anchor, row_lines in _row_groups(lines, anchors):
        if header_cutoff is not None and anchor.line.center_y > header_cutoff:
            row_lines = [
                line for line in row_lines if line.center_y > header_cutoff
            ]
        document_id, compound_age = _extract_document_and_compound_age(
            row_lines,
            width,
        )
        age = _extract_age(row_lines, width, compound_age)
        sex = _extract_sex(row_lines, width)
        origin = _column_text(row_lines, width, 0.56, 0.71)
        plan = _column_text(row_lines, width, 0.70, None)
        name_split = split_full_name(anchor.name, name_lexicons)
        notes: list[str] = []
        if not name_split.reliable:
            notes.append("Separación de nombre no confiable")
        if age is None:
            notes.append("Edad no reconocida")
        if not sex:
            notes.append("Sexo no reconocido")
        if not origin:
            notes.append("Procedencia no reconocida")

        raw_line = " ".join(
            line.text for line in sorted(row_lines, key=lambda item: item.center_x)
        )
        scores = [line.score for line in row_lines if line.text.strip()]
        confidence = sum(scores) / len(scores) if scores else anchor.line.score
        records.append(
            PatientRecord(
                full_name=anchor.name,
                first_name=name_split.first_name,
                last_name=name_split.last_name,
                name_split_confidence=name_split.confidence,
                detected_name_order=name_split.detected_order,
                center=center,
                age=age,
                age_unit="años" if age is not None else "",
                sex=sex,
                origin=origin,
                specialty="",
                area="",
                source_image=source_image,
                confidence=round(confidence, 4),
                needs_review=bool(notes),
                notes=notes,
                raw_line=raw_line,
                document_id=document_id,
                clinical_notes=f"Plan: {plan}" if plan else "",
            )
        )
    return records
