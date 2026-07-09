from __future__ import annotations

import re
from dataclasses import dataclass

from hospital_ocr.matching import detect_specialty, match_place, match_places
from hospital_ocr.models import (
    OcrLine,
    PatientRecord,
    Place,
    Specialty,
    TableGrid,
)
from hospital_ocr.name_splitter import NameLexicons, split_full_name
from hospital_ocr.table_parser import parse_table_lines
from hospital_ocr.text import clean_display_text, normalize_text


AGE_RE = re.compile(
    r"(?<!\d)(?P<age>\d{1,3})"
    r"(?:\s*(?P<unit>años?|anos?|a|e|mes(?:es)?|d[ií]as?))?"
    r"(?![A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])",
    re.IGNORECASE,
)
DOCUMENT_RE = re.compile(
    r"(?<!\d)(?P<prefix>[VEve])?\s*[-.]?\s*"
    r"(?P<number>\d(?:[.,\-·]?\d){5,8})(?!\d)"
)
LEADING_INDEX_RE = re.compile(
    r"^\s*(?P<index>\d{1,3})(?:\s*[.):\-]\s*|\s+)"
)
DATE_RE = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?\b", re.I)
NON_PATIENT_PHRASES = {
    "lista actualizada",
    "hospital",
    "traslado",
    "la guaira",
    "guarenas",
    "petare",
    "caribe",
    "los corales",
    "playa grande",
}


@dataclass(frozen=True)
class Heading:
    line: OcrLine
    specialty: str
    area: str


def _merge_pair(left: OcrLine, right: OcrLine) -> OcrLine:
    total = max(left.score + right.score, 0.0001)
    score = ((left.score * left.score) + (right.score * right.score)) / total
    return OcrLine(
        text=f"{left.text} {right.text}".strip(),
        score=score,
        box=(
            min(left.box[0], right.box[0]),
            min(left.box[1], right.box[1]),
            max(left.box[2], right.box[2]),
            max(left.box[3], right.box[3]),
        ),
        image_width=left.image_width,
        image_height=left.image_height,
    )


def _merge_close_lines(lines: list[OcrLine]) -> list[OcrLine]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: (item.center_y, item.center_x))
    merged: list[OcrLine] = []
    for line in ordered:
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        previous_height = max(1, previous.box[3] - previous.box[1])
        line_height = max(1, line.box[3] - line.box[1])
        same_row = abs(previous.center_y - line.center_y) <= 0.45 * max(
            previous_height, line_height
        )
        horizontal_gap = line.box[0] - previous.box[2]
        close = 0 <= horizontal_gap <= line.image_width * 0.06
        if same_row and close:
            merged[-1] = _merge_pair(previous, line)
        else:
            merged.append(line)
    return merged


def merge_nearby_lines(
    lines: list[OcrLine], *, two_columns: bool = False
) -> list[OcrLine]:
    if not lines:
        return []
    width = lines[0].image_width
    partitions: list[list[OcrLine]]
    if two_columns:
        partitions = [
            [line for line in lines if line.center_x < width / 2],
            [line for line in lines if line.center_x >= width / 2],
        ]
    else:
        partitions = [list(lines)]

    merged: list[OcrLine] = []
    for partition in partitions:
        clusters: list[list[OcrLine]] = []
        for line in sorted(partition, key=lambda item: item.center_y):
            if not clusters:
                clusters.append([line])
                continue
            cluster = clusters[-1]
            cluster_y = sum(item.center_y for item in cluster) / len(cluster)
            max_height = max(
                max(1, item.box[3] - item.box[1]) for item in [*cluster, line]
            )
            if abs(cluster_y - line.center_y) <= max_height * 0.45:
                cluster.append(line)
            else:
                clusters.append([line])
        for cluster in clusters:
            ordered = sorted(cluster, key=lambda item: item.center_x)
            combined = ordered[0]
            for line in ordered[1:]:
                combined = _merge_pair(combined, line)
            merged.append(combined)
    return sorted(merged, key=lambda item: (item.center_y, item.center_x))


def _is_metadata(text: str) -> bool:
    normalized = normalize_text(text)
    if DATE_RE.search(text) or TIME_RE.search(text):
        return True
    if re.match(
        r"^\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)",
        normalized,
    ):
        return True
    return normalized.startswith(
        ("lista ", "actualizada ", "hora ", "hospital ")
    )


def _overlaps(span: tuple[int, int], excluded: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in excluded)


def _document_match(text: str) -> re.Match[str] | None:
    candidates = []
    for match in DOCUMENT_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group("number"))
        if 6 <= len(digits) <= 9:
            candidates.append(match)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda match: (
            len(re.sub(r"\D", "", match.group("number"))) in {7, 8},
            bool(match.group("prefix")),
        ),
    )


def _normalized_document(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    digits = re.sub(r"\D", "", match.group("number"))
    prefix = (match.group("prefix") or "").upper()
    return f"{prefix}-{digits}" if prefix else digits


def _leading_index_span(text: str) -> tuple[int, int] | None:
    match = LEADING_INDEX_RE.match(text)
    if match is None:
        return None
    following = normalize_text(text[match.end() :]).split()
    if following and following[0] in {
        "a",
        "ano",
        "anos",
        "mes",
        "meses",
        "dia",
        "dias",
    }:
        return None
    return match.span()


def _strip_leading_index(text: str) -> str:
    span = _leading_index_span(text)
    return clean_display_text(text[span[1] :]) if span else clean_display_text(text)


def _age_match(
    text: str,
    excluded: list[tuple[int, int]] | None = None,
) -> re.Match[str] | None:
    excluded = excluded or []
    matches = [
        match
        for match in AGE_RE.finditer(text)
        if 0 <= int(match.group("age")) <= 115
        and not _overlaps(match.span(), excluded)
        and not _overlaps(
            match.span(),
            [item.span() for item in DATE_RE.finditer(text)],
        )
        and not _overlaps(
            match.span(),
            [item.span() for item in TIME_RE.finditer(text)],
        )
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda match: (
            bool(match.group("unit")),
            match.start() > 0,
            match.start(),
        ),
    )


def _normalize_age_unit(unit: str | None) -> tuple[str, bool]:
    normalized = normalize_text(unit or "")
    if normalized.startswith("dia"):
        return "días", False
    if normalized.startswith("mes"):
        return "meses", False
    if normalized == "e":
        return "años", True
    return "años", False


def _heading_for_line(line: OcrLine, headings: list[Heading]) -> Heading | None:
    if not headings:
        return None
    width = line.image_width
    has_left = any(heading.line.center_x < width * 0.4 for heading in headings)
    has_right = any(heading.line.center_x > width * 0.6 for heading in headings)
    two_columns = has_left and has_right

    candidates = [
        heading
        for heading in headings
        if heading.line.center_y <= line.center_y
    ]
    if two_columns:
        line_side = line.center_x < width / 2
        same_side = [
            heading
            for heading in candidates
            if (heading.line.center_x < width / 2) == line_side
        ]
        if same_side:
            candidates = same_side
    if not candidates:
        return None
    return max(candidates, key=lambda heading: heading.line.center_y)


def _looks_like_missing_age_candidate(text: str) -> bool:
    normalized = normalize_text(text)
    if normalized in NON_PATIENT_PHRASES:
        return False
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}", text)
    return 2 <= len(words) <= 6


def _standalone_sex(
    text: str,
    minimum_position: int = 0,
) -> tuple[str, tuple[int, int] | None]:
    candidates = [
        match
        for match in re.finditer(r"(?<![A-Za-z])([MFH])(?![A-Za-z])", text, re.I)
        if match.start() >= minimum_position
    ]
    if not candidates:
        return "", None
    selected = candidates[0]
    marker = selected.group(1).upper()
    return ("M" if marker == "H" else marker), selected.span()


def _fallback_name_text(
    text: str,
    excluded: list[tuple[int, int]],
    place_alias: str = "",
) -> str:
    characters = list(text)
    for start, end in excluded:
        for index in range(max(0, start), min(len(characters), end)):
            characters[index] = " "
    remaining = "".join(characters)
    place_words = set(place_alias.split())
    words = [
        word
        for word in re.findall(
            r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}",
            remaining,
        )
        if normalize_text(word)
        not in {
            "ano",
            "anos",
            "mes",
            "meses",
            "dia",
            "dias",
            "edad",
            "cedula",
            "ci",
            *place_words,
        }
    ]
    return clean_display_text(" ".join(words[:6]))


def _specialty_heading(
    line: OcrLine,
    specialties: list[Specialty],
) -> tuple[str, str] | None:
    detected = detect_specialty(line.text, specialties)
    if not detected:
        return None
    document = _document_match(line.text)
    excluded = [document.span()] if document else []
    if document or _age_match(line.text, excluded):
        return None
    return detected


def _inline_name_text(
    text: str,
    index_span: tuple[int, int] | None,
    document_match: re.Match[str] | None,
    age_match: re.Match[str] | None,
) -> str:
    start = index_span[1] if index_span else 0
    boundaries = [
        match.start()
        for match in (document_match, age_match)
        if match is not None
    ]
    comma = text.find(",", start)
    if comma >= 0:
        boundaries.append(comma)
    end = min(boundaries, default=len(text))
    return _strip_leading_index(text[start:end])


def _looks_like_inline_list(
    lines: list[OcrLine],
    specialties: list[Specialty],
    places: list[Place],
) -> bool:
    strong_rows = 0
    for line in lines:
        if _is_metadata(line.text):
            continue
        index_span = _leading_index_span(line.text)
        document = _document_match(line.text)
        excluded = [index_span] if index_span else []
        if document:
            excluded.append(document.span())
        age = _age_match(line.text, excluded)
        name_text = _inline_name_text(
            line.text,
            index_span,
            document,
            age,
        )
        name_tokens = re.findall(
            r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}",
            name_text,
        )
        sex, _ = _standalone_sex(
            line.text,
            age.end() if age else document.end() if document else 0,
        )
        has_other_field = bool(
            document
            or age
            or sex
            or match_place(line.text, places)
            or detect_specialty(line.text, specialties)
        )
        row_marker = bool(index_span or "," in line.text)
        if len(name_tokens) >= 2 and has_other_field and row_marker:
            strong_rows += 1
    return strong_rows >= 3


def parse_ocr_lines(
    lines: list[OcrLine],
    specialties: list[Specialty],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
    places: list[Place] | None = None,
    grid: TableGrid | None = None,
) -> list[PatientRecord]:
    places = places or []
    inline_list = _looks_like_inline_list(lines, specialties, places)
    table_records = (
        None
        if inline_list
        else parse_table_lines(
            lines,
            name_lexicons,
            center,
            source_image,
            specialties,
            places,
            grid,
        )
    )
    if table_records is not None:
        return table_records

    probe_lines = _merge_close_lines(lines)
    probe_headings = [
        line for line in probe_lines if _specialty_heading(line, specialties)
    ]
    width = lines[0].image_width if lines else 0
    two_columns = bool(
        any(line.center_x < width * 0.4 for line in probe_headings)
        and any(line.center_x > width * 0.6 for line in probe_headings)
    )
    merged = merge_nearby_lines(lines, two_columns=two_columns)
    headings: list[Heading] = []
    heading_ids: set[int] = set()
    for line in merged:
        detected = _specialty_heading(line, specialties)
        if detected:
            headings.append(Heading(line, detected[0], detected[1]))
            heading_ids.add(id(line))
    if not headings:
        for line in probe_headings:
            detected = _specialty_heading(line, specialties)
            if detected:
                headings.append(Heading(line, detected[0], detected[1]))

    records: list[PatientRecord] = []
    for line in merged:
        if id(line) in heading_ids or _is_metadata(line.text):
            continue
        context = _heading_for_line(line, headings)
        index_span = _leading_index_span(line.text)
        document_match = _document_match(line.text)
        document_id = _normalized_document(document_match)
        excluded = [index_span] if index_span else []
        if document_match:
            excluded.append(document_match.span())
        age_match = _age_match(line.text, excluded)
        if age_match:
            excluded.append(age_match.span())
        primary_field_positions = [
            match.start()
            for match in (document_match, age_match)
            if match is not None
        ]
        sex_search_start = (
            min(primary_field_positions)
            if primary_field_positions
            else index_span[1]
            if index_span
            else 0
        )
        sex, sex_span = _standalone_sex(line.text, sex_search_start)
        if sex_span:
            excluded.append(sex_span)
        first_field_position = min(
            [*primary_field_positions, sex_span[0] if sex_span else len(line.text)],
            default=len(line.text),
        )
        place_probe_start = (
            age_match.end()
            if age_match
            else document_match.end()
            if document_match
            else 0
        )
        place_matches = match_places(line.text[place_probe_start:], places)
        if not place_matches and inline_list:
            index_end = index_span[1] if index_span else 0
            comma = line.text.find(",", index_end)
            if comma >= 0:
                place_matches = match_places(line.text[comma + 1 :], places)
        place = place_matches[0] if place_matches else None
        if age_match:
            age = int(age_match.group("age"))
            age_unit, uncertain_unit = _normalize_age_unit(age_match.group("unit"))
            name_text = _inline_name_text(
                line.text,
                index_span,
                document_match,
                age_match,
            )
            remainder = DOCUMENT_RE.sub(" ", line.text[age_match.end() :])
            remainder = re.sub(
                r"^\s*[MFH]\s*(?:\b|(?=[^A-Za-z]))",
                " ",
                remainder,
                flags=re.IGNORECASE,
            )
            remainder = clean_display_text(remainder)
        else:
            if (
                context is None
                and not inline_list
            ) or (
                not _looks_like_missing_age_candidate(line.text)
                and not (
                    inline_list
                    and (document_match or place_matches or sex_span)
                )
            ):
                continue
            age = None
            age_unit = ""
            uncertain_unit = False
            name_text = _inline_name_text(
                line.text,
                index_span,
                document_match,
                age_match,
            )
            remainder = ""

        alphabetic_tokens = re.findall(
            r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}",
            name_text,
        )
        if len(alphabetic_tokens) < 2 or (
            age is None
            and not inline_list
            and (index_span or place or sex_span)
        ):
            name_text = _fallback_name_text(
                line.text,
                excluded,
                " ".join(match.alias for match in place_matches),
            )
        origin = " - ".join(match.name for match in place_matches)
        row_specialty = detect_specialty(line.text, specialties)
        specialty = context.specialty if context else ""
        area = context.area if context else ""
        specialty_source = "membrete"
        if not specialty and row_specialty:
            specialty, area = row_specialty
            specialty_source = "contenido de la fila"

        name_split = split_full_name(name_text, name_lexicons)
        first_name = name_split.first_name
        last_name = name_split.last_name
        if not first_name:
            alphabetic_tokens = re.findall(
                r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}", name_text
            )
            if not alphabetic_tokens:
                continue
        notes: list[str] = []
        if not name_split.reliable:
            notes.append("Separación de nombre no confiable")
        if age is None:
            notes.append("Edad no reconocida")
        if not sex:
            notes.append("Sexo no reconocido")
        if not origin:
            notes.append("Procedencia no reconocida")
        if not specialty:
            notes.append("Especialidad o área no reconocida")
        if line.score < 0.75:
            notes.append("Baja confianza del OCR")
        if uncertain_unit:
            notes.append("Unidad de edad interpretada como años")

        name_confidence = min(
            1.0,
            line.score * (0.75 + 0.25 * name_split.confidence),
        )
        document_confidence = line.score * 0.90 if document_id else 0.0
        age_confidence = (
            line.score * (0.95 if age_match and age_match.group("unit") else 0.75)
            if age is not None
            else 0.0
        )
        origin_confidence = (
            line.score * min(match.score for match in place_matches)
            if place_matches
            else line.score * 0.45 if origin
            else 0.0
        )
        specialty_confidence = line.score * 0.90 if specialty else 0.0
        evidence = {
            "nombre": (
                "formato alfabético y catálogo de nombres; índice descartado"
                if index_span
                else "formato alfabético y catálogo de nombres"
            ),
            "cédula": "formato de documento" if document_id else "",
            "edad": (
                "formato con unidad"
                if age_match and age_match.group("unit")
                else "formato numérico y posición de respaldo"
                if age is not None
                else ""
            ),
            "procedencia": (
                "catálogo geográfico"
                if place_matches
                else ""
            ),
            "especialidad": (
                f"catálogo de especialidades y {specialty_source}"
                if specialty
                else ""
            ),
        }

        records.append(
            PatientRecord(
                full_name=name_text,
                first_name=first_name,
                last_name=last_name,
                name_split_confidence=name_split.confidence,
                detected_name_order=name_split.detected_order,
                center=center,
                age=age,
                age_unit=age_unit,
                sex=sex,
                origin=origin,
                specialty=specialty,
                area=area,
                source_image=source_image,
                confidence=round(line.score, 4),
                needs_review=bool(notes),
                notes=notes,
                raw_line=line.text,
                document_id=document_id,
                name_confidence=round(name_confidence, 4),
                document_confidence=round(document_confidence, 4),
                age_confidence=round(age_confidence, 4),
                origin_confidence=round(origin_confidence, 4),
                specialty_confidence=round(specialty_confidence, 4),
                field_evidence=evidence,
            )
        )
    return records
