from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from hospital_ocr.models import OcrLine, PatientRecord, Specialty
from hospital_ocr.name_splitter import NameLexicons, split_full_name
from hospital_ocr.text import clean_display_text, normalize_text


AGE_RE = re.compile(
    r"(?<!\d)(?P<age>\d{1,3})"
    r"(?:\s*(?P<unit>años?|anos?|a|e|mes(?:es)?|d[ií]as?))?"
    r"(?!\d)",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?\b", re.I)
FLOOR_RE = re.compile(r"\bpiso\s*(\d+)\b", re.I)
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


def detect_specialty(
    text: str, specialties: list[Specialty]
) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    for item in specialties:
        if re.search(rf"(?:^|\s){re.escape(item.alias)}(?:$|\s)", normalized):
            area = item.area
            floor = FLOOR_RE.search(normalized)
            if floor:
                floor_text = f"Piso {floor.group(1)}"
                area = f"{area} - {floor_text}" if area else floor_text
            return item.specialty, area

    candidate = re.sub(r"\b(?:piso\s*)?\d+\b", "", normalized).strip()
    candidate_words = candidate.split()
    for item in specialties:
        alias_words = item.alias.split()
        if len(item.alias) <= 3:
            continue
        windows = [
            " ".join(candidate_words[index : index + len(alias_words)])
            for index in range(max(1, len(candidate_words) - len(alias_words) + 1))
        ]
        ratio = max(
            (
                SequenceMatcher(None, window, item.alias).ratio()
                for window in windows
                if window
            ),
            default=0.0,
        )
        if ratio >= 0.82:
            return item.specialty, item.area
    return None


def _is_metadata(text: str) -> bool:
    normalized = normalize_text(text)
    if DATE_RE.search(text) or TIME_RE.search(text):
        return True
    return normalized.startswith(("lista ", "actualizada ", "hora "))


def _age_match(text: str) -> re.Match[str] | None:
    matches = [
        match
        for match in AGE_RE.finditer(text)
        if 0 <= int(match.group("age")) <= 115
    ]
    return matches[-1] if matches else None


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
    if any(phrase in normalized for phrase in NON_PATIENT_PHRASES):
        return False
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]{2,}", text)
    return 2 <= len(words) <= 6


def parse_ocr_lines(
    lines: list[OcrLine],
    specialties: list[Specialty],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
) -> list[PatientRecord]:
    probe_lines = _merge_close_lines(lines)
    probe_headings = [
        line for line in probe_lines if detect_specialty(line.text, specialties)
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
        detected = detect_specialty(line.text, specialties)
        if detected:
            headings.append(Heading(line, detected[0], detected[1]))
            heading_ids.add(id(line))

    records: list[PatientRecord] = []
    for line in merged:
        if id(line) in heading_ids or _is_metadata(line.text):
            continue
        context = _heading_for_line(line, headings)
        age_match = _age_match(line.text)
        if age_match:
            age = int(age_match.group("age"))
            age_unit, uncertain_unit = _normalize_age_unit(age_match.group("unit"))
            name_text = clean_display_text(line.text[: age_match.start()])
            remainder = clean_display_text(line.text[age_match.end() :])
        else:
            if context is None or not _looks_like_missing_age_candidate(line.text):
                continue
            age = None
            age_unit = ""
            uncertain_unit = False
            name_text = clean_display_text(line.text)
            remainder = ""

        sex = ""
        if age is None:
            trailing_sex = re.search(
                r"[\s(\[]+([MF])[\s)\].,;]*$", name_text, re.IGNORECASE
            )
            if trailing_sex:
                sex = trailing_sex.group(1).upper()
                name_text = clean_display_text(name_text[: trailing_sex.start()])
        sex_match = re.match(r"^\s*([MF])(?:\b|(?=[^A-Za-z]))", remainder, re.I)
        if sex_match:
            sex = sex_match.group(1).upper()
            remainder = clean_display_text(remainder[sex_match.end() :])
        origin = remainder

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
        if context is None:
            notes.append("Especialidad o área no reconocida")
        if line.score < 0.75:
            notes.append("Baja confianza del OCR")
        if uncertain_unit:
            notes.append("Unidad de edad interpretada como años")

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
                specialty=context.specialty if context else "",
                area=context.area if context else "",
                source_image=source_image,
                confidence=round(line.score, 4),
                needs_review=bool(notes),
                notes=notes,
                raw_line=line.text,
            )
        )
    return records
