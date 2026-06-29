from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from hospital_ocr.matching import detect_specialty, match_place
from hospital_ocr.models import OcrLine, PatientRecord, Place, Specialty
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
}
HEADER_ALIASES = {
    "name": (
        "nombre y apellido",
        "apellidos y nombres",
        "nombre completo",
        "paciente",
        "nombre",
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
    "age": ("edad", "anos", "ano"),
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


@dataclass(frozen=True)
class _HeaderCandidate:
    field: str
    line: OcrLine
    score: float


@dataclass(frozen=True)
class _Column:
    field: str
    center: float
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class _TableSchema:
    columns: dict[str, _Column]
    header_bottom: float
    confidence: float


def _text_height(line: OcrLine) -> int:
    return max(1, line.box[3] - line.box[1])


def _header_candidate(line: OcrLine) -> _HeaderCandidate | None:
    normalized = normalize_text(line.text)
    if not normalized:
        return None
    best_field = ""
    best_score = 0.0
    for field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if normalized == alias:
                score = 1.0
            elif re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized):
                score = 0.94
            else:
                score = SequenceMatcher(None, normalized, alias).ratio()
                if score < 0.78:
                    continue
            if score > best_score:
                best_field = field
                best_score = score
    if not best_field:
        return None
    return _HeaderCandidate(best_field, line, best_score)


def _header_baseline(
    candidates: list[_HeaderCandidate],
) -> tuple[float, float]:
    points = [
        (candidate.line.center_x, candidate.line.center_y)
        for candidate in candidates
    ]
    slopes = [
        (right_y - left_y) / (right_x - left_x)
        for index, (left_x, left_y) in enumerate(points)
        for right_x, right_y in points[index + 1 :]
        if abs(right_x - left_x) >= 20
    ]
    slope = median(slopes) if slopes else 0.0
    intercept = median(y - slope * x for x, y in points)
    return slope, intercept


def _looks_like_unknown_header(line: OcrLine) -> bool:
    normalized = normalize_text(line.text)
    if not normalized or DATE_RE.search(line.text) or TIME_RE.search(line.text):
        return False
    if DOCUMENT_RE.search(line.text):
        return False
    letters = sum(character.isalpha() for character in normalized)
    words = normalized.split()
    return letters >= 3 and len(words) <= 5 and len(normalized) <= 50


def _unknown_header_candidates(
    lines: list[OcrLine],
    known: list[_HeaderCandidate],
    all_known: list[_HeaderCandidate],
    typical_height: float,
) -> list[_HeaderCandidate]:
    if len(known) < 3:
        return []
    slope, intercept = _header_baseline(known)
    width = max(1, known[0].line.image_width)
    vertical_tolerance = max(10.0, typical_height)
    minimum_x = min(item.line.center_x for item in known)
    maximum_x = max(item.line.center_x for item in known)
    known_line_ids = {id(item.line) for item in all_known}
    known_centers = [item.line.center_x for item in known]
    minimum_horizontal_gap = max(20.0, width * 0.025)

    eligible: list[tuple[float, OcrLine]] = []
    for line in lines:
        if id(line) in known_line_ids or not _looks_like_unknown_header(line):
            continue
        if not minimum_x < line.center_x < maximum_x:
            continue
        if min(abs(line.center_x - center) for center in known_centers) < (
            minimum_horizontal_gap
        ):
            continue
        expected_y = slope * line.center_x + intercept
        distance = abs(line.center_y - expected_y)
        if distance <= vertical_tolerance:
            eligible.append((distance, line))

    clusters: list[list[tuple[float, OcrLine]]] = []
    for candidate in sorted(eligible, key=lambda item: item[1].center_x):
        if (
            clusters
            and abs(
                candidate[1].center_x
                - median(item[1].center_x for item in clusters[-1])
            )
            <= width * 0.04
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    unknown: list[_HeaderCandidate] = []
    for index, cluster in enumerate(clusters, start=1):
        distance, line = min(cluster, key=lambda item: item[0])
        geometric_score = max(
            0.75,
            1.0 - (distance / vertical_tolerance) * 0.25,
        )
        unknown.append(
            _HeaderCandidate(
                f"ignored_unknown_{index}",
                line,
                min(0.90, line.score * geometric_score),
            )
        )
    return unknown


def _infer_schema(lines: list[OcrLine]) -> _TableSchema | None:
    candidates = [
        candidate
        for line in lines
        if (candidate := _header_candidate(line)) is not None
    ]
    if not candidates:
        return None

    typical_height = median(_text_height(item.line) for item in candidates)
    tolerance = max(8.0, typical_height * 1.25)
    clusters: list[list[_HeaderCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.line.center_y):
        if (
            clusters
            and candidate.line.center_y
            - max(item.line.center_y for item in clusters[-1])
            <= tolerance
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    eligible = [
        cluster
        for cluster in clusters
        if len({item.field for item in cluster}) >= 2
        and "name" in {item.field for item in cluster}
    ]
    if not eligible:
        return None
    header_cluster = max(
        eligible,
        key=lambda cluster: (
            len({item.field for item in cluster}),
            sum(item.score for item in cluster),
        ),
    )
    header_cluster.extend(
        _unknown_header_candidates(
            lines,
            header_cluster,
            candidates,
            typical_height,
        )
    )
    width = max(1, header_cluster[0].line.image_width)
    grouped: dict[str, list[_HeaderCandidate]] = {}
    for candidate in header_cluster:
        grouped.setdefault(candidate.field, []).append(candidate)

    centers = {
        field: sum(item.line.center_x for item in items) / len(items) / width
        for field, items in grouped.items()
    }
    ordered = sorted(centers, key=centers.get)
    columns: dict[str, _Column] = {}
    for index, field in enumerate(ordered):
        start = (
            0.0
            if index == 0
            else (centers[ordered[index - 1]] + centers[field]) / 2
        )
        end = (
            1.0
            if index == len(ordered) - 1
            else (centers[field] + centers[ordered[index + 1]]) / 2
        )
        confidence = max(item.score for item in grouped[field])
        columns[field] = _Column(
            field,
            centers[field],
            max(0.0, start),
            min(1.0, end),
            confidence,
        )
    return _TableSchema(
        columns=columns,
        header_bottom=max(item.line.box[3] for item in header_cluster) + 4,
        confidence=sum(column.confidence for column in columns.values())
        / len(columns),
    )


def _has_table_header(lines: list[OcrLine]) -> bool:
    if _infer_schema(lines) is not None:
        return True
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


def _has_leading_index(text: str) -> bool:
    match = re.match(
        r"^\s*\d{1,3}(?:\s*[.):\-]\s*|\s+)",
        text,
    )
    if match is None:
        return False
    following = normalize_text(text[match.end() :]).split()
    return not (
        following
        and following[0] in {
            "a",
            "ano",
            "anos",
            "mes",
            "meses",
            "dia",
            "dias",
        }
    )


def _row_index_lines(
    lines: list[OcrLine],
    anchor: _RowAnchor,
    schema: _TableSchema | None,
) -> list[OcrLine]:
    width = max(1, anchor.line.image_width)
    age_column = schema.columns.get("age") if schema else None
    indexes: list[OcrLine] = []
    for line in lines:
        if not re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text):
            continue
        normalized_center = line.center_x / width
        if (
            age_column
            and age_column.start <= normalized_center < age_column.end
        ):
            continue
        if line.box[2] <= anchor.line.box[0] or normalized_center < 0.16:
            indexes.append(line)
    return indexes


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


def _find_row_anchors(
    lines: list[OcrLine],
    schema: _TableSchema | None = None,
) -> list[_RowAnchor]:
    width = lines[0].image_width
    candidates: list[_RowAnchor] = []
    for line in lines:
        if schema and "name" in schema.columns:
            name_column = schema.columns["name"]
            normalized_center = line.center_x / width
            reaches_name_column = (
                name_column.start <= normalized_center < name_column.end
            )
        else:
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
    name_line: OcrLine | None = None,
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
        candidate_text = line.text
        if line is name_line and _has_leading_index(candidate_text):
            candidate_text = re.sub(
                r"^\s*\d{1,3}(?:\s*[.):\-]\s*|\s+)",
                " ",
                candidate_text,
                count=1,
            )
        text_without_document = DOCUMENT_RE.sub(" ", candidate_text)
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


def _schema_lines(
    lines: list[OcrLine],
    schema: _TableSchema,
    field: str,
) -> list[OcrLine]:
    column = schema.columns.get(field)
    if column is None:
        return []
    width = max(1, lines[0].image_width) if lines else 1
    return [
        line
        for line in lines
        if column.start <= line.center_x / width < column.end
        and not _is_header_or_metadata(line)
    ]


def _schema_text(
    lines: list[OcrLine],
    schema: _TableSchema,
    field: str,
) -> str:
    return clean_display_text(
        " ".join(
            line.text
            for line in sorted(
                _schema_lines(lines, schema, field),
                key=lambda item: item.center_x,
            )
        )
    )


def _extract_document(lines: list[OcrLine]) -> str:
    candidates: list[tuple[int, float, str]] = []
    for line in lines:
        for digits in _document_digits(line.text):
            document, _ = _split_document_and_age(digits)
            if document:
                preferred_length = int(len(document) in {7, 8})
                candidates.append((preferred_length, line.score, document))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _extract_schema_age(lines: list[OcrLine]) -> int | None:
    candidates: list[tuple[float, int]] = []
    for line in lines:
        if DATE_RE.search(line.text) or TIME_RE.search(line.text):
            continue
        without_document = DOCUMENT_RE.sub(" ", line.text)
        for token in re.findall(r"[A-Za-z]?\d{1,3}|[A-Za-z]\d", without_document):
            age = _ocr_number(token)
            if age is not None:
                candidates.append((line.score, age))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _extract_schema_sex(lines: list[OcrLine]) -> str:
    for line in sorted(lines, key=lambda item: item.score, reverse=True):
        marker = re.sub(r"[^A-Za-z]", "", line.text).upper()
        if marker in {"M", "H", "N"}:
            return "M"
        if marker == "F":
            return "F"
    return ""


def _average_score(lines: list[OcrLine], fallback: float = 0.0) -> float:
    scores = [line.score for line in lines if line.text.strip()]
    return sum(scores) / len(scores) if scores else fallback


def parse_table_lines(
    lines: list[OcrLine],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
    specialties: list[Specialty] | None = None,
    places: list[Place] | None = None,
) -> list[PatientRecord] | None:
    specialties = specialties or []
    places = places or []
    if not looks_like_table(lines):
        return None
    schema = _infer_schema(lines)
    data_lines = (
        [line for line in lines if line.center_y > schema.header_bottom]
        if schema
        else lines
    )
    anchors = _find_row_anchors(data_lines, schema)
    if len(anchors) < 2:
        return []

    width = lines[0].image_width
    header_cutoff = schema.header_bottom if schema else _header_cutoff(lines)
    page_specialty: tuple[str, str] | None = None
    if schema:
        for line in sorted(lines, key=lambda item: item.center_y):
            if line.box[3] >= schema.header_bottom:
                continue
            if detected := detect_specialty(line.text, specialties):
                page_specialty = detected
                break
    records: list[PatientRecord] = []
    for anchor, row_lines in _row_groups(data_lines, anchors):
        if header_cutoff is not None and anchor.line.center_y > header_cutoff:
            row_lines = [
                line for line in row_lines if line.center_y > header_cutoff
            ]
        raw_row_lines = list(row_lines)
        index_lines = _row_index_lines(row_lines, anchor, schema)
        row_lines = [line for line in row_lines if line not in index_lines]
        index_detected = bool(index_lines or _has_leading_index(anchor.line.text))
        if schema:
            document_lines = _schema_lines(row_lines, schema, "document")
            age_lines = _schema_lines(row_lines, schema, "age")
            sex_lines = _schema_lines(row_lines, schema, "sex")
            origin_lines = _schema_lines(row_lines, schema, "origin")
            specialty_lines = _schema_lines(row_lines, schema, "specialty")
            document_id = _extract_document(document_lines)
            age = _extract_schema_age(age_lines)
            sex = _extract_schema_sex(sex_lines)
            origin_text = _schema_text(row_lines, schema, "origin")
            plan = _schema_text(row_lines, schema, "plan")
            specialty_text = _schema_text(row_lines, schema, "specialty")
        else:
            document_id, compound_age = _extract_document_and_compound_age(
                row_lines,
                width,
            )
            age = _extract_age(
                row_lines,
                width,
                compound_age,
                name_line=anchor.line,
            )
            sex = _extract_sex(row_lines, width)
            origin_text = _column_text(row_lines, width, 0.56, 0.71)
            plan = _column_text(row_lines, width, 0.70, None)
            document_lines = age_lines = sex_lines = origin_lines = []
            specialty_lines = []
            specialty_text = ""

        place = match_place(origin_text, places)
        has_explicit_origin = bool(schema and "origin" in schema.columns)
        origin = (
            place.name
            if place
            else origin_text
            if has_explicit_origin
            else ""
        )
        detected_specialty = (
            detect_specialty(specialty_text, specialties)
            if specialty_text
            else page_specialty
        )
        specialty = detected_specialty[0] if detected_specialty else ""
        area = detected_specialty[1] if detected_specialty else ""
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
        elif has_explicit_origin and places and place is None:
            notes.append("Procedencia no validada en catálogo")
        if specialty_text and detected_specialty is None:
            notes.append("Especialidad o área no reconocida")

        raw_line = " ".join(
            line.text
            for line in sorted(raw_row_lines, key=lambda item: item.center_x)
        )
        scores = [line.score for line in row_lines if line.text.strip()]
        confidence = sum(scores) / len(scores) if scores else anchor.line.score
        if schema:
            name_confidence = min(
                1.0,
                anchor.line.score
                * (0.80 + 0.20 * schema.columns["name"].confidence),
            )
            document_confidence = (
                min(
                    1.0,
                    _average_score(document_lines)
                    * (0.80 + 0.20 * schema.columns["document"].confidence),
                )
                if document_id and "document" in schema.columns
                else 0.0
            )
            age_confidence = (
                min(
                    1.0,
                    _average_score(age_lines)
                    * (0.80 + 0.20 * schema.columns["age"].confidence),
                )
                if age is not None and "age" in schema.columns
                else 0.0
            )
            origin_confidence = (
                min(
                    1.0,
                    _average_score(origin_lines)
                    * (place.score if place else 0.65),
                )
                if origin
                else 0.0
            )
            specialty_confidence = (
                min(
                    1.0,
                    _average_score(specialty_lines, anchor.line.score)
                    * 0.90,
                )
                if specialty
                else 0.0
            )
            evidence = {
                "nombre": (
                    "columna inferida por encabezado; índice descartado"
                    if index_detected
                    else "columna inferida por encabezado"
                ),
                "cédula": (
                    "formato y columna inferida por encabezado"
                    if document_id
                    else ""
                ),
                "edad": (
                    "formato y columna inferida por encabezado"
                    if age is not None
                    else ""
                ),
                "procedencia": (
                    "encabezado y catálogo geográfico"
                    if place
                    else "columna inferida por encabezado"
                    if origin
                    else ""
                ),
                "especialidad": (
                    "catálogo y columna inferida por encabezado"
                    if specialty_text and specialty
                    else "catálogo y membrete"
                    if specialty
                    else ""
                ),
            }
        else:
            name_confidence = anchor.line.score * 0.70
            document_confidence = confidence * 0.70 if document_id else 0.0
            age_confidence = confidence * 0.60 if age is not None else 0.0
            origin_confidence = confidence * 0.50 if origin else 0.0
            specialty_confidence = 0.0
            evidence = {
                "nombre": (
                    "posición de respaldo; índice descartado"
                    if index_detected
                    else "posición de respaldo"
                ),
                "cédula": "formato y posición de respaldo" if document_id else "",
                "edad": "formato y posición de respaldo" if age is not None else "",
                "procedencia": (
                    "catálogo geográfico y posición de respaldo"
                    if place
                    else ""
                ),
            }
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
                specialty=specialty,
                area=area,
                source_image=source_image,
                confidence=round(confidence, 4),
                needs_review=bool(notes),
                notes=notes,
                raw_line=raw_line,
                document_id=document_id,
                clinical_notes=f"Plan: {plan}" if plan else "",
                name_confidence=round(name_confidence, 4),
                document_confidence=round(document_confidence, 4),
                age_confidence=round(age_confidence, 4),
                origin_confidence=round(origin_confidence, 4),
                specialty_confidence=round(specialty_confidence, 4),
                field_evidence=evidence,
            )
        )
    return records
