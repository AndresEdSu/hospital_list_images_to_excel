from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from hospital_ocr.matching import detect_specialty, match_place
from hospital_ocr.models import (
    OcrLine,
    PatientRecord,
    Place,
    Specialty,
    TableGrid,
)
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
    grid_index: int | None = None


@dataclass(frozen=True)
class _TableSchema:
    columns: dict[str, _Column]
    header_bottom: float
    confidence: float


@dataclass(frozen=True)
class _SexResult:
    value: str
    normalized_from: tuple[str, ...] = ()
    conflict: bool = False


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


def _infer_schema(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> _TableSchema | None:
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
    vertical_centers = {
        field: sum(item.line.center_y for item in items) / len(items)
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
        grid_index = (
            grid.column_index(
                centers[field] * width,
                vertical_centers[field],
            )
            if grid
            else None
        )
        columns[field] = _Column(
            field,
            centers[field],
            max(0.0, start),
            min(1.0, end),
            confidence,
            grid_index,
        )
    schema = _TableSchema(
        columns=columns,
        header_bottom=max(item.line.box[3] for item in header_cluster) + 4,
        confidence=sum(column.confidence for column in columns.values())
        / len(columns),
    )
    return _complete_partial_schema(schema, lines, grid)


def _complete_partial_schema(
    schema: _TableSchema,
    lines: list[OcrLine],
    grid: TableGrid | None,
) -> _TableSchema:
    if grid is None or "name" not in schema.columns:
        return schema

    used_grid_indexes = {
        column.grid_index
        for field, column in schema.columns.items()
        if column.grid_index is not None
        and not field.startswith("ignored_unknown_")
    }
    name_grid_index = schema.columns["name"].grid_index
    by_column_and_row: dict[int, dict[int, list[OcrLine]]] = {}
    for line in lines:
        if line.center_y <= schema.header_bottom:
            continue
        column_index = grid.column_for_box(line.box)
        row_index = grid.row_for_box(line.box)
        if column_index is None or row_index is None:
            continue
        by_column_and_row.setdefault(column_index, {}).setdefault(
            row_index,
            [],
        ).append(line)

    candidate_indexes = [
        index
        for index in by_column_and_row
        if index not in used_grid_indexes
        and (
            name_grid_index is None
            or index > name_grid_index
        )
    ]
    if not candidate_indexes:
        return schema

    row_texts = {
        column_index: [
            clean_display_text(
                " ".join(
                    line.text
                    for line in sorted(row_lines, key=lambda item: item.center_x)
                )
            )
            for row_lines in rows.values()
        ]
        for column_index, rows in by_column_and_row.items()
    }

    selected: dict[str, tuple[int, float]] = {}

    if "document" not in schema.columns:
        document_candidates = []
        for column_index in candidate_indexes:
            texts = row_texts[column_index]
            matches = sum(bool(DOCUMENT_RE.search(text)) for text in texts)
            if matches >= 2:
                document_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        -column_index,
                        column_index,
                    )
                )
        if document_candidates:
            matches, ratio, _, column_index = max(document_candidates)
            selected["document"] = (
                column_index,
                min(0.94, 0.72 + 0.18 * ratio + 0.01 * matches),
            )

    unavailable = {
        *used_grid_indexes,
        *(column_index for column_index, _ in selected.values()),
    }
    if "sex" not in schema.columns:
        sex_candidates = []
        for column_index in candidate_indexes:
            if column_index in unavailable:
                continue
            texts = row_texts[column_index]
            matches = sum(
                re.sub(r"[^A-Za-z]", "", text).upper()
                in {"M", "F", "H", "N"}
                for text in texts
            )
            if matches >= 2:
                sex_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        column_index,
                    )
                )
        if sex_candidates:
            matches, ratio, column_index = max(sex_candidates)
            selected["sex"] = (
                column_index,
                min(0.94, 0.72 + 0.18 * ratio + 0.01 * matches),
            )

    unavailable.update(
        column_index for column_index, _ in selected.values()
    )
    if "age" not in schema.columns:
        age_candidates = []
        for column_index in candidate_indexes:
            if column_index in unavailable:
                continue
            texts = row_texts[column_index]
            ages = [
                _ocr_number(text)
                for text in texts
                if not DOCUMENT_RE.search(text)
            ]
            matches = sum(age is not None for age in ages)
            if matches >= 2:
                age_candidates.append(
                    (
                        matches,
                        matches / max(1, len(texts)),
                        -column_index,
                        column_index,
                    )
                )
        if age_candidates:
            matches, ratio, _, column_index = max(age_candidates)
            selected["age"] = (
                column_index,
                min(0.92, 0.70 + 0.17 * ratio + 0.01 * matches),
            )

    if not selected:
        return schema

    columns = dict(schema.columns)
    width = max(1, lines[0].image_width)
    representative_y = schema.header_bottom + 1
    for field, (grid_index, confidence) in selected.items():
        columns = {
            existing_field: column
            for existing_field, column in columns.items()
            if not (
                existing_field.startswith("ignored_unknown_")
                and column.grid_index == grid_index
            )
        }
        left = grid.vertical[grid_index].coordinate_at(representative_y)
        right = grid.vertical[grid_index + 1].coordinate_at(representative_y)
        start = max(0.0, min(left, right) / width)
        end = min(1.0, max(left, right) / width)
        columns[field] = _Column(
            field=field,
            center=(start + end) / 2,
            start=start,
            end=end,
            confidence=confidence,
            grid_index=grid_index,
        )
    return _TableSchema(
        columns=columns,
        header_bottom=schema.header_bottom,
        confidence=sum(column.confidence for column in columns.values())
        / len(columns),
    )


def _has_table_header(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> bool:
    if _infer_schema(lines, grid) is not None:
        return True
    normalized = normalize_text(" ".join(line.text for line in lines))
    has_name = "nombre" in normalized and "apellido" in normalized
    supporting = sum(
        word in normalized
        for word in ("edad", "sexo", "procedencia", "plan", "telefono")
    )
    return has_name and supporting >= 1


def _headerless_name_candidates(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> list[OcrLine]:
    width = max(1, lines[0].image_width)
    candidates = [
        line
        for line in lines
        if _name_from_text(line.text)
        and not _is_header_or_metadata(line)
    ]
    if not candidates:
        return []

    clusters: list[list[OcrLine]] = []
    if grid:
        by_column: dict[int, list[OcrLine]] = {}
        for line in candidates:
            column = grid.column_for_box(line.box)
            if column is not None:
                by_column.setdefault(column, []).append(line)
        clusters.extend(by_column.values())
    if not clusters:
        for line in sorted(candidates, key=lambda item: item.box[0]):
            matching = next(
                (
                    cluster
                    for cluster in clusters
                    if abs(
                        line.box[0] - median(item.box[0] for item in cluster)
                    )
                    <= width * 0.06
                ),
                None,
            )
            if matching is None:
                clusters.append([line])
            else:
                matching.append(line)

    def column_score(cluster: list[OcrLine]) -> tuple[float, float, float]:
        names = [normalize_text(_name_from_text(line.text)) for line in cluster]
        unique_ratio = len(set(names)) / len(names)
        multiword_ratio = sum(len(name.split()) >= 2 for name in names) / len(names)
        weighted_rows = len(cluster) * (0.55 + 0.45 * unique_ratio)
        return (
            weighted_rows,
            _regular_row_ratio(cluster),
            multiword_ratio,
        )

    return max(clusters, key=column_score)


def _alignment_ratio(values: list[float], tolerance: float) -> float:
    if not values:
        return 0.0
    center = median(values)
    return sum(abs(value - center) <= tolerance for value in values) / len(values)


def _regular_row_ratio(lines: list[OcrLine]) -> float:
    centers = sorted({round(line.center_y, 1) for line in lines})
    if len(centers) < 4:
        return 0.0
    gaps = [
        current - previous
        for previous, current in zip(centers, centers[1:], strict=False)
        if current > previous
    ]
    if not gaps:
        return 0.0
    typical_gap = median(gaps)
    tolerance = max(4.0, typical_gap * 0.35)
    return sum(
        abs(gap - typical_gap) <= tolerance
        or abs(gap - typical_gap * 2) <= tolerance * 1.5
        for gap in gaps
    ) / len(gaps)


def _sequential_index_ratio(lines: list[OcrLine]) -> float:
    values = [
        int(re.sub(r"\D", "", line.text))
        for line in sorted(lines, key=lambda item: item.center_y)
    ]
    if len(values) < 2:
        return 0.0
    return sum(
        current == previous + 1
        for previous, current in zip(values, values[1:], strict=False)
    ) / (len(values) - 1)


def _has_repeated_auxiliary_column(
    lines: list[OcrLine],
    name_candidates: list[OcrLine],
    grid: TableGrid | None = None,
) -> bool:
    if not name_candidates:
        return False
    width = max(1, lines[0].image_width)
    name_ids = {id(line) for line in name_candidates}
    minimum_y = min(line.center_y for line in name_candidates)
    maximum_y = max(line.center_y for line in name_candidates)
    bins: dict[int, int] = {}
    for line in lines:
        if id(line) in name_ids or not minimum_y <= line.center_y <= maximum_y:
            continue
        if re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text):
            continue
        if not normalize_text(line.text):
            continue
        grid_column = grid.column_for_box(line.box) if grid else None
        bucket = (
            grid_column
            if grid_column is not None
            else round((line.box[0] / width) / 0.04)
        )
        bins[bucket] = bins.get(bucket, 0) + 1
    minimum_repetitions = max(3, round(len(name_candidates) * 0.30))
    return any(count >= minimum_repetitions for count in bins.values())


def looks_like_table(
    lines: list[OcrLine],
    grid: TableGrid | None = None,
) -> bool:
    if not lines:
        return False
    if _has_table_header(lines, grid):
        return True

    width = lines[0].image_width
    row_index_ids = _infer_headerless_index_ids(lines)
    sex_markers = [
        line.center_x
        for line in lines
        if re.fullmatch(r"\s*[MFH]\s*", line.text, re.IGNORECASE)
    ]
    aligned_sex_markers = (
        len(sex_markers) >= 4
        and _alignment_ratio(sex_markers, width * 0.06) >= 0.75
    )
    name_candidates = _headerless_name_candidates(lines, grid)
    document_markers = sum(
        bool(_document_digits(line.text))
        for line in lines
    )
    score = 2 if grid and grid.confidence >= 0.65 else 0
    if len(name_candidates) >= 5:
        score += 2
    if _alignment_ratio(
        [line.box[0] for line in name_candidates],
        width * 0.06,
    ) >= 0.75:
        score += 2
    if _regular_row_ratio(name_candidates) >= 0.65:
        score += 2
    if len(row_index_ids) >= 4:
        score += 2
    if aligned_sex_markers:
        score += 1
    if document_markers >= 3:
        score += 1
    if _has_repeated_auxiliary_column(lines, name_candidates, grid):
        score += 1

    return score >= 6


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
    headerless_index_ids: set[int] | None = None,
) -> list[OcrLine]:
    if schema is None:
        index_ids = headerless_index_ids or set()
        return [line for line in lines if id(line) in index_ids]

    width = max(1, anchor.line.image_width)
    age_column = schema.columns.get("age")
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


def _infer_headerless_index_ids(lines: list[OcrLine]) -> set[int]:
    if not lines:
        return set()
    width = max(1, lines[0].image_width)
    candidates = [
        line
        for line in lines
        if re.fullmatch(r"\s*\d{1,3}\s*[.):\-]?\s*", line.text)
    ]
    clusters: list[list[OcrLine]] = []
    for line in sorted(candidates, key=lambda item: item.center_x):
        matching = next(
            (
                cluster
                for cluster in clusters
                if abs(
                    line.center_x - median(item.center_x for item in cluster)
                )
                <= width * 0.04
            ),
            None,
        )
        if matching is None:
            clusters.append([line])
        else:
            matching.append(line)

    sequential = [
        cluster
        for cluster in clusters
        if len(cluster) >= 4 and _sequential_index_ratio(cluster) >= 0.70
    ]
    if not sequential:
        return set()
    selected = max(
        sequential,
        key=lambda cluster: (
            len(cluster),
            _sequential_index_ratio(cluster),
        ),
    )
    return {id(line) for line in selected}


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
    grid: TableGrid | None = None,
) -> list[_RowAnchor]:
    width = lines[0].image_width
    headerless_name_ids = (
        {id(line) for line in _headerless_name_candidates(lines, grid)}
        if schema is None
        else set()
    )
    candidates: list[_RowAnchor] = []
    for line in lines:
        if schema and "name" in schema.columns:
            name_column = schema.columns["name"]
            normalized_center = line.center_x / width
            reaches_name_column = (
                name_column.start <= normalized_center < name_column.end
            )
        else:
            reaches_name_column = id(line) in headerless_name_ids
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
    grid: TableGrid | None = None,
) -> list[tuple[_RowAnchor, list[OcrLine]]]:
    if grid:
        anchor_rows = [
            (anchor, grid.row_for_box(anchor.line.box))
            for anchor in anchors
        ]
        assigned = [
            (anchor, row)
            for anchor, row in anchor_rows
            if row is not None
        ]
        unique_rows = {row for _, row in assigned}
        if (
            len(assigned) >= max(2, round(len(anchors) * 0.70))
            and len(unique_rows) == len(assigned)
        ):
            lines_by_row: dict[int, list[OcrLine]] = {}
            for line in lines:
                row = grid.row_for_box(line.box)
                if row is not None:
                    lines_by_row.setdefault(row, []).append(line)
            return [
                (anchor, lines_by_row.get(row, []))
                for anchor, row in assigned
            ]

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


def _ocr_number(value: str) -> int | None:
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    compact = re.sub(r"^[CE]", "", compact)
    translation = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "B": "3", "S": "5"})
    compact = compact.translate(translation)
    if not compact.isdigit() or not 1 <= len(compact) <= 3:
        return None
    number = int(compact)
    return number if 0 <= number <= 115 else None


def _schema_lines(
    lines: list[OcrLine],
    schema: _TableSchema,
    field: str,
    grid: TableGrid | None = None,
) -> list[OcrLine]:
    column = schema.columns.get(field)
    if column is None:
        return []
    width = max(1, lines[0].image_width) if lines else 1
    selected: list[OcrLine] = []
    for line in lines:
        if _is_header_or_metadata(line):
            continue
        if grid and column.grid_index is not None:
            in_column = grid.column_for_box(line.box) == column.grid_index
        else:
            in_column = column.start <= line.center_x / width < column.end
        if in_column:
            selected.append(line)
    return selected


def _schema_text(
    lines: list[OcrLine],
    schema: _TableSchema,
    field: str,
    grid: TableGrid | None = None,
) -> str:
    return clean_display_text(
        " ".join(
            line.text
            for line in sorted(
                _schema_lines(lines, schema, field, grid),
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


def _extract_schema_sex(
    lines: list[OcrLine],
    *,
    allow_ocr_confusions: bool = False,
) -> _SexResult:
    candidates: dict[str, list[tuple[str, float]]] = {}
    direct_mapping = {"F": "F", "M": "M", "H": "M"}
    confusion_mapping = {"T": "F", "E": "F", "P": "F", "N": "M"}
    for line in lines:
        marker = re.sub(r"[^A-Za-z]", "", line.text).upper()
        value = direct_mapping.get(marker)
        if value is None and allow_ocr_confusions:
            value = confusion_mapping.get(marker)
        if value is not None:
            candidates.setdefault(value, []).append((marker, line.score))
    if len(candidates) > 1:
        markers = tuple(
            sorted(
                {
                    marker
                    for values in candidates.values()
                    for marker, _ in values
                }
            )
        )
        return _SexResult("", markers, conflict=True)
    if not candidates:
        return _SexResult("")

    value, values = next(iter(candidates.items()))
    canonical_present = any(marker == value for marker, _ in values)
    normalized_from = (
        ()
        if canonical_present
        else tuple(
            dict.fromkeys(
                marker
                for marker, _ in sorted(
                    values,
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        )
    )
    return _SexResult(value, normalized_from)


def _average_score(lines: list[OcrLine], fallback: float = 0.0) -> float:
    scores = [line.score for line in lines if line.text.strip()]
    return sum(scores) / len(scores) if scores else fallback


def _headerless_field_lines(
    lines: list[OcrLine],
    anchor: _RowAnchor,
    grid: TableGrid | None = None,
) -> list[OcrLine]:
    width = max(1, anchor.line.image_width)
    name_grid_column = grid.column_for_box(anchor.line.box) if grid else None
    selected: list[OcrLine] = []
    for line in lines:
        if line is anchor.line or _is_header_or_metadata(line):
            continue
        if grid and name_grid_column is not None:
            same_name_column = (
                grid.column_for_box(line.box) == name_grid_column
            )
        else:
            same_name_column = (
                bool(_name_from_text(line.text))
                and abs(line.box[0] - anchor.line.box[0]) <= width * 0.08
            )
        if not same_name_column:
            selected.append(line)
    return selected


def _extract_semantic_age(
    lines: list[OcrLine],
) -> tuple[int | None, str]:
    candidates: list[tuple[int, float, int, str]] = []
    for line in lines:
        if DATE_RE.search(line.text) or TIME_RE.search(line.text):
            continue
        if DOCUMENT_RE.search(line.text):
            continue
        normalized = normalize_text(line.text)
        match = re.fullmatch(
            r"(?:edad\s*)?(?P<age>\d{1,3})\s*"
            r"(?P<unit>anos?|a|mes(?:es)?|dias?)?",
            normalized,
        )
        if match is None:
            continue
        age = int(match.group("age"))
        if not 0 <= age <= 115:
            continue
        unit = match.group("unit") or ""
        normalized_unit = (
            "meses"
            if unit.startswith("mes")
            else "días"
            if unit.startswith("dia")
            else "años"
        )
        candidates.append((int(bool(unit)), line.score, age, normalized_unit))
    if not candidates:
        return None, ""
    candidates.sort(reverse=True)
    if (
        len(candidates) > 1
        and candidates[0][0] == 0
        and candidates[1][0] == 0
    ):
        return None, ""
    _, _, age, unit = candidates[0]
    return age, unit


def _joined_cell_text(lines: list[OcrLine]) -> str:
    return clean_display_text(
        " ".join(
            line.text for line in sorted(lines, key=lambda item: item.center_x)
        )
    )


def _grid_header_row(
    lines: list[OcrLine],
    grid: TableGrid | None,
) -> int | None:
    if grid is None:
        return None
    fields_by_row: dict[int, set[str]] = {}
    text_by_row: dict[int, list[str]] = {}
    for line in lines:
        row = grid.row_for_box(line.box)
        if row is None:
            continue
        text_by_row.setdefault(row, []).append(normalize_text(line.text))
        candidate = _header_candidate(line)
        if candidate is not None:
            fields_by_row.setdefault(row, set()).add(candidate.field)

    candidates: list[tuple[int, int]] = []
    for row, texts in text_by_row.items():
        fields = set(fields_by_row.get(row, set()))
        compact = re.sub(r"\s+", "", " ".join(texts))
        for field, aliases in HEADER_ALIASES.items():
            if any(
                re.sub(r"\s+", "", alias) in compact
                for alias in aliases
                if len(re.sub(r"\s+", "", alias)) >= 4
            ):
                fields.add(field)
        has_name = "name" in fields
        if has_name and len(fields) >= 2:
            candidates.append((len(fields), row))
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    return min(row for score, row in candidates if score == best_score)


def parse_table_lines(
    lines: list[OcrLine],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
    specialties: list[Specialty] | None = None,
    places: list[Place] | None = None,
    grid: TableGrid | None = None,
) -> list[PatientRecord] | None:
    specialties = specialties or []
    places = places or []
    if not looks_like_table(lines, grid):
        return None
    schema = _infer_schema(lines, grid)
    grid_header_row = _grid_header_row(lines, grid)
    if grid_header_row is not None and grid is not None:
        data_lines = [
            line
            for line in lines
            if (
                (row := grid.row_for_box(line.box)) is not None
                and row > grid_header_row
            )
        ]
    else:
        data_lines = (
            [line for line in lines if line.center_y > schema.header_bottom]
            if schema
            else lines
        )
    anchors = _find_row_anchors(data_lines, schema, grid)
    if len(anchors) < 2:
        return []

    headerless_index_ids = (
        _infer_headerless_index_ids(data_lines) if schema is None else set()
    )
    header_cutoff = (
        None
        if grid_header_row is not None
        else schema.header_bottom
        if schema
        else _header_cutoff(lines)
    )
    page_specialty: tuple[str, str] | None = None
    if schema:
        for line in sorted(lines, key=lambda item: item.center_y):
            if line.box[3] >= schema.header_bottom:
                continue
            if detected := detect_specialty(line.text, specialties):
                page_specialty = detected
                break
    records: list[PatientRecord] = []
    for anchor, row_lines in _row_groups(data_lines, anchors, grid):
        if header_cutoff is not None and anchor.line.center_y > header_cutoff:
            row_lines = [
                line for line in row_lines if line.center_y > header_cutoff
            ]
        raw_row_lines = list(row_lines)
        index_lines = _row_index_lines(
            row_lines,
            anchor,
            schema,
            headerless_index_ids,
        )
        row_lines = [line for line in row_lines if line not in index_lines]
        index_detected = bool(index_lines or _has_leading_index(anchor.line.text))
        if schema:
            document_lines = _schema_lines(
                row_lines, schema, "document", grid
            )
            age_lines = _schema_lines(row_lines, schema, "age", grid)
            sex_lines = _schema_lines(row_lines, schema, "sex", grid)
            origin_lines = _schema_lines(row_lines, schema, "origin", grid)
            specialty_lines = _schema_lines(
                row_lines, schema, "specialty", grid
            )
            document_id = _extract_document(document_lines)
            age = _extract_schema_age(age_lines)
            age_unit = "años" if age is not None else ""
            sex_result = _extract_schema_sex(
                sex_lines,
                allow_ocr_confusions=True,
            )
            sex = sex_result.value
            origin_text = _schema_text(
                row_lines, schema, "origin", grid
            )
            plan = _schema_text(row_lines, schema, "plan", grid)
            specialty_text = _schema_text(
                row_lines, schema, "specialty", grid
            )
        else:
            field_lines = _headerless_field_lines(row_lines, anchor, grid)
            document_id = _extract_document(field_lines)
            age, age_unit = _extract_semantic_age(field_lines)
            sex_result = _extract_schema_sex(field_lines)
            sex = sex_result.value
            semantic_text = _joined_cell_text(field_lines)
            origin_text = semantic_text
            specialty_text = semantic_text
            plan = ""
            document_lines = field_lines
            age_lines = field_lines
            sex_lines = field_lines
            origin_lines = field_lines
            specialty_lines = field_lines

        has_explicit_origin = bool(schema and "origin" in schema.columns)
        place = match_place(
            origin_text,
            places,
            contextual=has_explicit_origin,
        )
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
        if sex_result.conflict:
            notes.append(
                "Sexo ambiguo entre valores incompatibles: "
                + "/".join(sex_result.normalized_from)
            )
        elif not sex:
            notes.append("Sexo no reconocido")
        elif sex_result.normalized_from:
            notes.append(
                "Sexo normalizado desde OCR: "
                + "/".join(sex_result.normalized_from)
            )
        if not origin:
            notes.append("Procedencia no reconocida")
        elif has_explicit_origin and places and place is None:
            notes.append("Procedencia no validada en catálogo")
        elif place and place.contextual:
            notes.append(
                "Procedencia normalizada por coincidencia contextual"
            )
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
                    "cuadrícula física y encabezado; índice descartado"
                    if grid and index_detected
                    else "cuadrícula física y encabezado"
                    if grid
                    else "columna inferida por encabezado; índice descartado"
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
                "sexo": (
                    "conjunto cerrado y columna inferida por encabezado"
                    + (
                        "; normalizado desde "
                        + "/".join(sex_result.normalized_from)
                        if sex_result.normalized_from
                        else ""
                    )
                    if sex
                    else ""
                ),
                "procedencia": (
                    "cuadrícula, encabezado y catálogo geográfico contextual"
                    if grid and place and place.contextual
                    else "encabezado y catálogo geográfico contextual"
                    if place and place.contextual
                    else "cuadrícula, encabezado y catálogo geográfico"
                    if grid and place
                    else "encabezado y catálogo geográfico"
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
            document_confidence = confidence * 0.85 if document_id else 0.0
            age_confidence = confidence * 0.65 if age is not None else 0.0
            origin_confidence = (
                confidence * place.score if origin and place else 0.0
            )
            specialty_confidence = confidence * 0.75 if specialty else 0.0
            evidence = {
                "nombre": (
                    "cuadrícula física y columna repetida; índice descartado"
                    if grid and index_detected
                    else "cuadrícula física y columna repetida"
                    if grid
                    else "columna repetida y alineada; índice descartado"
                    if index_detected
                    else "columna repetida y alineada"
                ),
                "cédula": "formato de documento en la fila" if document_id else "",
                "edad": "formato de edad en la fila" if age is not None else "",
                "sexo": (
                    "valor cerrado en la fila"
                    + (
                        "; normalizado desde "
                        + "/".join(sex_result.normalized_from)
                        if sex_result.normalized_from
                        else ""
                    )
                    if sex
                    else ""
                ),
                "procedencia": (
                    "catálogo geográfico en celda física sin encabezado"
                    if grid and place
                    else "catálogo geográfico en celda sin encabezado"
                    if place
                    else ""
                ),
                "especialidad": (
                    "catálogo de especialidades en celda sin encabezado"
                    if specialty
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
                age_unit=age_unit,
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
