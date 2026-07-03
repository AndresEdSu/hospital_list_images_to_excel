from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from hospital_ocr.matching import detect_specialty
from hospital_ocr.models import OcrLine, Specialty, TableGrid
from hospital_ocr.table_extraction.common import (
    document_digits,
    name_from_text,
    semantic_age_tokens,
)
from hospital_ocr.text import normalize_text


@dataclass(frozen=True)
class SectionHeading:
    line: OcrLine
    specialty: str
    area: str
    line_ids: frozenset[int]


def _is_area_descriptor(line: OcrLine) -> bool:
    normalized = normalize_text(line.text)
    return bool(
        re.fullmatch(
            r"(?:piso\s*\d+|emergencia|uci|ucip|uti|"
            r"sala(?:\s+\w+){0,2})",
            normalized,
        )
    )


def _has_patient_fields(line: OcrLine) -> bool:
    return bool(
        document_digits(line.text)
        or semantic_age_tokens(line.text)
        or re.search(
            r"(?<![A-Za-z])[MFH](?![A-Za-z])",
            line.text,
            re.IGNORECASE,
        )
    )


def _shares_grid_row_with_patient(
    line: OcrLine,
    lines: list[OcrLine],
    grid: TableGrid | None,
) -> bool:
    if grid is None:
        return False
    row = grid.row_for_box(line.box)
    if row is None:
        return False
    companions = [
        candidate
        for candidate in lines
        if candidate is not line
        and grid.row_for_box(candidate.box) == row
        and candidate.text.strip()
    ]
    if any(_has_patient_fields(candidate) for candidate in companions):
        return True
    return any(
        len(name_from_text(candidate.text).split()) >= 2
        and not _is_area_descriptor(candidate)
        for candidate in companions
    )


def _lenient_specialty_heading(
    line: OcrLine,
    specialties: list[Specialty],
) -> tuple[str, str] | None:
    normalized = normalize_text(line.text)
    if not normalized or len(normalized.split()) > 3:
        return None
    best: tuple[Specialty, float] | None = None
    for specialty in specialties:
        alias = normalize_text(specialty.alias)
        if len(alias) < 7:
            continue
        score = SequenceMatcher(None, normalized, alias).ratio()
        if score >= 0.75 and (
            best is None
            or (score, len(alias)) > (best[1], len(best[0].alias))
        ):
            best = specialty, score
    if best is None:
        return None
    return best[0].specialty, best[0].area


def _nearby_descriptors(
    heading: OcrLine,
    lines: list[OcrLine],
) -> list[OcrLine]:
    heading_height = max(1, heading.box[3] - heading.box[1])
    descriptors: list[OcrLine] = []
    for line in lines:
        if line is heading or not _is_area_descriptor(line):
            continue
        line_height = max(1, line.box[3] - line.box[1])
        tolerance = max(heading_height, line_height) * 0.85
        if abs(line.center_y - heading.center_y) <= tolerance:
            descriptors.append(line)
    return descriptors


def find_section_headings(
    lines: list[OcrLine],
    specialties: list[Specialty],
    grid: TableGrid | None = None,
) -> list[SectionHeading]:
    headings: list[SectionHeading] = []
    consumed: set[int] = set()
    for line in sorted(lines, key=lambda item: item.center_y):
        if (
            id(line) in consumed
            or _has_patient_fields(line)
            or _shares_grid_row_with_patient(line, lines, grid)
        ):
            continue
        detected = detect_specialty(
            line.text,
            specialties,
        ) or _lenient_specialty_heading(line, specialties)
        if detected is None:
            continue

        descriptors = _nearby_descriptors(line, lines)
        members = [line, *descriptors]
        combined_text = " ".join(
            member.text
            for member in sorted(
                members,
                key=lambda item: (item.center_y, item.center_x),
            )
        )
        specialty, area = detect_specialty(
            combined_text,
            specialties,
        ) or detected
        normalized_combined = normalize_text(combined_text)
        normalized_area = normalize_text(area)
        if (
            "emergencia" in normalized_combined
            and "emergencia" not in normalized_area
        ):
            area = f"{area} - Emergencia" if area else "Emergencia"

        member_ids = frozenset(id(member) for member in members)
        consumed.update(member_ids)
        headings.append(
            SectionHeading(
                line=line,
                specialty=specialty,
                area=area,
                line_ids=member_ids,
            )
        )
    return headings


def section_for_line(
    line: OcrLine,
    headings: list[SectionHeading],
) -> SectionHeading | None:
    candidates = [
        heading
        for heading in headings
        if heading.line.center_y <= line.center_y
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda heading: heading.line.center_y)
