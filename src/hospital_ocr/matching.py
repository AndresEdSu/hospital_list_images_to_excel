from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from hospital_ocr.models import Place, Specialty
from hospital_ocr.text import normalize_text


FLOOR_RE = re.compile(r"\bpiso\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PlaceMatch:
    name: str
    score: float
    alias: str


def _word_windows(text: str, size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    ]


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
    for item in specialties:
        if len(item.alias) <= 3:
            continue
        windows = _word_windows(candidate, len(item.alias.split()))
        ratio = max(
            (
                SequenceMatcher(None, window, item.alias).ratio()
                for window in windows
            ),
            default=0.0,
        )
        if ratio >= 0.82:
            return item.specialty, item.area
    return None


def match_place(text: str, places: list[Place]) -> PlaceMatch | None:
    normalized = normalize_text(text)
    if not normalized or not places:
        return None
    for place in places:
        if re.search(rf"(?:^|\s){re.escape(place.alias)}(?:$|\s)", normalized):
            return PlaceMatch(place.name, 1.0, place.alias)

    best: PlaceMatch | None = None
    for place in places:
        if len(place.alias) < 5:
            continue
        windows = _word_windows(normalized, len(place.alias.split()))
        score = max(
            (
                SequenceMatcher(None, window, place.alias).ratio()
                for window in windows
            ),
            default=0.0,
        )
        threshold = 0.88 if len(place.alias) < 8 else 0.84
        if score >= threshold and (best is None or score > best.score):
            best = PlaceMatch(place.name, score, place.alias)
    return best
