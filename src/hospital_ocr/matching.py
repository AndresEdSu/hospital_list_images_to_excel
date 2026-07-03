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
    contextual: bool = False
    runner_up_score: float = 0.0


def _word_windows(text: str, size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    ]


def _alias_windows(text: str, alias: str) -> list[str]:
    alias_size = len(alias.split())
    sizes = {
        max(1, alias_size - 1),
        alias_size,
        alias_size + 1,
    }
    return list(
        dict.fromkeys(
            window
            for size in sorted(sizes)
            for window in _word_windows(text, size)
        )
    )


def _compact(value: str) -> str:
    return value.replace(" ", "")


def _strong_alias_score(text: str, alias: str) -> float:
    compact_alias = _compact(alias)
    if compact_alias == "mi":
        return 1.0 if _compact(text) == compact_alias else 0.0
    if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", text):
        return 1.0
    if len(compact_alias) < 5:
        return 0.0
    if any(_compact(window) == compact_alias for window in _alias_windows(text, alias)):
        return 0.98
    return 0.0


def _fuzzy_alias_score(text: str, alias: str) -> float:
    standard_score = max(
        (
            SequenceMatcher(None, window, alias).ratio()
            for window in _word_windows(text, len(alias.split()))
        ),
        default=0.0,
    )
    compact_alias = _compact(alias)
    compact_score = max(
        (
            SequenceMatcher(None, _compact(window), compact_alias).ratio()
            for window in _alias_windows(text, alias)
        ),
        default=0.0,
    )
    if compact_score < 0.90:
        compact_score = 0.0
    return max(standard_score, compact_score * 0.98)


def _specialty_result(
    item: Specialty,
    normalized_text: str,
) -> tuple[str, str]:
    area = item.area
    floor = FLOOR_RE.search(normalized_text)
    if floor:
        floor_text = f"Piso {floor.group(1)}"
        area = f"{area} - {floor_text}" if area else floor_text
    return item.specialty, area


def detect_specialty(
    text: str, specialties: list[Specialty]
) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    strong = [
        (item, _strong_alias_score(normalized, item.alias))
        for item in specialties
    ]
    strong = [(item, score) for item, score in strong if score]
    if strong:
        item, _ = max(
            strong,
            key=lambda candidate: (
                len(_compact(candidate[0].alias)),
                candidate[1],
            ),
        )
        return _specialty_result(item, normalized)

    candidate = re.sub(r"\b(?:piso\s*)?\d+\b", "", normalized).strip()
    best: tuple[Specialty, float] | None = None
    for item in specialties:
        if len(item.alias) <= 3:
            continue
        score = _fuzzy_alias_score(candidate, item.alias)
        if score >= 0.82 and (
            best is None
            or (score, len(_compact(item.alias)))
            > (best[1], len(_compact(best[0].alias)))
        ):
            best = (item, score)
    if best:
        return _specialty_result(best[0], normalized)
    return None


def _rank_place_matches(
    normalized_text: str,
    places: list[Place],
) -> list[PlaceMatch]:
    by_place: dict[str, PlaceMatch] = {}
    for place in places:
        strong_score = _strong_alias_score(normalized_text, place.alias)
        fuzzy_score = (
            _fuzzy_alias_score(normalized_text, place.alias)
            if len(place.alias) >= 5
            else 0.0
        )
        score = max(strong_score, fuzzy_score)
        if score <= 0:
            continue
        candidate = PlaceMatch(place.name, score, place.alias)
        current = by_place.get(place.name)
        if current is None or (
            candidate.score,
            len(_compact(candidate.alias)),
        ) > (
            current.score,
            len(_compact(current.alias)),
        ):
            by_place[place.name] = candidate
    return sorted(
        by_place.values(),
        key=_place_rank_key,
        reverse=True,
    )


def _place_rank_key(candidate: PlaceMatch) -> tuple[float, float, float]:
    if candidate.score >= 0.98:
        return 1.0, float(len(_compact(candidate.alias))), candidate.score
    return 0.0, candidate.score, float(len(_compact(candidate.alias)))


def match_place(
    text: str,
    places: list[Place],
    *,
    contextual: bool = False,
) -> PlaceMatch | None:
    normalized = normalize_text(text)
    if not normalized or not places:
        return None
    ranked = _rank_place_matches(normalized, places)
    if not ranked:
        return None
    best = ranked[0]
    runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0

    if best.score >= 0.98:
        return PlaceMatch(
            best.name,
            best.score,
            best.alias,
            runner_up_score=runner_up_score,
        )

    strict_threshold = 0.88 if len(best.alias) < 8 else 0.84
    if best.score >= strict_threshold:
        return PlaceMatch(
            best.name,
            best.score,
            best.alias,
            runner_up_score=runner_up_score,
        )

    contextual_threshold = 0.80 if len(best.alias) < 8 else 0.78
    minimum_margin = 0.06 if best.score >= 0.82 else 0.08
    if (
        contextual
        and best.score >= contextual_threshold
        and best.score - runner_up_score >= minimum_margin
    ):
        return PlaceMatch(
            best.name,
            best.score,
            best.alias,
            contextual=True,
            runner_up_score=runner_up_score,
        )
    return None
