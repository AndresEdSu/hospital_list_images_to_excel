from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from hospital_ocr.text import clean_display_text, normalize_text


SURNAME_CONNECTORS = {"de", "del", "la", "las", "los", "y"}
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]+")


@dataclass(frozen=True)
class NameLexicons:
    given_names: dict[str, float]
    surnames: dict[str, float]


@dataclass(frozen=True)
class NameSplit:
    first_name: str
    last_name: str
    confidence: float
    detected_order: str
    reliable: bool


@dataclass(frozen=True)
class _Candidate:
    first_name: str
    last_name: str
    score: float
    order: str


@dataclass(frozen=True)
class _IdentitySegment:
    text: str
    score: float
    known: bool


def _load_terms(path: Path) -> dict[str, float]:
    terms: dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            term = normalize_text(row.get("termino") or "")
            if not term:
                continue
            weight = float(row.get("peso") or 1.0)
            terms[term] = max(0.0, min(weight, 1.0))
    if not terms:
        raise ValueError(f"El catálogo está vacío: {path}")
    return terms


def load_name_lexicons(given_names_path: Path, surnames_path: Path) -> NameLexicons:
    return NameLexicons(
        given_names=_load_terms(given_names_path),
        surnames=_load_terms(surnames_path),
    )


def _token_score(token: str, role: str, lexicons: NameLexicons) -> float:
    normalized = normalize_text(token)
    if role == "surname" and normalized in SURNAME_CONNECTORS:
        return 1.0
    if role == "given":
        expected = lexicons.given_names.get(normalized)
        opposite = lexicons.surnames.get(normalized)
    else:
        expected = lexicons.surnames.get(normalized)
        opposite = lexicons.given_names.get(normalized)
    if expected is not None and opposite is not None:
        return expected * 0.75
    if expected is not None:
        return expected
    if opposite is not None:
        return 0.0
    if len(normalized) >= 4:
        expected_terms = (
            lexicons.given_names if role == "given" else lexicons.surnames
        )
        opposite_terms = (
            lexicons.surnames if role == "given" else lexicons.given_names
        )
        expected_fuzzy = max(
            (
                SequenceMatcher(None, normalized, term).ratio() * weight
                for term, weight in expected_terms.items()
                if abs(len(term) - len(normalized)) <= 2
            ),
            default=0.0,
        )
        opposite_fuzzy = max(
            (
                SequenceMatcher(None, normalized, term).ratio() * weight
                for term, weight in opposite_terms.items()
                if abs(len(term) - len(normalized)) <= 2
            ),
            default=0.0,
        )
        if expected_fuzzy >= 0.84 and expected_fuzzy > opposite_fuzzy + 0.05:
            return min(0.95, expected_fuzzy)
        if opposite_fuzzy >= 0.84 and opposite_fuzzy > expected_fuzzy + 0.05:
            return 0.10
    return 0.70


def _format_matched_term(term: str, raw: str) -> str:
    if raw.isupper():
        return term.upper()
    if raw[:1].isupper():
        return term.title()
    return term


def _exact_or_ocr_match(
    normalized: str,
    raw: str,
    role: str,
    lexicons: NameLexicons,
) -> _IdentitySegment | None:
    term_groups: list[dict[str, float]] = []
    if role in {"given", "mixed"}:
        term_groups.append(lexicons.given_names)
    if role in {"surname", "mixed"}:
        term_groups.append(lexicons.surnames)

    if role in {"surname", "mixed"} and normalized in SURNAME_CONNECTORS:
        return _IdentitySegment(raw, 0.95, False)

    variants = [normalized]
    if normalized.startswith("i") and len(normalized) >= 4:
        variants.append("j" + normalized[1:])
    if "0" in normalized:
        variants.append(normalized.replace("0", "o"))
    if "1" in normalized:
        variants.append(normalized.replace("1", "i"))

    for variant in dict.fromkeys(variants):
        for terms in term_groups:
            weight = terms.get(variant)
            if weight is None:
                continue
            text = raw if variant == normalized else _format_matched_term(
                variant,
                raw,
            )
            score = weight if variant == normalized else min(0.92, weight)
            return _IdentitySegment(text, score, True)
    return None


def _segment_piece(
    raw: str,
    normalized: str,
    role: str,
    lexicons: NameLexicons,
) -> _IdentitySegment | None:
    matched = _exact_or_ocr_match(normalized, raw, role, lexicons)
    if matched is not None:
        return matched
    if len(normalized) >= 4:
        return _IdentitySegment(raw, 0.55, False)
    return None


def _split_glued_token(
    token: str,
    role: str,
    lexicons: NameLexicons,
) -> list[str]:
    normalized = normalize_text(token).replace(" ", "")
    if len(normalized) < 7 or len(normalized) != len(token):
        return [token]
    if _exact_or_ocr_match(normalized, token, role, lexicons) is not None:
        return [token]

    best: tuple[float, int, int, list[_IdentitySegment]] | None = None
    max_pieces = 4
    max_piece_length = 14

    def visit(start: int, pieces: list[_IdentitySegment]) -> None:
        nonlocal best
        if start == len(normalized):
            if len(pieces) < 2 or not any(piece.known for piece in pieces):
                return
            if not any(
                piece.known and len(normalize_text(piece.text)) >= 4
                for piece in pieces
            ):
                return
            score = sum(piece.score for piece in pieces) / len(pieces)
            if score < 0.72:
                return
            candidate = (
                score,
                sum(piece.known for piece in pieces),
                -len(pieces),
                pieces.copy(),
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
            return
        if len(pieces) >= max_pieces:
            return

        remaining_slots = max_pieces - len(pieces) - 1
        for end in range(start + 2, min(len(normalized), start + max_piece_length) + 1):
            remaining = len(normalized) - end
            if remaining and remaining > remaining_slots * max_piece_length:
                continue
            raw_piece = token[start:end]
            normalized_piece = normalized[start:end]
            piece = _segment_piece(raw_piece, normalized_piece, role, lexicons)
            if piece is None:
                continue
            visit(end, [*pieces, piece])

    visit(0, [])
    return [piece.text for piece in best[3]] if best is not None else [token]


def normalize_identity_text(
    text: str,
    lexicons: NameLexicons,
    *,
    role: str,
) -> str:
    if role not in {"given", "surname", "mixed"}:
        raise ValueError(f"Rol de identidad invÃ¡lido: {role}")
    parts: list[str] = []
    for token in TOKEN_RE.findall(text):
        parts.extend(_split_glued_token(token, role, lexicons))
    return clean_display_text(" ".join(parts))


def _candidate_score(
    first_name_tokens: list[str],
    last_name_tokens: list[str],
    lexicons: NameLexicons,
) -> float:
    scores = [
        *(_token_score(token, "given", lexicons) for token in first_name_tokens),
        *(_token_score(token, "surname", lexicons) for token in last_name_tokens),
    ]
    return sum(scores) / len(scores)


def split_full_name(
    full_name: str,
    lexicons: NameLexicons,
    minimum_confidence: float = 0.85,
    minimum_margin: float = 0.10,
) -> NameSplit:
    tokens = TOKEN_RE.findall(full_name)
    if len(tokens) < 2:
        return NameSplit("", "", 0.0, "Indeterminado", False)

    candidates: list[_Candidate] = []
    for split_at in range(1, len(tokens)):
        names_first = tokens[:split_at]
        surnames_last = tokens[split_at:]
        candidates.append(
            _Candidate(
                first_name=" ".join(names_first),
                last_name=" ".join(surnames_last),
                score=_candidate_score(names_first, surnames_last, lexicons),
                order="Nombre-Apellido",
            )
        )

        surnames_first = tokens[:split_at]
        names_last = tokens[split_at:]
        candidates.append(
            _Candidate(
                first_name=" ".join(names_last),
                last_name=" ".join(surnames_first),
                score=_candidate_score(names_last, surnames_first, lexicons),
                order="Apellido-Nombre",
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    reliable = (
        best.score >= minimum_confidence
        and best.score - second_score >= minimum_margin
    )
    if not reliable:
        return NameSplit("", "", round(best.score, 4), "Indeterminado", False)
    return NameSplit(
        best.first_name,
        best.last_name,
        round(best.score, 4),
        best.order,
        True,
    )
