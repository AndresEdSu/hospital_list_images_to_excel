from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from hospital_ocr.text import normalize_text


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
    return 0.70


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
