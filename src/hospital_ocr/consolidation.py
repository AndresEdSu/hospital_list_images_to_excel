from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher

from hospital_ocr.models import (
    ConsolidationResult,
    EvidenceRecord,
    PatientRecord,
)
from hospital_ocr.name_splitter import NameLexicons, normalize_identity_text
from hospital_ocr.text import normalize_text


NAME_DUPLICATE_THRESHOLD = 0.92
NAME_CANONICAL_TOKEN_THRESHOLD = 0.90
NAME_CANONICAL_SHORT_TOKEN_THRESHOLD = 0.94
NAME_CANONICAL_MIN_MARGIN = 0.03


@dataclass(frozen=True)
class _NameKey:
    normalized: str
    canonical: str

    @property
    def was_canonicalized(self) -> bool:
        return self.normalized != self.canonical


def _normalized_document(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _name_terms(lexicons: NameLexicons) -> dict[str, float]:
    terms: dict[str, float] = {}
    for catalog in (lexicons.given_names, lexicons.surnames):
        for term, weight in catalog.items():
            normalized = normalize_text(term)
            if normalized:
                terms[normalized] = max(terms.get(normalized, 0.0), weight)
    return terms


def _name_ocr_variants(token: str) -> set[str]:
    variants = {token}
    replacements = (
        ("0", "o"),
        ("1", "i"),
        ("w", "u"),
        ("u", "v"),
        ("v", "u"),
    )
    for old, new in replacements:
        if old in token:
            variants.add(token.replace(old, new))
    if token.startswith("i") and len(token) >= 4:
        variants.add("j" + token[1:])
    if token.startswith("j") and len(token) >= 4:
        variants.add("i" + token[1:])
    return variants


def _canonical_name_token(token: str, terms: dict[str, float]) -> str:
    if token in terms or len(token) < 4:
        return token

    best_term = token
    best_score = 0.0
    second_score = 0.0
    variants = _name_ocr_variants(token)
    for term, weight in terms.items():
        if len(term) < 4 or abs(len(term) - len(token)) > 2:
            continue
        term_best = max(
            SequenceMatcher(None, variant, term).ratio()
            for variant in variants
        ) * max(0.75, min(weight, 1.0))
        if term_best > best_score:
            second_score = best_score
            best_score = term_best
            best_term = term
        elif term_best > second_score:
            second_score = term_best

    threshold = (
        NAME_CANONICAL_SHORT_TOKEN_THRESHOLD
        if len(token) <= 5
        else NAME_CANONICAL_TOKEN_THRESHOLD
    )
    if (
        best_score >= threshold
        and best_score - second_score >= NAME_CANONICAL_MIN_MARGIN
    ):
        return best_term
    return token


def _name_key(
    full_name: str,
    lexicons: NameLexicons | None,
    terms: dict[str, float] | None,
    cache: dict[str, _NameKey],
) -> _NameKey:
    normalized = normalize_text(full_name)
    cached = cache.get(normalized)
    if cached is not None:
        return cached
    if not normalized or lexicons is None or terms is None:
        key = _NameKey(normalized=normalized, canonical=normalized)
        cache[normalized] = key
        return key

    segmented = normalize_text(
        normalize_identity_text(full_name, lexicons, role="mixed")
    )
    canonical = " ".join(
        _canonical_name_token(token, terms)
        for token in segmented.split()
    )
    key = _NameKey(normalized=normalized, canonical=canonical)
    cache[normalized] = key
    return key


def _compatible_identity(left: PatientRecord, right: PatientRecord) -> bool:
    if left.center != right.center:
        return False
    left_document = _normalized_document(left.document_id)
    right_document = _normalized_document(right.document_id)
    if left_document and right_document:
        return left_document == right_document
    if left.age is None or right.age is None or left.age != right.age:
        return False
    if left.age_unit and right.age_unit and left.age_unit != right.age_unit:
        return False
    if normalize_text(left.full_name) != normalize_text(right.full_name):
        return False
    if left.sex and right.sex and left.sex != right.sex:
        return False
    if (
        left.origin
        and right.origin
        and normalize_text(left.origin) != normalize_text(right.origin)
    ):
        return False
    return True


def _possible_duplicate(
    left: PatientRecord,
    right: PatientRecord,
    *,
    name_lexicons: NameLexicons | None = None,
    name_terms: dict[str, float] | None = None,
    name_cache: dict[str, _NameKey] | None = None,
) -> tuple[float, str] | None:
    if left.center != right.center:
        return None
    left_document = _normalized_document(left.document_id)
    right_document = _normalized_document(right.document_id)
    if left_document and right_document and left_document != right_document:
        return None

    if name_cache is None:
        name_cache = {}
    left_key = _name_key(
        left.full_name,
        name_lexicons,
        name_terms,
        name_cache,
    )
    right_key = _name_key(
        right.full_name,
        name_lexicons,
        name_terms,
        name_cache,
    )
    left_name = left_key.normalized
    right_name = right_key.normalized
    left_canonical = left_key.canonical
    right_canonical = right_key.canonical
    if min(len(left_canonical), len(right_canonical)) < 8:
        return None

    raw_similarity = SequenceMatcher(None, left_name, right_name).ratio()
    canonical_similarity = SequenceMatcher(
        None,
        left_canonical,
        right_canonical,
    ).ratio()
    canonical_helped = (
        (left_key.was_canonicalized or right_key.was_canonicalized)
        and canonical_similarity > raw_similarity
    )
    name_similarity = max(raw_similarity, canonical_similarity)
    exact_name = left_name == right_name or left_canonical == right_canonical
    if left.age is None or right.age is None:
        if left_name == right_name:
            return raw_similarity, "mismo nombre | edad incompleta"
        return None
    if left.age_unit != right.age_unit or left.age != right.age:
        return None
    if not exact_name and name_similarity < NAME_DUPLICATE_THRESHOLD:
        return None

    name_reason = "nombre normalizado" if canonical_helped else "nombre"
    reasons = [f"{name_reason} {name_similarity:.0%}", "misma edad"]
    if left.sex and right.sex and left.sex != right.sex:
        reasons.append("sexo diferente")
    if (
        left.origin
        and right.origin
        and normalize_text(left.origin) != normalize_text(right.origin)
    ):
        reasons.append("procedencia diferente")
    return name_similarity, " | ".join(reasons)


def _merge_value(
    target: PatientRecord,
    field_name: str,
    incoming: PatientRecord,
    label: str,
) -> None:
    confidence_fields = {
        "document_id": ("document_confidence", "cédula"),
        "origin": ("origin_confidence", "procedencia"),
        "specialty": ("specialty_confidence", "especialidad"),
    }
    current = getattr(target, field_name)
    candidate = getattr(incoming, field_name)
    if not current and candidate:
        setattr(target, field_name, candidate)
        if field_name in confidence_fields:
            confidence_field, evidence_key = confidence_fields[field_name]
            setattr(
                target,
                confidence_field,
                getattr(incoming, confidence_field),
            )
            if incoming.field_evidence.get(evidence_key):
                target.field_evidence[evidence_key] = (
                    incoming.field_evidence[evidence_key]
                )
    elif (
        current
        and candidate
        and normalize_text(str(current)) == normalize_text(str(candidate))
        and field_name in confidence_fields
    ):
        confidence_field, evidence_key = confidence_fields[field_name]
        if getattr(incoming, confidence_field) > getattr(target, confidence_field):
            setattr(
                target,
                confidence_field,
                getattr(incoming, confidence_field),
            )
            if incoming.field_evidence.get(evidence_key):
                target.field_evidence[evidence_key] = (
                    incoming.field_evidence[evidence_key]
                )
    elif current and candidate and normalize_text(str(current)) != normalize_text(
        str(candidate)
    ):
        target.needs_review = True
        target.add_note(f"Conflicto de {label}: {current} / {candidate}")


def _remove_resolved_notes(record: PatientRecord) -> None:
    resolved = {
        "Sexo no reconocido": bool(record.sex),
        "Procedencia no reconocida": bool(record.origin),
        "Edad no reconocida": record.age is not None,
        "Especialidad o área no reconocida": bool(record.specialty),
        "Nombre incompleto": bool(record.first_name and record.last_name),
    }
    record.notes = [
        note for note in record.notes if not resolved.get(note, False)
    ]
    record.needs_review = bool(record.notes)


def _merge_record(target: PatientRecord, incoming: PatientRecord) -> None:
    target.occurrences += 1
    target.confidence = max(target.confidence, incoming.confidence)
    for image in incoming.source_images or [incoming.source_image]:
        if image not in target.source_images:
            target.source_images.append(image)
    _merge_value(target, "sex", incoming, "sexo")
    _merge_value(target, "document_id", incoming, "cédula")
    _merge_value(target, "origin", incoming, "procedencia")
    _merge_value(target, "specialty", incoming, "especialidad")
    _merge_value(target, "area", incoming, "área")
    target_document = _normalized_document(target.document_id)
    incoming_document = _normalized_document(incoming.document_id)
    same_document = (
        bool(target_document)
        and bool(incoming_document)
        and target_document == incoming_document
    )
    if (
        same_document
        and normalize_text(target.full_name) != normalize_text(incoming.full_name)
    ):
        target.needs_review = True
        target.add_note("Nombre OCR diferente para la misma cédula")
    if (
        same_document
        and target.age is not None
        and incoming.age is not None
        and target.age != incoming.age
    ):
        target.needs_review = True
        target.add_note("Edad diferente para la misma cédula")
    if incoming.clinical_notes:
        if not target.clinical_notes:
            target.clinical_notes = incoming.clinical_notes
        elif incoming.clinical_notes not in target.clinical_notes:
            target.clinical_notes = (
                f"{target.clinical_notes} | {incoming.clinical_notes}"
            )
    for note in incoming.notes:
        target.add_note(note)
    _remove_resolved_notes(target)


def _mark_possible_duplicates(
    records: list[PatientRecord],
    name_lexicons: NameLexicons | None = None,
) -> None:
    name_terms = _name_terms(name_lexicons) if name_lexicons is not None else None
    name_cache: dict[str, _NameKey] = {}
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            candidate = _possible_duplicate(
                left,
                right,
                name_lexicons=name_lexicons,
                name_terms=name_terms,
                name_cache=name_cache,
            )
            if candidate:
                _, reason = candidate
                left.needs_review = True
                right.needs_review = True
                left.add_note("Posible duplicado no fusionado")
                right.add_note("Posible duplicado no fusionado")
                left.duplicate_status = "Posible duplicado"
                right.duplicate_status = "Posible duplicado"
                left.add_duplicate_detail(
                    f"Coincide con {right.patient_id} | {reason} | no fusionado"
                )
                right.add_duplicate_detail(
                    f"Coincide con {left.patient_id} | {reason} | no fusionado"
                )


def consolidate_records(
    records: list[PatientRecord],
    name_lexicons: NameLexicons | None = None,
) -> ConsolidationResult:
    consolidated: list[PatientRecord] = []
    evidence_links: list[tuple[int, PatientRecord, PatientRecord]] = []
    first_occurrence: dict[int, int] = {}

    indexed_records = list(enumerate(records))
    for original_index, incoming in sorted(
        indexed_records,
        key=lambda item: (-item[1].confidence, item[0]),
    ):
        snapshot = deepcopy(incoming)
        if not incoming.source_images:
            incoming.source_images = [incoming.source_image]
        match = next(
            (
                existing
                for existing in consolidated
                if _compatible_identity(existing, incoming)
            ),
            None,
        )
        if match is None:
            consolidated.append(incoming)
            first_occurrence[id(incoming)] = original_index
            evidence_links.append((original_index, snapshot, incoming))
        else:
            _merge_record(match, incoming)
            first_occurrence[id(match)] = min(
                first_occurrence[id(match)],
                original_index,
            )
            evidence_links.append((original_index, snapshot, match))

    consolidated.sort(key=lambda item: first_occurrence[id(item)])
    for index, record in enumerate(consolidated, start=1):
        record.patient_id = f"PAC-{index:04d}"
        ordered_images: list[str] = []
        for _, snapshot, canonical in sorted(
            evidence_links,
            key=lambda item: item[0],
        ):
            if canonical is not record:
                continue
            for image in snapshot.source_images or [snapshot.source_image]:
                if image not in ordered_images:
                    ordered_images.append(image)
        record.source_images = ordered_images
        if record.occurrences > 1:
            record.duplicate_status = "Duplicado consolidado"
            image_count = len(record.source_images)
            record.duplicate_detail = (
                f"{record.occurrences} apariciones consolidadas "
                f"en {image_count} imagen(es)"
            )
    _mark_possible_duplicates(consolidated, name_lexicons)

    evidence = [
        EvidenceRecord(
            evidence_id=f"EVI-{index:05d}",
            patient_id=canonical.patient_id,
            record=snapshot,
        )
        for index, (_, snapshot, canonical) in enumerate(
            sorted(evidence_links, key=lambda item: item[0]),
            start=1,
        )
    ]
    return ConsolidationResult(consolidated, evidence)
