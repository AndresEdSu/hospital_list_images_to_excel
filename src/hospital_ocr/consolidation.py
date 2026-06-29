from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher

from hospital_ocr.models import (
    ConsolidationResult,
    EvidenceRecord,
    PatientRecord,
)
from hospital_ocr.text import normalize_text


def _normalized_document(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


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
    left: PatientRecord, right: PatientRecord
) -> tuple[float, str] | None:
    if left.center != right.center:
        return None
    left_document = _normalized_document(left.document_id)
    right_document = _normalized_document(right.document_id)
    if left_document and right_document and left_document != right_document:
        return None
    left_name = normalize_text(left.full_name)
    right_name = normalize_text(right.full_name)
    if min(len(left_name), len(right_name)) < 8:
        return None
    name_similarity = SequenceMatcher(None, left_name, right_name).ratio()
    exact_name = left_name == right_name
    if left.age is None or right.age is None:
        if exact_name:
            return name_similarity, "mismo nombre | edad incompleta"
        return None
    if left.age_unit != right.age_unit or left.age != right.age:
        return None
    if not exact_name and name_similarity < 0.92:
        return None

    reasons = [f"nombre {name_similarity:.0%}", "misma edad"]
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


def _mark_possible_duplicates(records: list[PatientRecord]) -> None:
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            candidate = _possible_duplicate(left, right)
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


def consolidate_records(records: list[PatientRecord]) -> ConsolidationResult:
    consolidated: list[PatientRecord] = []
    evidence_links: list[tuple[PatientRecord, PatientRecord]] = []

    for incoming in sorted(records, key=lambda item: item.confidence, reverse=True):
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
            evidence_links.append((snapshot, incoming))
        else:
            _merge_record(match, incoming)
            evidence_links.append((snapshot, match))

    consolidated.sort(
        key=lambda item: (
            item.center,
            item.specialty,
            item.area,
            normalize_text(item.full_name),
        )
    )
    for index, record in enumerate(consolidated, start=1):
        record.patient_id = f"PAC-{index:04d}"
        if record.occurrences > 1:
            record.duplicate_status = "Duplicado consolidado"
            image_count = len(record.source_images)
            record.duplicate_detail = (
                f"{record.occurrences} apariciones consolidadas "
                f"en {image_count} imagen(es)"
            )
    _mark_possible_duplicates(consolidated)

    evidence = [
        EvidenceRecord(
            evidence_id=f"EVI-{index:05d}",
            patient_id=canonical.patient_id,
            record=snapshot,
        )
        for index, (snapshot, canonical) in enumerate(evidence_links, start=1)
    ]
    return ConsolidationResult(consolidated, evidence)
