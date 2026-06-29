from __future__ import annotations

from typing import Any

import pandas as pd

from hospital_ocr.models import PatientRecord


EDITABLE_COLUMNS = [
    "nombre_completo",
    "nombre",
    "apellido",
    "cedula",
    "centro",
    "edad",
    "unidad_edad",
    "sexo",
    "procedencia",
    "especialidad",
    "area",
    "estado_revision",
    "observaciones",
]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _age(value: Any) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    age = int(float(value))
    if not 0 <= age <= 115:
        raise ValueError(f"Edad fuera del rango permitido: {age}")
    return age


def apply_patient_edits(
    records: list[PatientRecord],
    edited: pd.DataFrame,
) -> None:
    if "id_paciente" not in edited.columns:
        raise ValueError("La tabla editada no contiene id_paciente")
    by_id = {record.patient_id: record for record in records}
    for row in edited.to_dict(orient="records"):
        patient_id = _text(row.get("id_paciente"))
        record = by_id.get(patient_id)
        if record is None:
            continue
        record.full_name = _text(row.get("nombre_completo"))
        record.first_name = _text(row.get("nombre"))
        record.last_name = _text(row.get("apellido"))
        record.document_id = _text(row.get("cedula"))
        record.center = _text(row.get("centro"))
        record.age = _age(row.get("edad"))
        record.age_unit = _text(row.get("unidad_edad"))
        record.sex = _text(row.get("sexo")).upper()
        record.origin = _text(row.get("procedencia"))
        record.specialty = _text(row.get("especialidad"))
        record.area = _text(row.get("area"))
        record.review_status = _text(row.get("estado_revision"))
        observations = _text(row.get("observaciones"))
        record.clinical_notes = ""
        record.notes = [
            note.strip() for note in observations.split(";") if note.strip()
        ]
        record.needs_review = record.review_status == "Pendiente"
