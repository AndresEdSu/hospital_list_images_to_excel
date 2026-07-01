import pandas as pd

from hospital_ocr.editing import apply_patient_edits
from hospital_ocr.models import PatientRecord


def test_patient_edits_are_applied_before_export() -> None:
    patient = PatientRecord(
        full_name="María Perez",
        first_name="María",
        last_name="Perez",
        name_split_confidence=0.9,
        detected_name_order="Nombre-Apellido",
        center="Hospital de Prueba",
        age=8,
        age_unit="años",
        sex="",
        origin="",
        specialty="Pediatría",
        area="",
        source_image="lista.jpg",
        confidence=0.8,
        needs_review=True,
        notes=["Sexo no reconocido"],
        patient_id="PAC-0001",
    )
    edited = pd.DataFrame(
        [
            {
                "id_paciente": "PAC-0001",
                "nombre_completo": "María Pérez",
                "nombre": "María",
                "apellido": "Pérez",
                "cedula": "V-12345678",
                "centro": "Hospital de Prueba",
                "edad": 8,
                "unidad_edad": "años",
                "sexo": "F",
                "procedencia": "Petare",
                "especialidad": "Pediatría",
                "area": "UCI",
                "estado_revision": "Revisado",
                "observaciones": "",
            }
        ]
    )

    apply_patient_edits([patient], edited)

    assert patient.full_name == "María Pérez"
    assert patient.document_id == "V-12345678"
    assert patient.sex == "F"
    assert patient.origin == "Petare"
    assert patient.area == "UCI"
    assert patient.review_status == "Revisado"
    assert patient.needs_review is False
    assert patient.notes == []


def test_invalid_age_is_rejected() -> None:
    patient = PatientRecord(
        full_name="María Pérez",
        first_name="María",
        last_name="Pérez",
        name_split_confidence=1.0,
        detected_name_order="Nombre-Apellido",
        center="Hospital de Prueba",
        age=8,
        age_unit="años",
        sex="F",
        origin="Petare",
        specialty="Pediatría",
        area="",
        source_image="lista.jpg",
        confidence=0.9,
        needs_review=False,
        patient_id="PAC-0001",
    )
    edited = pd.DataFrame(
        [
            {
                "id_paciente": "PAC-0001",
                "nombre_completo": "María Pérez",
                "nombre": "María",
                "apellido": "Pérez",
                "cedula": "",
                "centro": "Hospital de Prueba",
                "edad": 999,
                "unidad_edad": "años",
                "sexo": "F",
                "procedencia": "Petare",
                "especialidad": "Pediatría",
                "area": "",
                "estado_revision": "No requerido",
                "observaciones": "",
            }
        ]
    )

    try:
        apply_patient_edits([patient], edited)
    except ValueError as error:
        assert "Edad fuera del rango" in str(error)
    else:
        raise AssertionError("La edad inválida debía rechazarse")


def test_age_unit_is_cleared_when_age_is_removed() -> None:
    patient = PatientRecord(
        full_name="María Pérez",
        first_name="María",
        last_name="Pérez",
        name_split_confidence=1.0,
        detected_name_order="Nombre-Apellido",
        center="Hospital de Prueba",
        age=8,
        age_unit="años",
        sex="F",
        origin="Petare",
        specialty="Pediatría",
        area="",
        source_image="lista.jpg",
        confidence=0.9,
        needs_review=False,
        patient_id="PAC-0001",
    )
    edited = pd.DataFrame(
        [
            {
                "id_paciente": "PAC-0001",
                "nombre_completo": "María Pérez",
                "nombre": "María",
                "apellido": "Pérez",
                "cedula": "",
                "centro": "Hospital de Prueba",
                "edad": None,
                "unidad_edad": "años",
                "sexo": "F",
                "procedencia": "Petare",
                "especialidad": "Pediatría",
                "area": "",
                "estado_revision": "No requerido",
                "observaciones": "",
            }
        ]
    )

    apply_patient_edits([patient], edited)

    assert patient.age is None
    assert patient.age_unit == ""
