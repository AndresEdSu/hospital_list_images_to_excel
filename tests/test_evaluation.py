import csv
from pathlib import Path

import pytest

from hospital_ocr.evaluation import (
    EVALUATION_FIELDS,
    DatasetValidationError,
    align_rows,
    evaluate_rows,
    evaluate_saved_predictions,
    load_evaluation_cases,
)


def expected_row(
    name: str,
    *,
    document: str = "",
    age: str = "",
    age_unit: str = "",
) -> dict[str, str]:
    values = {field: "" for field in EVALUATION_FIELDS}
    values.update(
        {
            "nombre_completo": name,
            "nombre": name.split()[0],
            "apellido": name.split()[-1],
            "cedula": document,
            "centro": "Hospital de Prueba",
            "edad": age,
            "unidad_edad": age_unit,
        }
    )
    return values


def write_case(
    root: Path,
    stem: str,
    rows: list[dict[str, str]],
) -> None:
    (root / f"{stem}.jpg").write_bytes(b"imagen")
    with (root / f"{stem}.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(EVALUATION_FIELDS),
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_dataset_loads_paired_utf8_cases(tmp_path: Path) -> None:
    write_case(
        tmp_path,
        "caso_1",
        [expected_row("María Pérez", age="8", age_unit="años")],
    )

    cases = load_evaluation_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].center == "Hospital de Prueba"
    assert cases[0].expected_rows[0]["nombre_completo"] == "María Pérez"


def test_evaluation_dataset_rejects_mojibake(tmp_path: Path) -> None:
    write_case(
        tmp_path,
        "caso_1",
        [expected_row("María Pérez", age="8", age_unit="aï¿½os")],
    )

    with pytest.raises(DatasetValidationError, match="mal codificado"):
        load_evaluation_cases(tmp_path)


def test_alignment_does_not_shift_after_a_missing_record() -> None:
    expected = [
        expected_row("Ana Pérez", document="11111111"),
        expected_row("Luis Gómez", document="22222222"),
        expected_row("María Torres", document="33333333"),
    ]
    predicted = [expected[0], expected[2]]

    alignment = align_rows(expected, predicted)

    assert alignment == [
        alignment[0].__class__(0, 0, 0.98),
        alignment[1].__class__(1, None, 0.0),
        alignment[2].__class__(2, 1, 0.98),
    ]


def test_field_metrics_count_missing_rows_as_incorrect() -> None:
    expected = [
        expected_row("Ana Pérez", document="11111111"),
        expected_row("Luis Gómez", document="22222222"),
    ]
    predicted = [expected[0]]

    result = evaluate_rows("caso", expected, predicted)

    assert result.expected_count == 2
    assert result.predicted_count == 1
    assert result.matched_count == 1
    assert result.field_total["cedula"] == 2
    assert result.field_correct["cedula"] == 1
    assert result.populated_correct["cedula"] == 1


def test_saved_predictions_can_be_recalculated_without_ocr(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    row = expected_row("María Pérez", age="8", age_unit="años")
    write_case(dataset, "caso_1", [row])
    prediction_dir = (
        tmp_path / "predictions" / "caso_1" / "predicciones"
    )
    prediction_dir.mkdir(parents=True)
    with (prediction_dir / "caso_1.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(EVALUATION_FIELDS),
            delimiter=";",
        )
        writer.writeheader()
        writer.writerow(row)

    output = tmp_path / "report"
    evaluations = evaluate_saved_predictions(
        dataset,
        tmp_path / "predictions",
        output,
    )

    assert evaluations[0].matched_count == 1
    assert (output / "resumen.json").is_file()
