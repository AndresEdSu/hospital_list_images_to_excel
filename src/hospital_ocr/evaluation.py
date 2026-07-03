from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from hospital_ocr.catalogs import load_places, load_specialties
from hospital_ocr.grid_detector import detect_table_grid
from hospital_ocr.models import PatientRecord
from hospital_ocr.name_splitter import load_name_lexicons
from hospital_ocr.ocr_engine import PaddleOcrEngine, save_raw_ocr
from hospital_ocr.parsing import parse_ocr_lines
from hospital_ocr.pipeline import OCR_MODES, OcrMode, _recognize_image
from hospital_ocr.preprocessing import preprocess_image
from hospital_ocr.text import normalize_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
EVALUATION_FIELDS = (
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
)
ALLOWED_AGE_UNITS = {"", "años", "meses", "días"}
ALLOWED_SEX_VALUES = {"", "M", "F"}
MOJIBAKE_MARKERS = ("�", "ï¿½", "Ã")


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class EvaluationCase:
    stem: str
    image_path: Path
    expected_path: Path
    center: str
    expected_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class Alignment:
    expected_index: int | None
    predicted_index: int | None
    similarity: float = 0.0


@dataclass(frozen=True)
class CaseEvaluation:
    stem: str
    expected_count: int
    predicted_count: int
    matched_count: int
    field_correct: dict[str, int]
    field_total: dict[str, int]
    populated_correct: dict[str, int]
    populated_total: dict[str, int]
    false_positives: dict[str, int]
    mismatches: tuple[dict[str, object], ...]

    @property
    def record_recall(self) -> float:
        return self.matched_count / self.expected_count if self.expected_count else 1.0

    @property
    def record_precision(self) -> float:
        return self.matched_count / self.predicted_count if self.predicted_count else (
            1.0 if not self.expected_count else 0.0
        )


def _read_expected_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            headers = list(reader.fieldnames or [])
            rows = [
                {
                    key: (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            ]
    except UnicodeDecodeError as error:
        raise DatasetValidationError(
            [f"{path.name}: no es UTF-8 válido ({error})"]
        ) from error
    return headers, rows


def _validate_expected_rows(
    path: Path,
    headers: list[str],
    rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    missing = [field for field in EVALUATION_FIELDS if field not in headers]
    if missing:
        errors.append(
            f"{path.name}: faltan columnas requeridas: {', '.join(missing)}"
        )
        return errors
    if not rows:
        errors.append(f"{path.name}: no contiene registros esperados")
        return errors

    centers = {row["centro"] for row in rows if row["centro"]}
    if len(centers) != 1:
        errors.append(
            f"{path.name}: debe contener exactamente un centro no vacío"
        )

    for row_number, row in enumerate(rows, start=2):
        for field in EVALUATION_FIELDS:
            value = row[field]
            if any(marker in value for marker in MOJIBAKE_MARKERS):
                errors.append(
                    f"{path.name}:{row_number}: texto mal codificado en "
                    f"{field}: {value!r}"
                )
        age = row["edad"]
        if age:
            try:
                parsed_age = int(age)
            except ValueError:
                errors.append(
                    f"{path.name}:{row_number}: edad no entera: {age!r}"
                )
            else:
                if not 0 <= parsed_age <= 115:
                    errors.append(
                        f"{path.name}:{row_number}: edad fuera de rango: {age}"
                    )
            if not row["unidad_edad"]:
                errors.append(
                    f"{path.name}:{row_number}: edad sin unidad de edad"
                )
        elif row["unidad_edad"]:
            errors.append(
                f"{path.name}:{row_number}: unidad de edad sin edad"
            )
        if row["unidad_edad"] not in ALLOWED_AGE_UNITS:
            errors.append(
                f"{path.name}:{row_number}: unidad de edad no válida: "
                f"{row['unidad_edad']!r}"
            )
        if row["sexo"].upper() not in ALLOWED_SEX_VALUES:
            errors.append(
                f"{path.name}:{row_number}: sexo no válido: {row['sexo']!r}"
            )
    return errors


def load_evaluation_cases(
    dataset_dir: Path,
    only: Iterable[str] | None = None,
) -> list[EvaluationCase]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise DatasetValidationError(
            [f"No existe el directorio de evaluación: {dataset_dir}"]
        )
    selected = {value.casefold() for value in (only or [])}
    image_paths = [
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not selected or path.stem.casefold() in selected)
    ]
    csv_paths = [
        path
        for path in dataset_dir.glob("*.csv")
        if not selected or path.stem.casefold() in selected
    ]
    images_by_stem: dict[str, list[Path]] = {}
    for path in image_paths:
        images_by_stem.setdefault(path.stem, []).append(path)
    csv_by_stem = {path.stem: path for path in csv_paths}

    errors: list[str] = []
    for stem, paths in images_by_stem.items():
        if len(paths) > 1:
            errors.append(
                f"{stem}: hay varias imágenes con el mismo nombre base"
            )
    image_stems = set(images_by_stem)
    csv_stems = set(csv_by_stem)
    for stem in sorted(image_stems - csv_stems):
        errors.append(f"{stem}: falta el CSV esperado")
    for stem in sorted(csv_stems - image_stems):
        errors.append(f"{stem}: falta la imagen")
    if selected:
        found = {stem.casefold() for stem in image_stems | csv_stems}
        for stem in sorted(selected - found):
            errors.append(f"{stem}: caso solicitado no encontrado")
    if not image_stems and not csv_stems:
        errors.append("No se encontraron parejas de imagen y CSV")

    cases: list[EvaluationCase] = []
    for stem in sorted(image_stems & csv_stems):
        expected_path = csv_by_stem[stem]
        headers, rows = _read_expected_csv(expected_path)
        errors.extend(_validate_expected_rows(expected_path, headers, rows))
        centers = {row.get("centro", "") for row in rows if row.get("centro", "")}
        center = next(iter(centers)) if len(centers) == 1 else ""
        cases.append(
            EvaluationCase(
                stem=stem,
                image_path=images_by_stem[stem][0],
                expected_path=expected_path,
                center=center,
                expected_rows=tuple(rows),
            )
        )
    if errors:
        raise DatasetValidationError(errors)
    return cases


def _normalized_field(field: str, value: object) -> str:
    text = "" if value is None else str(value).strip()
    if field == "cedula":
        return "".join(character for character in text if character.isdigit())
    if field == "edad":
        if not text:
            return ""
        try:
            return str(int(float(text)))
        except ValueError:
            return normalize_text(text)
    if field == "sexo":
        return text.upper()
    return normalize_text(text)


def patient_to_expected_row(record: PatientRecord) -> dict[str, str]:
    return {
        "nombre_completo": record.full_name,
        "nombre": record.first_name,
        "apellido": record.last_name,
        "cedula": record.document_id,
        "centro": record.center,
        "edad": "" if record.age is None else str(record.age),
        "unidad_edad": record.age_unit if record.age is not None else "",
        "sexo": record.sex,
        "procedencia": record.origin,
        "especialidad": record.specialty,
        "area": record.area,
    }


def _identity_similarity(
    expected: dict[str, str],
    predicted: dict[str, str],
) -> float:
    expected_name = _normalized_field(
        "nombre_completo", expected["nombre_completo"]
    )
    predicted_name = _normalized_field(
        "nombre_completo", predicted["nombre_completo"]
    )
    name_score = (
        SequenceMatcher(None, expected_name, predicted_name).ratio()
        if expected_name and predicted_name
        else 0.0
    )
    expected_document = _normalized_field("cedula", expected["cedula"])
    predicted_document = _normalized_field("cedula", predicted["cedula"])
    documents_present = bool(expected_document and predicted_document)
    document_match = (
        documents_present and expected_document == predicted_document
    )
    if documents_present and not document_match:
        name_score *= 0.35

    age_match = (
        bool(expected["edad"] and predicted["edad"])
        and _normalized_field("edad", expected["edad"])
        == _normalized_field("edad", predicted["edad"])
    )
    sex_match = (
        bool(expected["sexo"] and predicted["sexo"])
        and _normalized_field("sexo", expected["sexo"])
        == _normalized_field("sexo", predicted["sexo"])
    )
    score = (
        name_score * 0.78
        + float(document_match) * 0.14
        + float(age_match) * 0.05
        + float(sex_match) * 0.03
    )
    return max(score, 0.98 if document_match else 0.0)


def align_rows(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    minimum_similarity: float = 0.38,
) -> list[Alignment]:
    expected_count = len(expected_rows)
    predicted_count = len(predicted_rows)
    gap_penalty = -0.42
    scores = [
        [
            _identity_similarity(expected, predicted)
            for predicted in predicted_rows
        ]
        for expected in expected_rows
    ]
    matrix = [
        [0.0] * (predicted_count + 1)
        for _ in range(expected_count + 1)
    ]
    moves = [
        [""] * (predicted_count + 1)
        for _ in range(expected_count + 1)
    ]
    for index in range(1, expected_count + 1):
        matrix[index][0] = index * gap_penalty
        moves[index][0] = "missing"
    for index in range(1, predicted_count + 1):
        matrix[0][index] = index * gap_penalty
        moves[0][index] = "unexpected"

    for expected_index in range(1, expected_count + 1):
        for predicted_index in range(1, predicted_count + 1):
            similarity = scores[expected_index - 1][predicted_index - 1]
            diagonal = (
                matrix[expected_index - 1][predicted_index - 1] + similarity
                if similarity >= minimum_similarity
                else float("-inf")
            )
            missing = matrix[expected_index - 1][predicted_index] + gap_penalty
            unexpected = matrix[expected_index][predicted_index - 1] + gap_penalty
            best = max(diagonal, missing, unexpected)
            matrix[expected_index][predicted_index] = best
            moves[expected_index][predicted_index] = (
                "match"
                if best == diagonal
                else "missing"
                if best == missing
                else "unexpected"
            )

    alignment: list[Alignment] = []
    expected_index = expected_count
    predicted_index = predicted_count
    while expected_index or predicted_index:
        move = moves[expected_index][predicted_index]
        if move == "match":
            alignment.append(
                Alignment(
                    expected_index - 1,
                    predicted_index - 1,
                    scores[expected_index - 1][predicted_index - 1],
                )
            )
            expected_index -= 1
            predicted_index -= 1
        elif move == "missing":
            alignment.append(Alignment(expected_index - 1, None))
            expected_index -= 1
        else:
            alignment.append(Alignment(None, predicted_index - 1))
            predicted_index -= 1
    alignment.reverse()
    return alignment


def evaluate_rows(
    stem: str,
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
) -> CaseEvaluation:
    alignment = align_rows(expected_rows, predicted_rows)
    field_correct = {field: 0 for field in EVALUATION_FIELDS}
    field_total = {
        field: len(expected_rows) for field in EVALUATION_FIELDS
    }
    populated_correct = {field: 0 for field in EVALUATION_FIELDS}
    populated_total = {
        field: sum(
            bool(_normalized_field(field, row[field]))
            for row in expected_rows
        )
        for field in EVALUATION_FIELDS
    }
    false_positives = {field: 0 for field in EVALUATION_FIELDS}
    mismatches: list[dict[str, object]] = []
    matched_count = 0

    for pair in alignment:
        if pair.expected_index is None:
            predicted = predicted_rows[pair.predicted_index]  # type: ignore[index]
            mismatches.append(
                {
                    "imagen": stem,
                    "tipo": "registro_adicional",
                    "fila_esperada": "",
                    "fila_obtenida": pair.predicted_index + 1,  # type: ignore[operator]
                    "campo": "",
                    "esperado": "",
                    "obtenido": predicted["nombre_completo"],
                    "similitud": "",
                }
            )
            continue
        expected = expected_rows[pair.expected_index]
        if pair.predicted_index is None:
            mismatches.append(
                {
                    "imagen": stem,
                    "tipo": "registro_faltante",
                    "fila_esperada": pair.expected_index + 1,
                    "fila_obtenida": "",
                    "campo": "",
                    "esperado": expected["nombre_completo"],
                    "obtenido": "",
                    "similitud": "",
                }
            )
            continue

        matched_count += 1
        predicted = predicted_rows[pair.predicted_index]
        for field in EVALUATION_FIELDS:
            expected_value = _normalized_field(field, expected[field])
            predicted_value = _normalized_field(field, predicted[field])
            if expected_value == predicted_value:
                field_correct[field] += 1
                if expected_value:
                    populated_correct[field] += 1
            else:
                if not expected_value and predicted_value:
                    false_positives[field] += 1
                mismatches.append(
                    {
                        "imagen": stem,
                        "tipo": "campo_incorrecto",
                        "fila_esperada": pair.expected_index + 1,
                        "fila_obtenida": pair.predicted_index + 1,
                        "campo": field,
                        "esperado": expected[field],
                        "obtenido": predicted[field],
                        "similitud": round(pair.similarity, 4),
                    }
                )

    return CaseEvaluation(
        stem=stem,
        expected_count=len(expected_rows),
        predicted_count=len(predicted_rows),
        matched_count=matched_count,
        field_correct=field_correct,
        field_total=field_total,
        populated_correct=populated_correct,
        populated_total=populated_total,
        false_positives=false_positives,
        mismatches=tuple(mismatches),
    )


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _write_evaluation_report(
    output_dir: Path,
    evaluations: list[CaseEvaluation],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_expected = sum(item.expected_count for item in evaluations)
    total_predicted = sum(item.predicted_count for item in evaluations)
    total_matched = sum(item.matched_count for item in evaluations)
    field_rows: list[dict[str, object]] = []
    for field in EVALUATION_FIELDS:
        correct = sum(item.field_correct[field] for item in evaluations)
        total = sum(item.field_total[field] for item in evaluations)
        populated_correct = sum(
            item.populated_correct[field] for item in evaluations
        )
        populated_total = sum(
            item.populated_total[field] for item in evaluations
        )
        field_rows.append(
            {
                "campo": field,
                "correctos": correct,
                "total": total,
                "exactitud": round(correct / total, 4) if total else 1.0,
                "correctos_con_dato": populated_correct,
                "total_con_dato": populated_total,
                "exactitud_con_dato": (
                    round(populated_correct / populated_total, 4)
                    if populated_total
                    else 1.0
                ),
                "falsos_positivos": sum(
                    item.false_positives[field] for item in evaluations
                ),
            }
        )
    case_rows = [
        {
            "imagen": item.stem,
            "esperados": item.expected_count,
            "obtenidos": item.predicted_count,
            "alineados": item.matched_count,
            "precision_registros": round(item.record_precision, 4),
            "cobertura_registros": round(item.record_recall, 4),
            "diferencias": len(item.mismatches),
        }
        for item in evaluations
    ]
    mismatch_rows = [
        mismatch
        for item in evaluations
        for mismatch in item.mismatches
    ]
    _write_csv(
        output_dir / "casos.csv",
        case_rows,
        (
            "imagen",
            "esperados",
            "obtenidos",
            "alineados",
            "precision_registros",
            "cobertura_registros",
            "diferencias",
        ),
    )
    _write_csv(
        output_dir / "campos.csv",
        field_rows,
        (
            "campo",
            "correctos",
            "total",
            "exactitud",
            "correctos_con_dato",
            "total_con_dato",
            "exactitud_con_dato",
            "falsos_positivos",
        ),
    )
    _write_csv(
        output_dir / "diferencias.csv",
        mismatch_rows,
        (
            "imagen",
            "tipo",
            "fila_esperada",
            "fila_obtenida",
            "campo",
            "esperado",
            "obtenido",
            "similitud",
        ),
    )
    summary = {
        "imagenes": len(evaluations),
        "registros_esperados": total_expected,
        "registros_obtenidos": total_predicted,
        "registros_alineados": total_matched,
        "precision_registros": (
            round(total_matched / total_predicted, 4)
            if total_predicted
            else (1.0 if not total_expected else 0.0)
        ),
        "cobertura_registros": (
            round(total_matched / total_expected, 4)
            if total_expected
            else 1.0
        ),
        "casos": case_rows,
        "campos": field_rows,
    }
    (output_dir / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate_saved_predictions(
    dataset_dir: Path,
    predictions_root: Path,
    output_dir: Path,
    *,
    only: Iterable[str] | None = None,
) -> list[CaseEvaluation]:
    cases = load_evaluation_cases(dataset_dir, only)
    evaluations: list[CaseEvaluation] = []
    errors: list[str] = []
    for case in cases:
        candidates = (
            predictions_root / "predicciones" / f"{case.stem}.csv",
            predictions_root
            / case.stem
            / "predicciones"
            / f"{case.stem}.csv",
        )
        prediction_path = next(
            (path for path in candidates if path.is_file()),
            None,
        )
        if prediction_path is None:
            errors.append(f"{case.stem}: no se encontró el CSV de predicción")
            continue
        headers, predicted_rows = _read_expected_csv(prediction_path)
        missing = [
            field for field in EVALUATION_FIELDS if field not in headers
        ]
        if missing:
            errors.append(
                f"{prediction_path}: faltan columnas: {', '.join(missing)}"
            )
            continue
        evaluations.append(
            evaluate_rows(
                case.stem,
                list(case.expected_rows),
                predicted_rows,
            )
        )
    if errors:
        raise DatasetValidationError(errors)
    _write_evaluation_report(output_dir, evaluations)
    return evaluations


def evaluate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    project_root: Path,
    *,
    ocr_mode: OcrMode = "auto",
    only: Iterable[str] | None = None,
) -> list[CaseEvaluation]:
    cases = load_evaluation_cases(dataset_dir, only)
    specialties = load_specialties(project_root / "config" / "especialidades.csv")
    places = load_places(project_root / "config" / "lugares.csv")
    name_lexicons = load_name_lexicons(
        project_root / "config" / "nombres_comunes.csv",
        project_root / "config" / "apellidos_comunes.csv",
    )
    engine = PaddleOcrEngine(project_root / ".cache" / "paddlex")
    evaluations: list[CaseEvaluation] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.stem}: preprocesamiento", flush=True)
        interim = output_dir / "interim"
        processed_path = interim / "preprocessed" / f"{case.stem}.jpg"
        preprocess_image(case.image_path, processed_path)
        grid_path = interim / "grids" / f"{case.stem}.jpg"
        grid = detect_table_grid(processed_path, grid_path)
        print(f"[{index}/{len(cases)}] {case.stem}: OCR {ocr_mode}", flush=True)
        lines, audit = _recognize_image(
            engine,
            processed_path,
            grid,
            ocr_mode,
            interim / "handwriting_rows" / case.stem,
        )
        save_raw_ocr(
            interim / "ocr" / f"{case.stem}.json",
            case.image_path,
            lines,
        )
        if audit is not None:
            audit_path = interim / "handwriting_rows" / case.stem / "audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        records = parse_ocr_lines(
            lines,
            specialties,
            name_lexicons,
            case.center,
            case.image_path.name,
            places,
            grid,
        )
        predicted_rows = [patient_to_expected_row(record) for record in records]
        _write_csv(
            output_dir / "predicciones" / f"{case.stem}.csv",
            predicted_rows,
            EVALUATION_FIELDS,
        )
        evaluation = evaluate_rows(
            case.stem,
            list(case.expected_rows),
            predicted_rows,
        )
        evaluations.append(evaluation)
        print(
            f"[{index}/{len(cases)}] {case.stem}: "
            f"{evaluation.matched_count}/{evaluation.expected_count} "
            "registros alineados",
            flush=True,
        )

    _write_evaluation_report(output_dir, evaluations)
    return evaluations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa el OCR contra imágenes reales y CSV revisados."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/evaluation/test_images"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--ocr-mode",
        choices=OCR_MODES,
        default="auto",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Nombre base de una imagen; puede repetirse.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--from-predictions",
        type=Path,
        help="Recalcula métricas desde predicciones existentes sin ejecutar OCR.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        cases = load_evaluation_cases(args.dataset, args.only)
    except DatasetValidationError as error:
        raise SystemExit(
            "Corpus de evaluación inválido:\n- " + "\n- ".join(error.errors)
        ) from error
    print(
        f"Corpus válido: {len(cases)} imágenes, "
        f"{sum(len(case.expected_rows) for case in cases)} registros esperados."
    )
    if args.validate_only:
        return
    output_dir = args.output or args.dataset / "results"
    if args.from_predictions:
        try:
            evaluations = evaluate_saved_predictions(
                args.dataset,
                args.from_predictions,
                output_dir,
                only=args.only,
            )
        except DatasetValidationError as error:
            raise SystemExit(
                "Predicciones inválidas:\n- " + "\n- ".join(error.errors)
            ) from error
    else:
        evaluations = evaluate_dataset(
            args.dataset,
            output_dir,
            args.project_root.resolve(),
            ocr_mode=args.ocr_mode,
            only=args.only,
        )
    total_expected = sum(item.expected_count for item in evaluations)
    total_matched = sum(item.matched_count for item in evaluations)
    print(
        f"Línea base completada: {total_matched}/{total_expected} "
        f"registros alineados. Reporte: {output_dir}"
    )


if __name__ == "__main__":
    main()
