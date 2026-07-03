from __future__ import annotations

from statistics import median

from hospital_ocr.matching import detect_specialty, match_place
from hospital_ocr.models import (
    OcrLine,
    PatientRecord,
    Place,
    Specialty,
    TableGrid,
)
from hospital_ocr.name_splitter import NameLexicons, split_full_name
from hospital_ocr.table_extraction.detection import (
    infer_headerless_index_ids as _infer_headerless_index_ids,
    looks_like_table,
)
from hospital_ocr.table_extraction.fields import (
    average_score as _average_score,
    extract_document as _extract_document,
    extract_schema_age as _extract_schema_age,
    extract_schema_sex as _extract_schema_sex,
    extract_semantic_age as _extract_semantic_age,
    extract_semantic_sex as _extract_semantic_sex,
    headerless_field_lines as _headerless_field_lines,
    joined_cell_text as _joined_cell_text,
    schema_lines as _schema_lines,
    schema_text as _schema_text,
)
from hospital_ocr.table_extraction.rows import (
    complete_cropped_top_row as _complete_cropped_top_row,
    find_row_anchors as _find_row_anchors,
    grid_header_row as _grid_header_row,
    has_leading_index as _has_leading_index,
    header_cutoff as _header_cutoff,
    row_groups as _row_groups,
    row_index_lines as _row_index_lines,
)
from hospital_ocr.table_extraction.sections import (
    find_section_headings as _find_section_headings,
    section_for_line as _section_for_line,
)
from hospital_ocr.table_extraction.schema import infer_schema as _infer_schema


def parse_table_lines(
    lines: list[OcrLine],
    name_lexicons: NameLexicons,
    center: str,
    source_image: str,
    specialties: list[Specialty] | None = None,
    places: list[Place] | None = None,
    grid: TableGrid | None = None,
) -> list[PatientRecord] | None:
    specialties = specialties or []
    places = places or []
    grid = _complete_cropped_top_row(lines, grid, places)
    if not looks_like_table(lines, grid, places):
        return None
    schema = _infer_schema(lines, grid)
    grid_header_row = _grid_header_row(lines, grid)
    if grid_header_row is not None and grid is not None:
        data_lines = [
            line
            for line in lines
            if (
                (row := grid.row_for_box(line.box)) is not None
                and row > grid_header_row
            )
        ]
    else:
        data_lines = (
            [line for line in lines if line.center_y > schema.header_bottom]
            if schema
            else lines
        )
    section_headings = (
        _find_section_headings(data_lines, specialties, grid)
        if schema is None
        else []
    )
    section_line_ids = {
        line_id
        for heading in section_headings
        for line_id in heading.line_ids
    }
    record_lines = [
        line for line in data_lines if id(line) not in section_line_ids
    ]
    anchors = _find_row_anchors(record_lines, schema, grid, places)
    if len(anchors) < 2:
        return []

    headerless_index_ids = (
        _infer_headerless_index_ids(record_lines) if schema is None else set()
    )
    header_cutoff = (
        None
        if grid_header_row is not None
        else schema.header_bottom
        if schema
        else _header_cutoff(lines)
    )
    page_specialty: tuple[str, str] | None = None
    if schema:
        for line in sorted(lines, key=lambda item: item.center_y):
            if line.box[3] >= schema.header_bottom:
                continue
            if detected := detect_specialty(line.text, specialties):
                page_specialty = detected
                break
    groups = _row_groups(record_lines, anchors, grid)
    anchor_gaps = [
        current.line.center_y - previous.line.center_y
        for previous, current in zip(anchors, anchors[1:], strict=False)
        if current.line.center_y > previous.line.center_y
    ]
    section_gap_limit = (
        max(
            median(anchor_gaps) * 2.2,
            anchors[0].line.image_height * 0.035,
        )
        if anchor_gaps
        else float("inf")
    )
    records: list[PatientRecord] = []
    previous_anchor = None
    active_section = None
    last_section_candidate = None
    for anchor, row_lines in groups:
        section_candidate = _section_for_line(anchor.line, section_headings)
        if section_candidate is not last_section_candidate:
            active_section = section_candidate
            last_section_candidate = section_candidate
        if (
            active_section is not None
            and previous_anchor is not None
            and anchor.line.center_y - previous_anchor.line.center_y
            > section_gap_limit
            and active_section.line.center_y <= previous_anchor.line.center_y
        ):
            active_section = None
        section = active_section
        previous_anchor = anchor
        if header_cutoff is not None and anchor.line.center_y > header_cutoff:
            row_lines = [
                line for line in row_lines if line.center_y > header_cutoff
            ]
        raw_row_lines = list(row_lines)
        index_lines = _row_index_lines(
            row_lines,
            anchor,
            schema,
            headerless_index_ids,
        )
        row_lines = [line for line in row_lines if line not in index_lines]
        index_detected = bool(index_lines or _has_leading_index(anchor.line.text))
        if schema:
            document_lines = _schema_lines(
                row_lines, schema, "document", grid
            )
            age_lines = _schema_lines(row_lines, schema, "age", grid)
            sex_lines = _schema_lines(row_lines, schema, "sex", grid)
            origin_lines = _schema_lines(row_lines, schema, "origin", grid)
            specialty_lines = _schema_lines(
                row_lines, schema, "specialty", grid
            )
            document_id = _extract_document(document_lines)
            age = _extract_schema_age(age_lines)
            age_unit = "años" if age is not None else ""
            sex_result = _extract_schema_sex(
                sex_lines,
                allow_ocr_confusions=True,
            )
            sex = sex_result.value
            origin_text = _schema_text(
                row_lines, schema, "origin", grid
            )
            plan = _schema_text(row_lines, schema, "plan", grid)
            specialty_text = _schema_text(
                row_lines, schema, "specialty", grid
            )
        else:
            field_lines = _headerless_field_lines(row_lines, anchor, grid)
            semantic_lines = [anchor.line, *field_lines]
            document_lines = semantic_lines
            document_id = _extract_document(semantic_lines)
            age, age_unit = _extract_semantic_age(semantic_lines)
            sex_result = _extract_semantic_sex(semantic_lines)
            sex = sex_result.value
            semantic_text = _joined_cell_text(field_lines)
            origin_text = semantic_text
            specialty_text = semantic_text
            plan = ""
            age_lines = field_lines
            sex_lines = field_lines
            origin_lines = field_lines
            specialty_lines = field_lines

        has_explicit_origin = bool(schema and "origin" in schema.columns)
        place = match_place(
            origin_text,
            places,
            contextual=has_explicit_origin,
        )
        origin = (
            place.name
            if place
            else origin_text
            if has_explicit_origin
            else ""
        )
        row_specialty = next(
            (
                detected
                for line in specialty_lines
                if (detected := detect_specialty(line.text, specialties))
                is not None
            ),
            None,
        ) or (
            detect_specialty(specialty_text, specialties)
            if specialty_text
            else None
        )
        contextual_specialty = (
            (section.specialty, section.area)
            if section is not None
            else page_specialty
        )
        detected_specialty = row_specialty or contextual_specialty
        specialty = detected_specialty[0] if detected_specialty else ""
        area = detected_specialty[1] if detected_specialty else ""
        name_split = split_full_name(anchor.name, name_lexicons)
        notes: list[str] = []
        if not name_split.reliable:
            notes.append("Separación de nombre no confiable")
        if age is None:
            notes.append("Edad no reconocida")
        if sex_result.conflict:
            notes.append(
                "Sexo ambiguo entre valores incompatibles: "
                + "/".join(sex_result.normalized_from)
            )
        elif not sex:
            notes.append("Sexo no reconocido")
        elif sex_result.normalized_from:
            notes.append(
                "Sexo normalizado desde OCR: "
                + "/".join(sex_result.normalized_from)
            )
        if not origin:
            notes.append("Procedencia no reconocida")
        elif has_explicit_origin and places and place is None:
            notes.append("Procedencia no validada en catálogo")
        elif place and place.contextual:
            notes.append(
                "Procedencia normalizada por coincidencia contextual"
            )
        if specialty_text and detected_specialty is None:
            notes.append("Especialidad o área no reconocida")

        raw_line = " ".join(
            line.text
            for line in sorted(raw_row_lines, key=lambda item: item.center_x)
        )
        scores = [line.score for line in row_lines if line.text.strip()]
        confidence = sum(scores) / len(scores) if scores else anchor.line.score
        if schema:
            name_confidence = min(
                1.0,
                anchor.line.score
                * (0.80 + 0.20 * schema.columns["name"].confidence),
            )
            document_confidence = (
                min(
                    1.0,
                    _average_score(document_lines)
                    * (0.80 + 0.20 * schema.columns["document"].confidence),
                )
                if document_id and "document" in schema.columns
                else 0.0
            )
            age_confidence = (
                min(
                    1.0,
                    _average_score(age_lines)
                    * (0.80 + 0.20 * schema.columns["age"].confidence),
                )
                if age is not None and "age" in schema.columns
                else 0.0
            )
            origin_confidence = (
                min(
                    1.0,
                    _average_score(origin_lines)
                    * (place.score if place else 0.65),
                )
                if origin
                else 0.0
            )
            specialty_confidence = (
                min(
                    1.0,
                    _average_score(specialty_lines, anchor.line.score)
                    * 0.90,
                )
                if specialty
                else 0.0
            )
            evidence = {
                "nombre": (
                    "cuadrícula física y encabezado; índice descartado"
                    if grid and index_detected
                    else "cuadrícula física y encabezado"
                    if grid
                    else "columna inferida por encabezado; índice descartado"
                    if index_detected
                    else "columna inferida por encabezado"
                ),
                "cédula": (
                    "formato y columna inferida por encabezado"
                    if document_id
                    else ""
                ),
                "edad": (
                    "formato y columna inferida por encabezado"
                    if age is not None
                    else ""
                ),
                "sexo": (
                    "conjunto cerrado y columna inferida por encabezado"
                    + (
                        "; normalizado desde "
                        + "/".join(sex_result.normalized_from)
                        if sex_result.normalized_from
                        else ""
                    )
                    if sex
                    else ""
                ),
                "procedencia": (
                    "cuadrícula, encabezado y catálogo geográfico contextual"
                    if grid and place and place.contextual
                    else "encabezado y catálogo geográfico contextual"
                    if place and place.contextual
                    else "cuadrícula, encabezado y catálogo geográfico"
                    if grid and place
                    else "encabezado y catálogo geográfico"
                    if place
                    else "columna inferida por encabezado"
                    if origin
                    else ""
                ),
                "especialidad": (
                    "catálogo y columna inferida por encabezado"
                    if specialty_text and specialty
                    else "catálogo y membrete"
                    if specialty
                    else ""
                ),
            }
        else:
            name_confidence = anchor.line.score * 0.70
            document_confidence = confidence * 0.85 if document_id else 0.0
            age_confidence = confidence * 0.65 if age is not None else 0.0
            origin_confidence = (
                confidence * place.score if origin and place else 0.0
            )
            specialty_confidence = confidence * 0.75 if specialty else 0.0
            evidence = {
                "nombre": (
                    "cuadrícula física y columna repetida; índice descartado"
                    if grid and index_detected
                    else "cuadrícula física y columna repetida"
                    if grid
                    else "columna repetida y alineada; índice descartado"
                    if index_detected
                    else "columna repetida y alineada"
                ),
                "cédula": "formato de documento en la fila" if document_id else "",
                "edad": "formato de edad en la fila" if age is not None else "",
                "sexo": (
                    "valor cerrado en la fila"
                    + (
                        "; normalizado desde "
                        + "/".join(sex_result.normalized_from)
                        if sex_result.normalized_from
                        else ""
                    )
                    if sex
                    else ""
                ),
                "procedencia": (
                    "catálogo geográfico en celda física sin encabezado"
                    if grid and place
                    else "catálogo geográfico en celda sin encabezado"
                    if place
                    else ""
                ),
                "especialidad": (
                    "catálogo y encabezado de sección"
                    if specialty and section is not None
                    else "catálogo de especialidades en celda sin encabezado"
                    if specialty
                    else ""
                ),
            }
        records.append(
            PatientRecord(
                full_name=anchor.name,
                first_name=name_split.first_name,
                last_name=name_split.last_name,
                name_split_confidence=name_split.confidence,
                detected_name_order=name_split.detected_order,
                center=center,
                age=age,
                age_unit=age_unit,
                sex=sex,
                origin=origin,
                specialty=specialty,
                area=area,
                source_image=source_image,
                confidence=round(confidence, 4),
                needs_review=bool(notes),
                notes=notes,
                raw_line=raw_line,
                document_id=document_id,
                clinical_notes=f"Plan: {plan}" if plan else "",
                name_confidence=round(name_confidence, 4),
                document_confidence=round(document_confidence, 4),
                age_confidence=round(age_confidence, 4),
                origin_confidence=round(origin_confidence, 4),
                specialty_confidence=round(specialty_confidence, 4),
                field_evidence=evidence,
            )
        )
    return records
