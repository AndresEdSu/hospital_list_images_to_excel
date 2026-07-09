from __future__ import annotations

from statistics import median

from hospital_ocr.matching import detect_specialty, match_places
from hospital_ocr.models import (
    OcrLine,
    PatientRecord,
    Place,
    Specialty,
    TableGrid,
)
from hospital_ocr.name_splitter import (
    NameLexicons,
    normalize_identity_text,
    split_full_name,
)
from hospital_ocr.table_extraction.common import name_from_text
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
    SectionHeading,
    find_section_headings as _find_section_headings,
    section_for_line as _section_for_line,
)
from hospital_ocr.table_extraction.schema import infer_schema as _infer_schema
from hospital_ocr.table_extraction.types import TableSchema
from hospital_ocr.text import clean_display_text, normalize_text


def _schema_name_fields(schema: TableSchema) -> tuple[str, ...]:
    if "name" in schema.columns:
        return ("name",)
    return tuple(
        field
        for field in ("given_names", "surnames")
        if field in schema.columns
    )


def _schema_name_lines(
    lines: list[OcrLine],
    schema: TableSchema,
    grid: TableGrid | None,
) -> list[OcrLine]:
    selected: list[OcrLine] = []
    for field in _schema_name_fields(schema):
        selected.extend(
            line
            for line in _schema_lines(lines, schema, field, grid)
            if name_from_text(
                line.text,
                allow_short_single=field in {"given_names", "surnames"},
            )
        )
    return selected


def _schema_name_text(
    lines: list[OcrLine],
    schema: TableSchema,
    field: str,
    grid: TableGrid | None,
) -> str:
    names = [
        name
        for line in _schema_lines(lines, schema, field, grid)
        if (
            name := name_from_text(
                line.text,
                allow_short_single=True,
            )
        )
    ]
    return clean_display_text(" ".join(names))


def _schema_name_confidence(schema: TableSchema) -> float:
    columns = [
        schema.columns[field]
        for field in _schema_name_fields(schema)
    ]
    if not columns:
        return 0.0
    return sum(column.confidence for column in columns) / len(columns)


def _headerless_left_name_lines(
    lines: list[OcrLine],
    anchor: OcrLine,
    places: list[Place],
    specialties: list[Specialty],
    grid: TableGrid | None,
) -> list[OcrLine]:
    if grid is None:
        return []
    anchor_column = grid.column_for_box(anchor.box)
    if anchor_column is None:
        return []

    candidates_by_column: dict[int, list[OcrLine]] = {}
    for line in lines:
        if line is anchor:
            continue
        column = grid.column_for_box(line.box)
        if column is None or column >= anchor_column:
            continue
        candidate_name = name_from_text(line.text, allow_short_single=True)
        if not candidate_name:
            continue
        if detect_specialty(line.text, specialties) is not None:
            continue
        if (
            len(candidate_name.split()) <= 2
            and match_places(line.text, places, contextual=True)
        ):
            continue
        candidates_by_column.setdefault(column, []).append(line)

    if not candidates_by_column:
        return []
    target_column = max(candidates_by_column)
    return sorted(
        candidates_by_column[target_column],
        key=lambda item: item.center_x,
    )


def _line_names(lines: list[OcrLine]) -> str:
    return clean_display_text(
        " ".join(
            name
            for line in sorted(lines, key=lambda item: item.center_x)
            if (name := name_from_text(line.text, allow_short_single=True))
        )
    )


def _has_surname_hint(text: str, lexicons: NameLexicons) -> bool:
    return any(
        normalize_text(token) in lexicons.surnames
        for token in text.split()
    )


def _headerless_section_groups(
    lines: list[OcrLine],
    section_headings: list[SectionHeading],
) -> list[list[OcrLine]]:
    if len(section_headings) < 2:
        return [lines]

    groups: list[list[OcrLine]] = []
    unsectioned = [
        line for line in lines if _section_for_line(line, section_headings) is None
    ]
    if unsectioned:
        groups.append(unsectioned)

    for section in sorted(
        section_headings,
        key=lambda item: item.line.center_y,
    ):
        section_lines = [
            line
            for line in lines
            if _section_for_line(line, section_headings) is section
        ]
        if section_lines:
            groups.append(section_lines)

    return groups or [lines]


def _headerless_section_anchors(
    lines: list[OcrLine],
    section_headings: list[SectionHeading],
    grid: TableGrid | None,
    places: list[Place],
) -> list[RowAnchor]:
    groups = _headerless_section_groups(lines, section_headings)
    if len(groups) <= 1:
        return _find_row_anchors(lines, None, grid, places)

    anchors: list[RowAnchor] = []
    for group in groups:
        anchors.extend(_find_row_anchors(group, None, grid, places))

    if len(anchors) < 2:
        return _find_row_anchors(lines, None, grid, places)
    return sorted(anchors, key=lambda item: item.line.center_y)


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
    anchors = (
        _headerless_section_anchors(
            record_lines,
            section_headings,
            grid,
            places,
        )
        if schema is None
        else _find_row_anchors(record_lines, schema, grid, places)
    )
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
            has_explicit_document = "document" in schema.columns
            name_lines = _schema_name_lines(row_lines, schema, grid)
            has_split_name_columns = (
                "given_names" in schema.columns
                and "surnames" in schema.columns
            )
            given_name_text = _schema_name_text(
                row_lines, schema, "given_names", grid
            )
            surname_text = _schema_name_text(
                row_lines, schema, "surnames", grid
            )
            headerless_given_name_text = ""
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
            if not document_id and not has_explicit_document:
                document_lines = [
                    line for line in row_lines if line not in name_lines
                ]
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
            headerless_given_name_lines = _headerless_left_name_lines(
                row_lines,
                anchor.line,
                places,
                specialties,
                grid,
            )
            name_lines = [*headerless_given_name_lines, anchor.line]
            has_split_name_columns = False
            given_name_text = ""
            surname_text = ""
            headerless_given_name_text = _line_names(
                headerless_given_name_lines
            )
            field_lines = [
                line
                for line in _headerless_field_lines(row_lines, anchor, grid)
                if line not in headerless_given_name_lines
            ]
            semantic_lines = [anchor.line, *field_lines]
            document_lines = semantic_lines
            document_id = _extract_document(semantic_lines)
            age, age_unit = _extract_semantic_age(
                semantic_lines,
                allow_trailing_name_age=len(section_headings) >= 2,
            )
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
            has_explicit_document = False

        has_explicit_origin = bool(schema and "origin" in schema.columns)
        place_matches = match_places(
            origin_text,
            places,
            contextual=has_explicit_origin,
        )
        origin = (
            " - ".join(match.name for match in place_matches)
            if place_matches
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
        if has_split_name_columns:
            given_name_text = normalize_identity_text(
                given_name_text,
                name_lexicons,
                role="given",
            )
            surname_text = normalize_identity_text(
                surname_text,
                name_lexicons,
                role="surname",
            )
            full_name = clean_display_text(
                " ".join(
                    part for part in (given_name_text, surname_text) if part
                )
            ) or anchor.name
            first_name = given_name_text
            last_name = surname_text
            name_split_confidence = min(
                1.0,
                _average_score(name_lines, anchor.line.score)
                * (1.0 if first_name and last_name else 0.75),
            )
            detected_name_order = "Nombre-Apellido"
            name_split_reliable = bool(first_name and last_name)
        elif headerless_given_name_text:
            headerless_given_name_text = normalize_identity_text(
                headerless_given_name_text,
                name_lexicons,
                role="given",
            )
            headerless_surname_text = normalize_identity_text(
                anchor.name,
                name_lexicons,
                role="surname",
            )
            full_name = clean_display_text(
                f"{headerless_given_name_text} {headerless_surname_text}"
            )
            first_name = headerless_given_name_text
            last_name = headerless_surname_text
            name_split_confidence = min(
                1.0,
                _average_score(name_lines, anchor.line.score),
            )
            detected_name_order = "Nombre-Apellido"
            name_split_reliable = True
        else:
            source_name = (
                normalize_identity_text(
                    anchor.name,
                    name_lexicons,
                    role="mixed",
                )
                if schema and "name" in schema.columns
                else anchor.name
            )
            name_split = split_full_name(source_name, name_lexicons)
            if schema is None and not name_split.reliable:
                surname_only = normalize_identity_text(
                    anchor.name,
                    name_lexicons,
                    role="surname",
                )
                if (
                    surname_only != anchor.name
                    or _has_surname_hint(surname_only, name_lexicons)
                ):
                    full_name = surname_only
                    first_name = ""
                    last_name = surname_only
                    name_split_confidence = min(
                        1.0,
                        _average_score(name_lines, anchor.line.score) * 0.75,
                    )
                    detected_name_order = "Indeterminado"
                    name_split_reliable = False
                else:
                    full_name = source_name
                    first_name = name_split.first_name
                    last_name = name_split.last_name
                    name_split_confidence = name_split.confidence
                    detected_name_order = name_split.detected_order
                    name_split_reliable = name_split.reliable
            else:
                full_name = source_name
                first_name = name_split.first_name
                last_name = name_split.last_name
                name_split_confidence = name_split.confidence
                detected_name_order = name_split.detected_order
                name_split_reliable = name_split.reliable
        notes: list[str] = []
        if not name_split_reliable:
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
        elif has_explicit_origin and places and not place_matches:
            notes.append("Procedencia no validada en catálogo")
        elif any(match.contextual for match in place_matches):
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
                _average_score(name_lines, anchor.line.score)
                * (0.80 + 0.20 * _schema_name_confidence(schema)),
            )
            document_confidence = (
                min(
                    1.0,
                    _average_score(document_lines)
                    * (0.80 + 0.20 * schema.columns["document"].confidence),
                )
                if document_id and "document" in schema.columns
                else min(1.0, _average_score(document_lines) * 0.75)
                if document_id
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
                    * (
                        min(match.score for match in place_matches)
                        if place_matches
                        else 0.65
                    ),
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
                    if document_id and has_explicit_document
                    else "formato de documento en la fila sin encabezado"
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
                    if grid and any(match.contextual for match in place_matches)
                    else "encabezado y catálogo geográfico contextual"
                    if any(match.contextual for match in place_matches)
                    else "cuadrícula, encabezado y catálogo geográfico"
                    if grid and place_matches
                    else "encabezado y catálogo geográfico"
                    if place_matches
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
                confidence * min(match.score for match in place_matches)
                if origin and place_matches
                else 0.0
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
                    if grid and place_matches
                    else "catálogo geográfico en celda sin encabezado"
                    if place_matches
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
                full_name=full_name,
                first_name=first_name,
                last_name=last_name,
                name_split_confidence=name_split_confidence,
                detected_name_order=detected_name_order,
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
