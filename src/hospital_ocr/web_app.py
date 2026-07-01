from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hospital_ocr.catalogs import load_centers, write_center_catalog
from hospital_ocr.editing import apply_patient_edits
from hospital_ocr.exporting import export_results, patient_records_dataframe
from hospital_ocr.pipeline import OcrMode, PipelineConfig, process_images
from hospital_ocr.web_service import (
    cleanup_old_sessions,
    create_session,
    remove_session,
    save_uploaded_images,
)


def _project_root() -> Path:
    current = Path.cwd()
    if (current / "config" / "centros.csv").exists():
        return current
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
SESSIONS_ROOT = PROJECT_ROOT / "data" / "interim" / "web_sessions"
OTHER_CENTER_SLUG = "otro_centro"
EDITOR_COLUMNS = [
    "id_paciente",
    "nombre_completo",
    "nombre",
    "apellido",
    "cedula",
    "edad",
    "unidad_edad",
    "sexo",
    "procedencia",
    "especialidad",
    "area",
    "centro",
    "estado_revision",
    "observaciones",
    "estado_duplicado",
    "detalle_duplicado",
    "imagenes_origen",
    "confianza_ocr",
    "confianza_nombre",
    "confianza_cedula",
    "confianza_edad",
    "confianza_procedencia",
    "confianza_especialidad",
    "evidencia_extraccion",
    "linea_ocr_original",
]
READ_ONLY_COLUMNS = [
    "id_paciente",
    "estado_duplicado",
    "detalle_duplicado",
    "imagenes_origen",
    "confianza_ocr",
    "confianza_nombre",
    "confianza_cedula",
    "confianza_edad",
    "confianza_procedencia",
    "confianza_especialidad",
    "evidencia_extraccion",
    "linea_ocr_original",
]


def _clear_current_session() -> None:
    session_dir = st.session_state.get("session_dir")
    if session_dir:
        remove_session(Path(session_dir), SESSIONS_ROOT)
    for key in [
        "session_dir",
        "processing_result",
        "patients_df",
        "uploaded_paths",
        "excel_bytes",
    ]:
        st.session_state.pop(key, None)


def _merge_edited_rows(master: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    updated = master.copy()
    indexes = {
        str(patient_id): index
        for index, patient_id in updated["id_paciente"].items()
    }
    for _, row in edited.iterrows():
        index = indexes.get(str(row["id_paciente"]))
        if index is None:
            continue
        for column in edited.columns:
            updated.at[index, column] = row[column]
    return updated


def _process_uploads(
    uploaded_files: list[object],
    center_slug: str,
    custom_center_name: str = "",
    ocr_mode: OcrMode = "auto",
) -> None:
    if st.session_state.get("session_dir"):
        _clear_current_session()
    session_dir = create_session(SESSIONS_ROOT)
    st.session_state["session_dir"] = str(session_dir)
    images_root = session_dir / "images"
    try:
        uploaded_paths = save_uploaded_images(
            uploaded_files,
            images_root,
            center_slug,
        )
        centers_path = PROJECT_ROOT / "config" / "centros.csv"
        if center_slug == OTHER_CENTER_SLUG:
            centers_path = session_dir / "centros.csv"
            write_center_catalog(
                centers_path,
                OTHER_CENTER_SLUG,
                custom_center_name,
            )
        progress = st.progress(0, text="Preparando OCR…")

        def update_progress(
            completed: float,
            total: float,
            message: str,
        ) -> None:
            value = completed / total if total else 0
            progress.progress(value, text=f"{value:.0%} · {message}")

        config = PipelineConfig(
            images_dir=images_root,
            centers_path=centers_path,
            specialties_path=PROJECT_ROOT / "config" / "especialidades.csv",
            given_names_path=PROJECT_ROOT / "config" / "nombres_comunes.csv",
            surnames_path=PROJECT_ROOT / "config" / "apellidos_comunes.csv",
            places_path=PROJECT_ROOT / "config" / "lugares.csv",
            interim_dir=session_dir / "interim",
            output_path=session_dir / "pacientes.xlsx",
            cache_dir=PROJECT_ROOT / ".cache" / "paddlex",
            overwrite=True,
            ocr_mode=ocr_mode,
        )
        result = process_images(config, progress_callback=update_progress)
        progress.empty()
        st.session_state["processing_result"] = result
        st.session_state["patients_df"] = patient_records_dataframe(
            result.consolidation.patients
        )
        st.session_state["uploaded_paths"] = [str(path) for path in uploaded_paths]
        st.session_state.pop("excel_bytes", None)
    except Exception:
        remove_session(session_dir, SESSIONS_ROOT)
        st.session_state.pop("session_dir", None)
        raise


def _render_editor() -> None:
    result = st.session_state["processing_result"]
    master = st.session_state["patients_df"]
    st.subheader("Revisión de pacientes")
    metrics = st.columns(4)
    metrics[0].metric("Imágenes", result.processed_images)
    metrics[1].metric("Registros", len(master))
    metrics[2].metric(
        "Pendientes",
        int((master["estado_revision"] == "Pendiente").sum()),
    )
    metrics[3].metric(
        "Posibles duplicados",
        int((master["estado_duplicado"] == "Posible duplicado").sum()),
    )

    filter_name = st.selectbox(
        "Mostrar",
        ["Todos", "Pendientes", "Posibles duplicados"],
    )
    if filter_name == "Pendientes":
        visible = master[master["estado_revision"] == "Pendiente"]
    elif filter_name == "Posibles duplicados":
        visible = master[master["estado_duplicado"] == "Posible duplicado"]
    else:
        visible = master

    specialty_options = [""] + list(result.specialty_values)
    edited = st.data_editor(
        visible[EDITOR_COLUMNS],
        hide_index=True,
        num_rows="fixed",
        disabled=READ_ONLY_COLUMNS,
        width="stretch",
        height=520,
        key=f"patient_editor_{filter_name}",
        column_config={
            "sexo": st.column_config.SelectboxColumn(
                "Sexo", options=["", "M", "F"]
            ),
            "unidad_edad": st.column_config.SelectboxColumn(
                "Unidad de edad",
                options=["", "años", "meses", "días"],
            ),
            "especialidad": st.column_config.SelectboxColumn(
                "Especialidad", options=specialty_options
            ),
            "estado_revision": st.column_config.SelectboxColumn(
                "Estado",
                options=["Pendiente", "No requerido", "Revisado"],
            ),
            "confianza_ocr": st.column_config.ProgressColumn(
                "Confianza OCR",
                min_value=0.0,
                max_value=1.0,
                format="%.0f%%",
            ),
        },
    )
    st.session_state["patients_df"] = _merge_edited_rows(master, edited)

    if result.errors:
        with st.expander(f"Errores de procesamiento ({len(result.errors)})"):
            st.dataframe(pd.DataFrame(result.errors), hide_index=True)


def _render_preview() -> None:
    paths = [Path(path) for path in st.session_state.get("uploaded_paths", [])]
    if not paths:
        return
    st.subheader("Imagen de referencia")
    selected = st.selectbox(
        "Imagen",
        paths,
        format_func=lambda path: path.name,
    )
    st.image(str(selected), caption=selected.name, width="stretch")


def _render_download() -> None:
    st.subheader("Generar Excel")
    st.caption(
        "Las correcciones de la tabla se aplicarán antes de crear el archivo."
    )
    if st.button("Preparar Excel", type="primary"):
        result = st.session_state["processing_result"]
        apply_patient_edits(
            result.consolidation.patients,
            st.session_state["patients_df"],
        )
        output_path = Path(st.session_state["session_dir"]) / "pacientes.xlsx"
        export_results(
            result.consolidation,
            output_path,
            specialty_values=list(result.specialty_values),
        )
        st.session_state["excel_bytes"] = output_path.read_bytes()
        st.success("Excel preparado.")
    if excel_bytes := st.session_state.get("excel_bytes"):
        st.download_button(
            "Descargar pacientes.xlsx",
            data=excel_bytes,
            file_name="pacientes.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )


def main() -> None:
    st.set_page_config(
        page_title="Hospital OCR",
        page_icon="🏥",
        layout="wide",
    )
    if not st.session_state.get("old_sessions_cleaned"):
        cleanup_old_sessions(SESSIONS_ROOT)
        st.session_state["old_sessions_cleaned"] = True
    st.title("Hospital OCR")
    st.info(
        "Procesamiento local: las imágenes permanecen en esta computadora. "
        "No cierre la aplicación mientras el OCR está trabajando."
    )

    centers = load_centers(PROJECT_ROOT / "config" / "centros.csv")
    center_options = [*centers, OTHER_CENTER_SLUG]
    center_slug = st.selectbox(
        "Centro hospitalario",
        center_options,
        format_func=lambda slug: (
            "Otro centro de salud"
            if slug == OTHER_CENTER_SLUG
            else centers[slug]
        ),
    )
    custom_center_name = ""
    if center_slug == OTHER_CENTER_SLUG:
        custom_center_name = st.text_input(
            "Nombre del centro de salud",
            max_chars=120,
            help="Escriba el nombre oficial y, si es necesario, la ciudad o el estado.",
        ).strip()
    ocr_mode_label = st.radio(
        "Tipo de texto",
        ["Automático", "Manuscrito", "Impreso"],
        horizontal=True,
        help=(
            "Automático activa el refuerzo cuando detecta baja cobertura; "
            "Manuscrito refuerza renglones y rectifica celdas de cuadrícula; "
            "Impreso utiliza solamente el OCR normal."
        ),
    )
    ocr_mode: OcrMode = {
        "Automático": "auto",
        "Manuscrito": "handwritten",
        "Impreso": "printed",
    }[ocr_mode_label]
    uploads = st.file_uploader(
        "Imágenes de listas",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        accept_multiple_files=True,
        help="Máximo 15 MB por imagen.",
    )
    if st.button(
        "Procesar imágenes",
        disabled=not uploads or (
            center_slug == OTHER_CENTER_SLUG and not custom_center_name
        ),
        type="primary",
    ):
        try:
            _process_uploads(
                list(uploads),
                center_slug,
                custom_center_name,
                ocr_mode,
            )
            st.success("Procesamiento terminado. Revise los datos antes de descargar.")
        except Exception as error:
            st.error(f"No se pudo completar el procesamiento: {error}")

    if "processing_result" in st.session_state:
        review_tab, preview_tab, download_tab = st.tabs(
            ["Revisar", "Ver imágenes", "Descargar"]
        )
        with review_tab:
            _render_editor()
        with preview_tab:
            _render_preview()
        with download_tab:
            _render_download()

        st.divider()
        if st.button("Borrar esta sesión y sus archivos", type="secondary"):
            _clear_current_session()
            st.rerun()


main()
