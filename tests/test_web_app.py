from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_home_page_starts_without_errors() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hospital_ocr"
        / "web_app.py"
    )

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert len(app.exception) == 0
    assert [title.value for title in app.title] == ["Hospital OCR"]
    assert [button.label for button in app.button] == ["Procesar imágenes"]
    assert [selectbox.label for selectbox in app.selectbox] == [
        "Centro hospitalario"
    ]
    assert [radio.label for radio in app.radio] == ["Tipo de texto"]
    assert app.radio[0].options == ["Automático", "Manuscrito", "Impreso"]
    assert len(app.get("file_uploader")) == 1


def test_other_center_requires_its_name() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hospital_ocr"
        / "web_app.py"
    )
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    app.selectbox[0].select("otro_centro").run(timeout=30)

    assert [field.label for field in app.text_input] == [
        "Nombre del centro de salud"
    ]
    assert app.button[0].disabled
