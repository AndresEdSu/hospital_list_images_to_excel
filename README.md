# Hospital OCR

Herramienta local para convertir imágenes de listas hospitalarias en un archivo Excel revisable. El proyecto usa OCR, reglas de interpretación y catálogos configurables para extraer pacientes, cédulas, edades, procedencias, especialidades y posibles duplicados.

El procesamiento está pensado para ejecutarse en la computadora del usuario. Las imágenes, resultados intermedios, caché y archivos Excel generados no se guardan en Git.

## Qué Hace

- Procesa una o varias imágenes de listas hospitalarias.
- Detecta tablas, filas, columnas y listas escritas en formato libre.
- Extrae nombre completo, nombre, apellido, cédula, edad, sexo, procedencia, especialidad, área y observaciones.
- Consolida apariciones repetidas de un mismo paciente dentro de un centro.
- Marca posibles duplicados sin fusionarlos automáticamente.
- Genera un Excel con hojas `Pacientes`, `Plantilla` y `Diccionario`.
- Permite revisar y corregir los datos antes de descargar el Excel desde la interfaz web.

## Requisitos

- Windows, macOS o Linux.
- Python `>=3.11,<3.12`.
- Un entorno con las dependencias del proyecto instaladas.
- Conexión a internet solo para instalar dependencias y descargar los modelos OCR la primera vez.

El OCR se ejecuta localmente. La primera corrida puede tardar porque PaddleOCR descarga modelos en `.cache/paddlex`. Las siguientes corridas son más rápidas gracias a la caché OCR.

## Instalación

Con Conda:

```powershell
conda create -n hospital-ocr python=3.11
conda activate hospital-ocr
python -m pip install -e .
```

Si ya tienes un entorno preparado con las dependencias pesadas instaladas:

```powershell
conda activate hospital-ocr
python -m pip install -e . --no-deps
```

Para instalar también dependencias de pruebas:

```powershell
python -m pip install -e ".[dev]"
```

## Uso Rápido

### Interfaz Web

```powershell
conda activate hospital-ocr
hospital-ocr-web
```

La aplicación abre Streamlit en `127.0.0.1`. Desde ahí puedes:

- Elegir el centro hospitalario.
- Cargar varias imágenes.
- Seleccionar el tipo de texto: `Automático`, `Manuscrito` o `Impreso`.
- Ejecutar el OCR.
- Revisar pacientes, pendientes y posibles duplicados.
- Comparar con la imagen original.
- Descargar `pacientes.xlsx`.

Si el centro no aparece en el catálogo, selecciona `Otro centro de salud` y escribe su nombre oficial. Ese valor se usa solo en la sesión actual.

Hay una imagen ficticia para preparar capturas y probar la app sin datos reales:

![Lista hospitalaria ficticia](docs/examples/fake_hospital_list.png)

Puedes cargar `docs/examples/fake_hospital_list.png`, seleccionar `Otro centro de salud` y usar `Hospital Demo San Gabriel` como nombre del centro.

Vista de la interfaz web luego de procesar la imagen de ejemplo:

![Interfaz web con revisión de pacientes](docs/examples/fake_app.png)

### Línea de Comandos

La línea de comandos espera imágenes organizadas por centro:

```text
data/input/images/
  hospital_domingo_luciani/
    lista_01.jpg
    lista_02.jpg
  hospital_miguel_perez_carreno/
    lista_03.jpg
```

Cada carpeta debe coincidir con un valor de la columna `carpeta` en `config/centros.csv`.

Procesar todas las imágenes:

```powershell
hospital-ocr
```

Procesar una muestra distribuida de 5 imágenes:

```powershell
hospital-ocr --limit 5 --output data/output/piloto_pacientes.xlsx
```

Reemplazar un Excel existente de forma intencional:

```powershell
hospital-ocr --output data/output/pacientes_consolidados.xlsx --force
```

Elegir modo OCR:

```powershell
hospital-ocr --ocr-mode auto
hospital-ocr --ocr-mode handwritten
hospital-ocr --ocr-mode printed
```

## Modos OCR

- `auto`: modo recomendado. Ejecuta OCR global y refuerza celdas o renglones cuando detecta baja cobertura, campos estructurados o baja confianza. El refuerzo solo se acepta si mejora calidad y conserva cobertura.
- `handwritten`: intenta refuerzo con mayor sensibilidad para listas manuscritas. Si el refuerzo empeora el resultado, se conserva el OCR global.
- `printed`: usa una sola pasada global. Es útil para documentos impresos o cuando se quiere evitar el refuerzo.

Los resultados OCR se guardan en caché por contenido de imagen, modo OCR y versión de política. Si cambia la política OCR, la caché vieja se ignora automáticamente.

## Estructura del Excel

El archivo generado contiene:

- `Pacientes`: hoja principal editable, con una fila por paciente consolidado.
- `Plantilla`: vista protegida con las columnas finales `nombre`, `apellido`, `cedula`, `centro` y `edad_sector`.
- `Diccionario`: descripción de cada columna, tipo de dato, valores permitidos y ejemplos.

Ejemplo de la hoja `Pacientes`:

![Hoja Pacientes del Excel generado](docs/examples/fake_pacientes.png)

Columnas importantes en `Pacientes`:

- `estado_revision`: `Pendiente`, `No requerido` o `Revisado`.
- `observaciones`: motivos de revisión, conflictos o notas clínicas extraídas.
- `estado_duplicado`: `Único`, `Posible duplicado` o `Duplicado consolidado`.
- `detalle_duplicado`: explica con qué paciente coincide y por qué.
- `imagenes_origen`: imágenes donde apareció el paciente.
- `linea_ocr_original`: texto OCR original usado como evidencia.
- `confianza_*`: confianza separada para OCR, nombre, cédula, edad, procedencia y especialidad.

La hoja `Plantilla` se alimenta desde `Pacientes`. Haz las correcciones en `Pacientes` y abre/recalcula el libro para que `Plantilla` refleje los cambios.

Ejemplo de la hoja `Plantilla`:

![Hoja Plantilla del Excel generado](docs/examples/fake_plantilla.png)

## Revisión Humana

El sistema no reemplaza la revisión humana. Antes de usar la hoja `Plantilla`, revisa especialmente:

- Filas con `estado_revision = Pendiente`.
- Filas con `estado_duplicado = Posible duplicado`.
- Registros sin cédula.
- Nombres o apellidos vacíos.
- Procedencias o especialidades dudosas.
- Registros con observaciones o conflictos.

Las coincidencias aproximadas de nombres no se fusionan automáticamente. Se marcan como posibles duplicados para que una persona decida.

## Catálogos Configurables

Los catálogos viven en `config/`:

- `centros.csv`: relación entre carpeta y nombre oficial del centro.
- `lugares.csv`: aliases de ciudades, estados, sectores, instituciones y procedencias.
- `especialidades.csv`: aliases de especialidades y áreas.
- `nombres_comunes.csv`: nombres frecuentes para separar nombres/apellidos y corregir algunos errores OCR.
- `apellidos_comunes.csv`: apellidos frecuentes.

Cuando agregues aliases, procura usar términos específicos y revisar que no creen coincidencias ambiguas.

## Evaluación con Imágenes de Prueba

El corpus privado de evaluación se guarda en `data/evaluation/test_images/`. Cada imagen debe tener un CSV con el mismo nombre base, codificado en UTF-8 y separado por punto y coma.

Validar estructura sin ejecutar OCR:

```powershell
hospital-ocr-evaluate data/evaluation/test_images --validate-only
```

Ejecutar evaluación en modo automático:

```powershell
hospital-ocr-evaluate data/evaluation/test_images --ocr-mode auto
```

Evaluar solo algunas imágenes:

```powershell
hospital-ocr-evaluate data/evaluation/test_images --ocr-mode auto --only test_image_8 --only test_image_17
```

Recalcular métricas desde predicciones existentes sin repetir OCR:

```powershell
hospital-ocr-evaluate data/evaluation/test_images --from-predictions data/evaluation/test_images/results/predicciones
```

Los resultados incluyen resumen JSON, métricas por imagen/campo, diferencias, predicciones y artefactos OCR.

## Directorios

- `src/hospital_ocr/`: código fuente del pipeline.
- `src/hospital_ocr/table_extraction/`: detección y parsing de tablas.
- `tests/`: pruebas automatizadas.
- `config/`: catálogos editables.
- `data/input/images/`: imágenes de entrada para CLI, excluidas de Git.
- `data/interim/`: preprocesamiento, OCR, auditoría y sesiones web, excluido de Git.
- `data/output/`: Excel generados, excluidos de Git.
- `data/evaluation/`: corpus privado y resultados de evaluación, excluido de Git.
- `.cache/paddlex/`: modelos y caché OCR local, excluido de Git.

## Privacidad

Las imágenes pueden contener datos médicos sensibles. Por eso el repositorio ignora:

- Imágenes de entrada.
- Corpus de evaluación.
- Resultados OCR.
- Archivos intermedios.
- Excel finales.
- Caché local.

No subas al repositorio archivos con datos reales de pacientes.

## Desarrollo

Ejecutar pruebas:

```powershell
python -m pytest -q
```

Ejecutar solo pruebas de modos OCR:

```powershell
python -m pytest tests/test_pipeline_modes.py -q
```

Ejecutar solo pruebas de consolidación:

```powershell
python -m pytest tests/test_consolidation.py -q
```

## Limitaciones Conocidas

- El OCR puede fallar con texto muy borroso, inclinado o manuscrito irregular.
- Los modos `auto` y `handwritten` pueden tardar más porque prueban refuerzos por celda o renglón.
- La separación entre nombres y apellidos depende de catálogos y confianza mínima.
- Procedencias fuera del catálogo pueden quedar vacías o pendientes.
- Los posibles duplicados requieren revisión humana.

## Fuente Inicial del Catálogo de Centros

El catálogo parte de hospitales y centros de referencia de Venezuela. Se tomó como base la lista reproducida en el Plan intersectorial de preparación y atención a la COVID-19 de Naciones Unidas en Venezuela, páginas 31 a 33, y se agregaron centros identificados para el proyecto. Es una lista operativa, no un ranking de calidad ni un directorio exhaustivo.
