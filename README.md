# Hospital List Images to Excel

Pipeline local para transformar imágenes de listas hospitalarias en archivos Excel.

## Estructura

- `src/hospital_ocr/`: código fuente del pipeline OCR.
- `tests/`: pruebas automatizadas.
- `config/centros.csv`: relación entre carpetas y nombres oficiales de centros.
- `config/especialidades.csv`: equivalencias para normalizar especialidades y áreas.
- `config/nombres_comunes.csv`: catálogo configurable de nombres frecuentes.
- `config/apellidos_comunes.csv`: catálogo configurable de apellidos frecuentes.
- `data/input/images/`: imágenes originales, clasificadas en una subcarpeta por centro; su contenido no se guarda en Git.
- `data/interim/`: imágenes preprocesadas y resultados OCR para auditoría.
- `data/output/`: archivos Excel finales.

Cada subcarpeta de imágenes se relaciona con el nombre oficial del centro mediante `config/centros.csv`.

## Preparación

```powershell
conda activate hospital-ocr
python -m pip install -e . --no-deps
```

PaddleOCR descarga sus modelos oficiales durante la primera ejecución. El procesamiento posterior se realiza localmente.

## Uso

Piloto distribuido de cinco imágenes:

```powershell
hospital-ocr --limit 5 --output data/output/piloto_pacientes.xlsx
```

Si el archivo ya existe, el comando se detiene para proteger correcciones manuales. Solo debe reemplazarse de forma intencional:

```powershell
hospital-ocr --limit 5 --output data/output/piloto_pacientes.xlsx --force
```

Todas las imágenes:

```powershell
hospital-ocr
```

El Excel contiene:

- `Plantilla`: columnas `nombre`, `apellido`, `cedula`, `centro` y `edad_sector`.
- `Pacientes`: una fila consolidada por paciente, con nombre completo, separación confiable de nombres y apellidos, revisión, duplicados, imágenes de origen y confianza del OCR.
- `Diccionario`: significado, tipo, valores permitidos y ejemplo de cada columna.

`Plantilla` es una vista automática protegida con 1.000 filas enlazadas mediante fórmulas tradicionales. Los cambios realizados en `Pacientes` se reflejan al recalcular el libro. El archivo se configura para recalcular al abrirse y no se genera un CSV adicional.

`Pacientes` se entrega como tabla con filtros. `sexo`, `unidad_edad`, `especialidad` y `estado_revision` tienen listas desplegables. Las filas pendientes y los duplicados se resaltan mediante formato condicional.

Solo se fusionan automáticamente registros del mismo centro con nombre y edad exactos y sin conflictos de sexo o procedencia. Las coincidencias aproximadas permanecen separadas y pasan a revisión.

En `Pacientes`, `estado_duplicado` distingue registros únicos, posibles duplicados y duplicados consolidados. `detalle_duplicado` identifica las filas relacionadas y explica la coincidencia. Los posibles duplicados se resaltan en naranja y los consolidados en azul.

`edad` y `unidad_edad` se conservan por separado en las hojas internas para distinguir años, meses y días. En `Plantilla` se combinan dentro de `edad_sector`.

Cada palabra de `nombre_completo` se contrasta con los catálogos de nombres y apellidos. `nombre` y `apellido` solo se completan cuando la confianza es al menos 85% y existe una diferencia clara frente a otras separaciones posibles. Los casos ambiguos quedan vacíos y pendientes de revisión.

`estado_revision` muestra `Pendiente` cuando el registro necesita verificación y `No requerido` cuando no presenta alertas. La separación entre nombres y apellidos, los datos ausentes y las coincidencias dudosas siempre deben revisarse antes de utilizar la hoja `Plantilla`.

> Las imágenes y los resultados pueden contener datos médicos sensibles y están excluidos del repositorio.
