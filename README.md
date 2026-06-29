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

El catálogo parte de 46 hospitales y centros de referencia de todos los estados.
Se tomó como base la lista de hospitales y centros centinela reproducida en el
[Plan intersectorial de preparación y atención a la COVID-19 de Naciones Unidas
en Venezuela](https://venezuela.un.org/es/download/48785/88968), páginas 31 a
33. A esa base se incorporan centros adicionales identificados para el proyecto,
por lo que el total puede ser mayor. Es una lista operativa nacional, no un
ranking de calidad ni un directorio exhaustivo, y la propia fuente indica que
está sujeta a actualización.

## Preparación

```powershell
conda activate hospital-ocr
python -m pip install -e . --no-deps
```

PaddleOCR descarga sus modelos oficiales durante la primera ejecución. El procesamiento posterior se realiza localmente.

## Uso

### Interfaz web local

```powershell
conda activate hospital-ocr
hospital-ocr-web
```

La aplicación abre una página local en el navegador. Permite seleccionar el centro, cargar varias imágenes, seguir el progreso, revisar y corregir pacientes, comparar con las imágenes y descargar el Excel. No expone el servidor fuera de `127.0.0.1` y limita cada archivo a 15 MB.

Las imágenes seleccionadas en Streamlit pueden estar en cualquier carpeta de
la computadora. La aplicación las copia a una sesión temporal y les asigna el
centro escogido; el nombre de su carpeta original no tiene ningún efecto.

Si el centro no aparece en el catálogo, seleccione `Otro centro de salud` y
escriba su nombre oficial. Este valor se utiliza en el Excel de esa sesión, pero
no se agrega automáticamente a `config/centros.csv`, para evitar duplicados y
errores ortográficos en la lista compartida.

Los archivos temporales se guardan en `data/interim/web_sessions/`. Pueden eliminarse desde la interfaz y las sesiones con más de 24 horas se limpian automáticamente.

### Línea de comandos

La organización por subcarpetas solo es obligatoria en este modo. Cada carpeta
dentro de `data/input/images/` debe usar exactamente un identificador de la
columna `carpeta` de `config/centros.csv`. Las imágenes sueltas o ubicadas en
carpetas desconocidas se omiten y se reportan como errores.

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

El pipeline detecta automáticamente las tablas con columnas de nombre, cédula,
edad, sexo, procedencia y plan. Para otros diseños conserva el análisis de
listas libres. En las tablas, `Plan` se conserva dentro de `observaciones` y no
se interpreta automáticamente como especialidad.

`Plantilla` es una vista automática protegida con 1.000 filas enlazadas mediante fórmulas tradicionales. Los cambios realizados en `Pacientes` se reflejan al recalcular el libro. El archivo se configura para recalcular al abrirse y no se genera un CSV adicional.

`Pacientes` se entrega como tabla con filtros. `sexo`, `unidad_edad`, `especialidad` y `estado_revision` tienen listas desplegables. Las filas pendientes y los duplicados se resaltan mediante formato condicional.

La cédula es el criterio principal para fusionar apariciones del mismo paciente
dentro de un centro. Cuando no está disponible, solo se fusionan registros con
nombre y edad exactos y sin conflictos de sexo o procedencia. Las coincidencias
aproximadas permanecen separadas y pasan a revisión.

En `Pacientes`, `estado_duplicado` distingue registros únicos, posibles duplicados y duplicados consolidados. `detalle_duplicado` identifica las filas relacionadas y explica la coincidencia. Los posibles duplicados se resaltan en naranja y los consolidados en azul.

`edad` y `unidad_edad` se conservan por separado en las hojas internas para distinguir años, meses y días. En `Plantilla` se combinan dentro de `edad_sector`.

Cada palabra de `nombre_completo` se contrasta con los catálogos de nombres y apellidos. `nombre` y `apellido` solo se completan cuando la confianza es al menos 85% y existe una diferencia clara frente a otras separaciones posibles. Los casos ambiguos quedan vacíos y pendientes de revisión.

`estado_revision` muestra `Pendiente` cuando el registro necesita verificación y `No requerido` cuando no presenta alertas. La separación entre nombres y apellidos, los datos ausentes y las coincidencias dudosas siempre deben revisarse antes de utilizar la hoja `Plantilla`.

> Las imágenes y los resultados pueden contener datos médicos sensibles y están excluidos del repositorio.
