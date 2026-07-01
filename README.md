# Hospital List Images to Excel

Pipeline local para transformar imágenes de listas hospitalarias en archivos Excel.

## Estructura

- `src/hospital_ocr/`: código fuente del pipeline OCR.
- `tests/`: pruebas automatizadas.
- `config/centros.csv`: relación entre carpetas y nombres oficiales de centros.
- `config/lugares.csv`: equivalencias configurables de ciudades, sectores y localidades.
- `config/especialidades.csv`: equivalencias para normalizar especialidades y áreas.
- `config/nombres_comunes.csv`: catálogo configurable de nombres frecuentes.
- `config/apellidos_comunes.csv`: catálogo configurable de apellidos frecuentes.
- `data/input/images/`: imágenes originales, clasificadas en una subcarpeta por centro; su contenido no se guarda en Git.
- `data/interim/`: imágenes preprocesadas, resultados OCR y cuadrículas detectadas para auditoría.
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

La aplicación abre una página local en el navegador. Permite seleccionar el centro, cargar varias imágenes, seguir el progreso por etapa e imagen, revisar y corregir pacientes, comparar con las imágenes y descargar el Excel. No expone el servidor fuera de `127.0.0.1` y limita cada archivo a 15 MB.

Las imágenes seleccionadas en Streamlit pueden estar en cualquier carpeta de
la computadora. La aplicación las copia a una sesión temporal y les asigna el
centro escogido; el nombre de su carpeta original no tiene ningún efecto.
El Excel conserva el orden de carga y el orden visual de arriba hacia abajo
dentro de cada imagen. Si varias apariciones se consolidan, el paciente mantiene
la posición de la primera.

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
edad, sexo, procedencia, especialidad y plan. Los encabezados se comparan con
aliases comunes y sus coordenadas se utilizan para inferir los límites y el
orden real de las columnas. Cuando existen bordes visibles, las líneas
horizontales y verticales se detectan sobre una copia de la imagen y cada caja
OCR se asigna por solapamiento a una celda física. Esto permite seguir filas
inclinadas o afectadas por perspectiva. Sin encabezados, cada fila se clasifica por formato
y catálogo: patrón documental para cédula, rango y unidad para edad, valores
cerrados para sexo y aliases para procedencia y especialidad. La columna de
nombres se aprende por repetición, variedad y alineación, sin asumir una
posición fija. La detección de la tabla combina regularidad entre filas,
índices consecutivos y columnas auxiliares; edad, sexo y cédula son señales
opcionales, no requisitos. En las tablas, `Plan` se conserva dentro de
`observaciones` y no se interpreta automáticamente como especialidad.

Las listas donde cada fila aparece en una sola caja OCR se distinguen de las
tablas por su estructura. Una página se considera lista en línea cuando varias
filas contienen un nombre y al menos otro campo reconocible —edad, cédula,
sexo, procedencia o especialidad—. Una vez identificado ese patrón, también se
conservan pacientes de la misma página que solo tengan nombre.

Las superposiciones de auditoría de las cuadrículas aceptadas se guardan en
`data/interim/grids/`. Si las líneas son débiles, incompletas o la confianza no
alcanza el umbral, el procesamiento continúa con la agrupación semántica y
geométrica anterior.

En documentos sin cuadrícula, si el OCR inicial cubre pocos renglones, se
activa un segundo pase para escritura manuscrita. Los renglones se detectan por
concentración de trazos, se amplían y se leen en dos ventanas horizontales
solapadas; luego sus fragmentos se alinean de nuevo en una sola fila lógica.
Los recortes y el resumen de cobertura se guardan en
`data/interim/handwriting_rows/` para auditoría.

La interfaz permite seleccionar `Automático`, `Manuscrito` o `Impreso`.
En listas sin cuadrícula, `Manuscrito` procesa directamente todos los
renglones reforzados. Cuando existe una cuadrícula, rectifica sus celdas con
perspectiva, conserva su fila y columna de origen y compara esos resultados con
el OCR global para no degradar celdas que ya se leyeron correctamente.
`Impreso` desactiva por completo el pase reforzado. En la línea de comandos se
ofrecen los mismos modos mediante
`--ocr-mode auto|handwritten|printed`.

`Plantilla` es una vista automática protegida con 1.000 filas enlazadas mediante fórmulas tradicionales. Los cambios realizados en `Pacientes` se reflejan al recalcular el libro. El archivo se configura para recalcular al abrirse y no se genera un CSV adicional.

`Pacientes` se entrega como tabla con filtros. `sexo`, `unidad_edad` —mostrada
como `Unidad de edad` en la interfaz—, `especialidad` y `estado_revision` tienen
listas desplegables. Las filas pendientes y los duplicados se resaltan mediante
formato condicional.

La cédula es el criterio principal para fusionar apariciones del mismo paciente
dentro de un centro. Cuando no está disponible, solo se fusionan registros con
nombre y edad exactos y sin conflictos de sexo o procedencia. Las coincidencias
aproximadas permanecen separadas y pasan a revisión.

En `Pacientes`, `estado_duplicado` distingue registros únicos, posibles duplicados y duplicados consolidados. `detalle_duplicado` identifica las filas relacionadas y explica la coincidencia. Los posibles duplicados se resaltan en naranja y los consolidados en azul.

`edad` y `unidad_edad` se conservan por separado en las hojas internas para distinguir años, meses y días. En `Plantilla` se combinan dentro de `edad_sector`.

Cada palabra de `nombre_completo` se contrasta con los catálogos de nombres y apellidos. `nombre` y `apellido` solo se completan cuando la confianza es al menos 85% y existe una diferencia clara frente a otras separaciones posibles. Los casos ambiguos quedan vacíos y pendientes de revisión.

Las procedencias y especialidades también comparan aliases en una representación
compacta para tolerar espacios insertados o eliminados por el OCR. Las
coincidencias más largas y específicas tienen prioridad sobre aliases parciales.
Esta corrección de espacios no se aplica automáticamente a nombres personales.
Dentro de una columna confirmada como procedencia, una coincidencia OCR más
tolerante se acepta solamente cuando el mejor lugar del catálogo supera el
umbral contextual y aventaja claramente al segundo candidato. El valor canónico
se exporta y la normalización queda indicada para revisión.

En una columna confirmada como sexo, `F` y `M` tienen prioridad. `H` se
normaliza a `M`; las confusiones OCR `T`, `E` o `P` se interpretan como `F` y
`N` como `M`, siempre marcadas para revisión. Si aparecen valores incompatibles
como `F` y `M`, el campo queda vacío. En todos los casos
`linea_ocr_original` conserva la fila reconocida sin modificar.

La hoja `Pacientes` incluye confianzas separadas para nombre, cédula, edad,
procedencia y especialidad, además de la evidencia usada para clasificar cada
campo. La confianza OCR sigue representando solamente la calidad del texto
reconocido.

Los índices iniciales de filas numeradas se descartan antes de analizar nombre
y edad. Fuera de una columna identificada explícitamente como procedencia, un
lugar solo se completa cuando coincide con `config/lugares.csv`; el texto
restante no se acepta automáticamente como procedencia. Dentro de una columna
explícita se conserva el valor desconocido, marcado para revisión.

Las columnas intermedias que no forman parte de la salida, como `Cama`,
`Afiliación` y `Diagnóstico`, se incluyen al calcular los límites de la tabla y
luego se ignoran. También se detectan como separadores neutrales los encabezados
desconocidos que estén geométricamente alineados con el membrete, incluso cuando
la fotografía esté inclinada. De este modo, el contenido de una columna nueva
no invade las columnas adyacentes que sí se exportan.

`estado_revision` muestra `Pendiente` cuando el registro necesita verificación y `No requerido` cuando no presenta alertas. La separación entre nombres y apellidos, los datos ausentes y las coincidencias dudosas siempre deben revisarse antes de utilizar la hoja `Plantilla`.

> Las imágenes y los resultados pueden contener datos médicos sensibles y están excluidos del repositorio.
