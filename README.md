# UMSA Auditor

Panel interactivo Streamlit en español del experimento PLN de `repositorioUMSA.ipynb`: preparación del corpus, gráficos interactivos, figuras originales, similitud, clústeres, benchmark y chat con RAG sobre las tesis y la metodología. El cuaderno original se conserva sin modificaciones.

El proyecto migró desde el tema anterior de sismos (`SismoLab`) hacia el corpus académico de la UMSA. Toda la lógica de análisis vive en el paquete `umsa/` y se apoya en los artefactos de `dataset/`.

## Ejecutar con uv

Requisito: [uv](https://docs.astral.sh/uv/getting-started/installation/). La versión de Python está declarada en `.python-version`.

```bash
uv sync --frozen
uv run streamlit run app.py
```

Abrir <http://localhost:8501>. La app lee el corpus local; no depende de Google Colab ni de rutas absolutas. Las dependencias exactas están en `uv.lock`.

Para las verificaciones:

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check .
```

## Recorrido para presentar el proyecto

1. **Resumen:** tamaño del corpus, mediana del índice de rigor, citas, anomalías y estado de los artefactos. Incluye el contraste exploratorio con las notas reales del tribunal.
2. **Exploración:** filtros por año, modalidad y clúster temático; distribuciones, correlaciones, tutores con más tesis, términos dominantes y cumplimiento de las ocho secciones reglamentarias.
3. **Corpus:** búsqueda por título, autor o tutor; ficha de cada tesis con secciones presentes, perfil radial, conclusiones extraídas y tesis similares.
4. **Auditoría PDF:** subir un PDF nuevo o elegir uno de `dataset/PDFs/`. Genera puntaje 0–100, veredicto, diagnóstico explicable, cobertura de secciones, citas APA/IEEE, coherencia, estilometría, tesis similares y contraste opcional con OpenAlex. Cada informe queda en el registro local.
5. **Modelos:** benchmark ML vs. Deep Learning del cuaderno, modelo sustituto interpretable del índice, validación contra notas reales, correlaciones y anomalías detectadas por Isolation Forest.
6. **Registro:** auditorías persistidas en SQLite, descarga en JSON y eliminación controlada.
7. **Experimento:** ocho etapas visuales con pregunta, explicación del código, interpretación, celda original y datos descargables: preparación, tutores/metadatos, TF-IDF, coherencia/citas, clústeres, rigor/estilometría, modelos/validación y cobertura del cuaderno.
8. **Chat RAG y datos:** preguntas sobre docentes, rankings, años, estadísticas, cuaderno, benchmark o una tesis específica, con consultas calculadas, conversación, fuentes consultables y exportación JSON.

## Gráficos para explicar el experimento

En **Experimento**, selecciona la etapa que quieres presentar. Cada bloque conecta la operación del código con el resultado, incluye las celdas originales de referencia y permite consultar una tabla o descargar los datos del gráfico. Se distingue la figura original de su reconstrucción y de las visualizaciones adicionales de la app.

- **Datos y depuración:** contraste de la extracción inicial y v3, recorrido 79→73→71→70, cobertura de ocho secciones, ausencias por documento y distribuciones de modalidad/mención. El paso de 71 documentos corresponde a la salida histórica: el CSV final conservado ya tiene 70.
- **Tutores y metadatos:** reproduce las figuras de las celdas 23–25 y 28. El ranking permite alternar entre celdas sin limpiar, limpieza exacta de sufijos del cuaderno y conteo por persona de la app. Se explica por qué esos criterios producen cifras distintas. Los resúmenes ausentes se identifican y no se tratan como textos de longitud cero en el boxplot.
- **TF-IDF y similitud:** usa `min_df=2` y máximo 1.500 términos como el cuaderno, con escenarios alternativos identificados. La etapa inicial reproduce 73×1.500; el vocabulario efectivo de la etapa final puede ser menor que el límite. El mapa muestra 20 documentos inicialmente y permite seleccionar hasta 30. La distribución usa todos los pares distintos, excluyendo la diagonal. La comparación de dos tesis muestra los productos de pesos que suman el coseno y los textos para contrastarlos.
- **Coherencia y citación:** enfrenta las métricas iniciales con la estrategia por párrafos, dibuja la diagonal de igualdad y comunica cuántos pares válidos existen. También muestra patrones APA/IEEE y la distribución de coincidencias por tesis.
- **Clústeres:** curva histórica de silhouette para K=3…7, tamaño de grupos, términos originales y sus miembros. No se reentrena K-Means ni se atribuye significado ordinal a sus etiquetas.
- **Rigor y estilometría:** un gráfico de cascada descompone cada puntaje en sus cinco aportes con el p95 del corpus completo. Un escenario de citas permite ver el efecto del recorte sin modificar los datos. Se muestran longitudes de oraciones, ventanas MATTR y la frontera cero de Isolation Forest, explicando `contamination=0.10`.
- **Modelos y validación:** las cinco métricas del benchmark tienen escala 0–1; ROC-AUC y Accuracy incorporan la referencia 0,5. Sólo Accuracy conserva desviación entre folds y la muestra como barras de error. La relación con las diez notas reales se interpreta por separado. No se fabrican curvas ROC, pérdidas ni matrices de confusión históricas sin predicciones o historiales conservados.
- **Cuaderno y cobertura:** inventario de las cinco figuras originales (celdas 23, 24, 25, 28 y 33) y de los resultados originalmente tabulares/textuales que ahora tienen una representación visual. Conserva las imágenes, el código y las salidas originales.

## Chat con OpenRouter

La app lee `.env` desde la raíz sin mostrar sus valores. Las variables del entorno tienen prioridad. Para una instalación nueva, copia `.env.example` a `.env` y configura:

```dotenv
OPENROUTER_API_KEY=tu_clave_para_chat
OPENROUTER_API_KEY_EMBEDDINGS=tu_clave_para_embeddings
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

El asistente de GPT-4o mini puede combinar estas herramientas en un mismo turno:

- **consultar_datasets:** SQL de sólo lectura sobre todos los CSV/Parquet: conteos, rankings, medias, medianas (`median`), desviación muestral (`stddev`), filtros y cruces entre etapas.
- **buscar_proyectos:** búsqueda por título, tema, materias, resumen, autor, docente y año; incluye paginación, identificadores y enlaces al repositorio. La búsqueda es léxica y no sustituye un censo temático.
- **consultar_proyecto:** ficha unificada de cualquiera de los 79 registros, con docentes/roles, resumen, materias, conclusiones y métricas disponibles. Los campos ausentes permanecen `null`.
- **buscar_documentos:** fragmentos del cuaderno, documentación y tesis; puede limitarse a hasta tres PDFs identificados para evitar mezclar proyectos.
- **leer_pdf:** busca objetivos, metodología, tecnologías, pruebas o conclusiones dentro del PDF seleccionado y cita páginas físicas reales. Penaliza los índices y prioriza encabezados de sección. Los textos de hasta ocho PDFs se mantienen en caché; no hace OCR.
- **leer_pagina_pdf:** lectura de una página física concreta, numerada desde 1.
- **leer_cuaderno:** código y salidas de una celda original, con continuación mediante `offset` para celdas extensas.
- **consultar_documentacion:** instrucciones actuales de ejecución/configuración de la app desde README, separadas del contenido de las tesis.

Una búsqueda textual vacía no impide consultar tablas o fichas. Las fuentes muestran SQL y resultados, metadatos de proyectos, fragmentos o páginas PDF con identificadores `[F1]`, `[F2]`, etc. El chat conserva referencias ordenadas a proyectos/docentes de las respuestas anteriores para resolver seguimientos como «el segundo» o «compáralos»; vuelve a consultar las fuentes para responder. Un turno admite hasta siete rondas de herramientas y una respuesta final.

El catálogo distingue los 79 registros originales (`metadatos`) de las 70 tesis únicas (`corpus`). Las vistas `docentes_original` y `docentes_final` separan personas por punto y coma y distinguen tutores de asesores. Normalizan acentos, roles y variantes de nombres que son prefijos sin ambigüedad; conservan `nombre_original` para revisión. Por defecto, el ranking cuenta PDFs distintos por tutor o docente sin rol explícito en los 79 registros originales. Los asesores se incluyen cuando se solicita participación o asesoría. No se fusionan nombres ambiguos. El gráfico de tutores de Exploración usa el mismo criterio sobre el corpus filtrado.

La tabla **proyectos** une los 79 metadatos con las métricas finales sin descartar los nueve registros excluidos del modelado; incluye `en_corpus_final` y `pdf_disponible`. **materias** separa las etiquetas originales por proyecto. El selector «Una tesis» cubre los 79 registros. Si falta un PDF, se conservan su ficha y los datos extraídos disponibles; el asistente debe indicarlo sin inventar páginas ni métricas.

`motivo_exclusion` conserva el retiro explícito registrado en el cuaderno y marca como no documentados los motivos individuales que no constan. La exclusión del corpus no se presenta como un juicio de calidad de la tesis. Las consultas de documentación también reciben los identificadores de los modelos activos, sin sus claves, para distinguir el chat de los modelos del experimento.

La consulta SQL se ejecuta en SQLite en memoria: no puede modificar los datasets, adjuntar bases, cargar extensiones ni acceder al sistema. Tiene un límite de ejecución y devuelve hasta 100 filas. El ámbito «Una tesis» filtra las tablas antes de consultar. Los textos extensos se limitan en los resultados tabulares; para contenido se usa la recuperación documental.

El índice documental TF-IDF comprende el texto completo y las conclusiones de las 70 tesis, sus fichas, el benchmark y el código/resultados textuales del cuaderno. Se fragmenta con 1.800 caracteres y 250 de solapamiento. Recupera hasta 24 candidatos; los embeddings opcionales reordenan estos candidatos. Se envían hasta seis fragmentos por búsqueda, junto con los últimos ocho turnos. Los localizadores indican archivo, sección/celda y desplazamiento de caracteres, no páginas PDF. El índice documental se construye sólo cuando se necesita.

La generación usa la [API de chat y herramientas de OpenRouter](https://openrouter.ai/docs/guides/features/tool-calling) y el reordenamiento usa su [API de embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings). Las consultas consumen el saldo de esas claves. No se hacen solicitudes al abrir el panel; sólo al enviar una pregunta. El modo **Sólo buscar evidencia** es un buscador documental sin generación; para que sea totalmente local desactiva también **Reordenar con embeddings**. Si embeddings falla, se conserva la recuperación léxica. El chat normal calcula datos con las herramientas y puede corregir consultas dentro del mismo turno antes de responder.

Las claves permanecen en el servidor y `.env` se excluye de Git y Docker. Compose inyecta sus variables mediante `env_file` opcional. El historial pertenece a la sesión de Streamlit y puede descargarse o reiniciarse. El chat informa al usuario que las preguntas y fragmentos seleccionados se envían a OpenRouter.

## Qué mide el índice de rigor

El índice reproduce la fórmula del cuaderno:

| Componente | Peso |
|---|---:|
| Estructura reglamentaria (8 secciones) | 25% |
| Coherencia objetivos ↔ marco aplicativo | 25% |
| Coherencia objetivos ↔ conclusiones | 15% |
| Citas detectadas (APA + IEEE) | 20% |
| Extensión en páginas | 15% |

Umbrales del veredicto: **≥ 72** sólido, **≥ 60** aceptable con observaciones, por debajo **deficiente**.

La coherencia se aproxima con TF-IDF y similitud de coseno para evitar dependencias pesadas de `torch`/`sentence-transformers`. El cuaderno original usa embeddings; los valores no son idénticos, pero la app documenta la diferencia y mantiene el mismo protocolo de comparación contra percentiles del corpus.

## Archivos de datos utilizados

| Archivo | Uso |
|---|---|
| `dataset/df_maestro_final_modelos_70.parquet` | Corpus de 70 tesis con texto, métricas, clúster y anomalías. |
| `dataset/resultados_modelos_umsa_70.csv` | Respaldo ligero del corpus si falta el Parquet. |
| `dataset/auditor_umsa_produccion.joblib` | Percentiles, metadatos y embeddings de referencia. |
| `dataset/benchmark_ml_vs_dl_umsa.csv` | Resultados comparativos del cuaderno. |
| `dataset/metadatos_informatica.csv` | Metadatos originales del repositorio. |
| `dataset/PDFs/` | Documentos para auditoría y comparación interna. |

## Límites y advertencias

- El índice es una **señal de apoyo**, no una calificación oficial ni una decisión automática de aprobación.
- La validación con notas reales usa una muestra pequeña (10 actas). En esta muestra la correlación de Spearman es negativa y no significativa; la app la muestra sin ocultarla.
- El modelo logístico de la pestaña Modelos es un **sustituto** del índice: comparte sus variables, por lo que su AUC alto es esperado y no demuestra validez predictiva.
- La coherencia TF-IDF mide solapamiento léxico, no comprensión semántica profunda.
- Isolation Forest marca perfiles atípicos; una anomalía no implica fraude ni mala calidad.
- OpenAlex requiere internet y sólo aporta contexto externo; no modifica el puntaje.

## Estructura

```
app.py                 Interfaz Streamlit
umsa/data.py           Carga del corpus, secciones, citas, estilometría y similitud
umsa/audit.py          Extracción de PDF y auditoría explicable
umsa/modeling.py       Modelo sustituto, correlaciones y validación con notas
umsa/storage.py        Registro SQLite de auditorías
umsa/notebook.py       Celdas, figuras originales y fragmentación
umsa/rag.py            Recuperación y cliente OpenRouter
umsa/analytics.py      Catálogo, normalización de docentes y consultas de sólo lectura
umsa/research.py       Búsqueda/fichas de proyectos, lectura PDF por página y documentación
umsa/presentation.py   Laboratorio interactivo y chat con fuentes
umsa/figures.py        Cálculos y figuras reproducibles del cuaderno
umsa/lessons.py        Recorrido visual por etapas con código y datos
tests/                 Pruebas del pipeline y de la interfaz
dataset/               Corpus, artefactos y PDFs
repositorioUMSA.ipynb  Cuaderno original de referencia
```

## Docker

```bash
docker compose up --build
```

La app escucha en `127.0.0.1:8501` y persiste el registro en el volumen `umsa_data`. El directorio de datos se puede cambiar con `UMSA_DATA_DIR`.
