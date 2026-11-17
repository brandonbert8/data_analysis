# SismoLab

Aplicación Streamlit en español que convierte `sismos_integrado.ipynb` en un flujo reproducible de exploración, limpieza, entrenamiento, evaluación y validación con eventos nuevos. El cuaderno original se conserva sin modificaciones.

## Ejecutar con uv

Requisito: [uv](https://docs.astral.sh/uv/getting-started/installation/). La versión de Python está declarada en `.python-version`; uv puede instalarla automáticamente.

```bash
uv sync --frozen
uv run streamlit run app.py
```

Abrir <http://localhost:8501>. El catálogo incluido contiene 1.050 eventos y se carga sin depender de Colab ni de descargas externas. Las dependencias exactas están en `uv.lock`; no es necesario activar manualmente `.venv`.

Para las verificaciones:

```bash
uv sync --frozen --group dev
uv run pytest -q
```

## Recorrido para presentar el proyecto

1. **Resumen:** pregunta de negocio, cobertura del catálogo, positivos elegibles y eventos pendientes.
2. **Preprocesamiento:** filtrar la exploración por fechas/país, examinar magnitudes, profundidad, epicentros, correlaciones y Gutenberg–Richter. Revisar conversiones, nulos, reglas físicas y rechazos. Exportar catálogo limpio y auditoría.
3. **Modelo:** observar las fechas de entrenamiento/validación/prueba y sus embargos; comparar F19/F25 y familias. Ejecutar entrenamiento y revisar cada fold, configuración y candidato no seleccionable.
4. **Evaluación:** comparar con referencias constante y de actividad reciente, inspeccionar curvas PR/ROC, matriz de confusión, calibración y scores temporales. Simular umbrales; exportar métricas y predicciones.
5. **Nuevos datos:** cargar un CSV o registrar un evento, generar scores con un modelo congelado y evaluar únicamente horizontes completamente observados. Incorporar los eventos al catálogo si corresponde y entrenar después otra versión.
6. **Registro:** recuperar experimentos y validaciones, consultar procedencia y versiones, y descargar evidencia del experimento en ZIP.

## Qué estima y qué no demuestra

Por cada evento en `t`, la etiqueta vale 1 si existe **otro evento** con magnitud ≥ 5 dentro de su misma celda de 0,5° × 0,5°, en `(t, t + 7 días]`. Horizonte, magnitud y celda se pueden modificar en la barra lateral.

Esto es un estudio estadístico condicionado a eventos presentes en un catálogo. No es una estimación diaria para cualquier punto del mapa, ni una predicción sísmica científicamente validada. Las salidas se presentan como **scores**; el balanceo de clases puede afectar su calibración. No utilizar para decisiones de emergencia.

El CSV tiene 54 eventos M ≥ 5, pero eso **no significa 54 etiquetas positivas**: la etiqueta se refiere a un evento fuerte posterior en la misma celda. Con los parámetros originales hay 21 positivos observables; las exclusiones de historia y los embargos reducen los usados al entrenar.

En la separación predeterminada del catálogo original quedan 607 eventos/4 positivos en entrenamiento, 196/4 en validación y 206/12 en prueba; 19 eventos quedan en embargo. Con 3 folds sólo uno permite medir AP después de exigir ambas clases en entrenamiento y positivos en validación. Es una limitación real que la interfaz expone, no una certificación del modelo. Los resultados nuevos no deben compararse directamente con las cifras guardadas del cuaderno, porque cambió el protocolo.

## Protocolo de modelado

- Ventanas históricas `[t−ventana, t)` sin eventos simultáneos; etiqueta futura `(t, t+h]`. Las variables predictoras nunca incluyen `future_max`, `target`, fecha de actualización ni datos del futuro.
- El 60% de las fechas elegibles define entrenamiento, el siguiente 20% validación y el último 20% prueba. Las proporciones finales cambian al aplicar un embargo estricto de `h` días en **ambas** fronteras. Eventos simultáneos permanecen juntos.
- Primeros 30 días fuera del modelado por historia incompleta. Últimos `h` días sin etiquetas; permanecen disponibles para scoring.
- Dentro del entrenamiento: CV de origen expansivo con embargo por **días**, no por número de filas. Imputación mediana, indicadores de ausencia, escalado y one-hot se ajustan dentro de cada fold. Las categorías desconocidas se admiten.
- Candidatos: logística balanceada, SVM RBF, Random Forest, bagging de árboles y boosting `HistGradientBoostingClassifier`. Se usa el boosting de scikit-learn en lugar de hacer una instalación dinámica de XGBoost. La opción de búsqueda compara dos configuraciones predeterminadas por familia; no es una búsqueda exhaustiva.
- Stacking combina logística y Random Forest; su metamodelo aprende sólo de predicciones temporales fuera de muestra con embargo. Con este catálogo se puede ajustar al bloque completo, pero su CV interna es insuficiente: aparece como no seleccionable y se explica el motivo.
- Dos referencias siempre presentes: prevalencia constante de entrenamiento y actividad en celda a 30 días, transformada monótonamente con escala calculada en entrenamiento.
- Se elige por AP media de CV, incluyendo las referencias. Un candidato con CV incompleta no participa en la selección. El umbral se ajusta con F2 **sólo en validación**. No se reajusta el modelo con validación/prueba antes de mostrar las métricas.
- AP corresponde a `average_precision_score`, no a una integración trapezoidal de PR. Se reportan ROC AUC, recall, precisión, F2, MCC, balanced accuracy, Brier cuando corresponde, lift y tasa de señales. Las métricas indefinidas quedan vacías; no se inventan ceros.
- SVM utiliza `decision_function` transformada con sigmoid exclusivamente como score, evitando la calibración interna aleatoria de `SVC(probability=True)`. No se calcula Brier para SVM ni para la referencia de actividad.
- IC exploratorios por bootstrap de bloques de duración `max(7, horizonte)` días; no reemplazan una validación prospectiva. Importancia por permutación en validación, nunca usada para volver a elegir variables mirando prueba.

Los folds sin positivos en validación no tienen AP definida. Los folds cuyo entrenamiento tiene una sola clase se omiten y se documentan. Si ningún fold permite evaluar AP, el entrenamiento se detiene con una explicación. Cambiar parámetros buscando mejorar repetidamente la prueba consume su independencia; hace falta un catálogo externo para confirmar.

## Contrato de datos nuevos

CSV UTF-8/Latin-1, separado por coma o punto y coma. Columnas mínimas:

- `time`: fecha y hora ISO 8601. Si no tiene zona se interpreta como UTC.
- `latitude`, `longitude`: grados decimales.
- `depth`: kilómetros.
- `mag`: magnitud numérica.

Columnas recomendadas: `id`, `place`, `magType`, `magSource`, `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`, `magNst`, `updated`. Si falta ID se genera uno estable a partir de los campos mínimos. Los decimales usan punto.

Fechas imposibles, coordenadas fuera de rango, magnitud fuera de [−2, 10] o profundidad fuera de [−10, 800] se rechazan con motivo. Los nulos de columnas opcionales se imputan durante el modelado. Los indicadores de calidad negativos se convierten a nulos. Los valores extremos físicamente válidos se conservan. Las columnas constantes se auditan y se mantienen en el contrato para aceptar posteriores datos con variación.

Máximo 20 MB por archivo y 50.000 filas por catálogo combinado. La app no acepta artefactos pickle/joblib subidos por usuarios.

**Validar no equivale a reentrenar:**

1. Entrenar o recuperar un experimento en Registro.
2. Cargar un lote de eventos **posteriores al último evento del catálogo de ese experimento**, sin IDs repetidos del histórico.
3. Pulsar «Evaluar con modelo congelado». El histórico se usa para calcular ventanas pasadas. La etiqueta se calcula sólo cuando el lote cubre hasta `t+h`. El último tramo queda pendiente y no entra en métricas.
4. Para resolver pendientes, cargar un lote acumulado completo desde el final del catálogo congelado, incluyendo los eventos anteriores y los posteriores que completan el horizonte. Cada validación queda asociada al experimento y a la huella del lote; el mismo lote no duplica registros.
5. «Incorporar al catálogo y guardar» conserva IDs existentes y agrega eventos nuevos. No altera el experimento congelado. Después crear otro entrenamiento para obtener una nueva versión.

El catálogo debe mantener región, cobertura y criterio de selección. No basta con cargar sólo sismos fuertes o eventos aislados para construir etiquetas negativas fiables. La cobertura se **supone continua hasta el último evento cargado**; una pausa de captura no puede inferirse como ausencia real de sismos. La app muestra rangos, faltantes y separación respecto al histórico, pero no certifica completitud ni comparabilidad de escalas de magnitud.

El CSV histórico contiene mediciones revisadas. Aunque se excluye `updated` como predictor, no hay versiones de las mediciones tal como estaban disponibles en cada instante: el embargo no resuelve ese sesgo de revisión. Para una evaluación prospectiva hacen falta snapshots de adquisición y predicciones registradas antes de conocer el futuro.

## Persistencia y operación

`runtime/sismolab.sqlite3` guarda el catálogo activo, metadatos de experimentos y reportes externos. `runtime/models/` contiene los pipelines y snapshots congelados. Cada experimento incluye UUID, SHA-256 del catálogo, parámetros, semilla, versiones de Python/scikit-learn, límites temporales y resultados. El checksum detecta corrupción accidental del artefacto antes de cargarlo; no es una firma de confianza contra un atacante con acceso al servidor.

`SISMO_DATA_DIR` cambia la ubicación persistente. El catálogo cargado en la barra lateral se utiliza en sesión hasta pulsar «Guardar catálogo local»; el catálogo incluido también puede restaurarse en sesión sin borrar el guardado. Los experimentos se guardan automáticamente al entrenar. Las incorporaciones desde Nuevos datos se guardan de inmediato. `runtime/` se excluye de Git y de la imagen.

Esta entrega funciona como una instancia interna con un catálogo compartido. Para operación empresarial multiusuario faltan autenticación/roles, aislamiento por organización, historial y aprobación de revisiones de catálogo, coordinación de escrituras entre sesiones, backups y monitoreo de deriva con etiquetas reales. SQLite y los artefactos requieren un volumen persistente; respaldar el directorio completo con la aplicación detenida o con un procedimiento consistente de backup.

## Docker

```bash
docker compose up --build -d
docker compose logs -f
```

Abrir <http://localhost:8501>. El contenedor corre como usuario sin privilegios, expone la comprobación de salud `/_stcore/health` y guarda datos en el volumen `sismolab_data`. Para detenerlo sin perder los datos: `docker compose down`.

Para un servidor remoto: ejecutar detrás de HTTPS y autenticación. No publicar un catálogo compartido de escritura anónima. El despliegue público no está realizado por esta entrega; requiere elegir servidor/cuenta.

También se incluye `requirements.txt` exportado desde `uv.lock` para plataformas basadas en pip como Streamlit Community Cloud. Seleccionar Python 3.12 y `app.py`; esas plataformas pueden tener almacenamiento efímero, por lo que no sustituyen el volumen de la instalación Docker.

## Estructura

- `app.py`: interfaz, navegación, formularios y visualizaciones.
- `sismo/data.py`: contrato CSV, limpieza, auditoría, huellas y exportación.
- `sismo/features.py`: ingeniería causal, etiqueta, particiones y folds.
- `sismo/modeling.py`: candidatos, stacking temporal, selección, métricas, incertidumbre e inferencia externa.
- `sismo/storage.py`: SQLite, artefactos locales y registro de validaciones.
- `tests/`: fronteras temporales, censura, separación, ausencia de fuga, persistencia e interacción con el runtime de Streamlit sobre el catálogo real.

Referencias de implementación: [TimeSeriesSplit y su `gap` expresado en muestras](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [AppTest de Streamlit](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest). Aquí las particiones se implementan por fecha para aplicar el embargo a eventos irregularmente espaciados.
