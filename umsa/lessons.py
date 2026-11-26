"""Recorrido visual: pregunta, operación del código, resultado y evidencia."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from umsa.data import DATASET_DIR, SECTION_LABELS, load_benchmark
from umsa.figures import (
    COMPONENTS,
    WEIGHTS,
    benchmark_figure,
    benchmark_values,
    coverage_inventory,
    notebook_clusters,
    rigor_components,
    sentence_profile,
    term_contributions,
    tfidf_experiment,
    tutor_counts,
)
from umsa.modeling import note_validation
from umsa.notebook import NOTEBOOK_PATH, notebook_cells

STAGES = ["Datos y depuración", "Tutores y metadatos", "TF-IDF y similitud", "Coherencia y citación",
          "Clústeres temáticos", "Rigor y estilometría", "Modelos y validación", "Cuaderno y cobertura"]


@st.cache_data(show_spinner=False)
def cells():
    return notebook_cells()


@st.cache_resource(show_spinner="Vectorizando las conclusiones…")
def get_tfidf(stage, min_df):
    return tfidf_experiment(stage, min_df)


def explanation(calculation: str, reading: str):
    st.markdown(f"**Qué hace el código.** {calculation}")
    st.markdown(f"**Cómo leer el resultado.** {reading}")


def evidence(key: str, cell_numbers: list[int], data: pd.DataFrame, note: str = ""):
    st.caption("Fuente: repositorioUMSA.ipynb · " + ", ".join(f"celda {i}" for i in cell_numbers) + ". " + note)
    with st.expander("Revisar código original y datos de este gráfico"):
        code_tab, data_tab = st.tabs(["Código original", "Datos del gráfico"])
        with code_tab:
            selected = st.selectbox("Celda de referencia", cell_numbers, key=f"code_{key}")
            cell = next((c for c in cells() if c["cell"] == selected), None)
            if cell:
                st.code(cell["source"], language="python")
                if cell["output"]:
                    st.text(cell["output"])
        with data_tab:
            st.table(data.head(40))
            if len(data) > 40:
                st.caption(f"Primeras 40 de {len(data)} filas. La descarga contiene todos los datos del gráfico.")
            st.download_button("Descargar datos del gráfico", data.to_csv(index=False).encode("utf-8"),
                               f"datos_{key}.csv", "text/csv", key=f"download_{key}")


def preparation(corpus, metadata, chart):
    st.header("¿Qué cambia al corregir la extracción de secciones?")
    first = pd.read_csv(DATASET_DIR / "dataset_textos_extraidos.csv")
    corrected = pd.read_csv(DATASET_DIR / "dataset_segmentado_v3.csv")
    rows = [{"Etapa": label, "Sección": section, "Detectados": int(frame[column].sum()), "Total": len(frame)}
            for label, frame in (("Extracción inicial", first), ("Segmentación v3", corrected))
            for section, column in (("Introducción", "tiene_introduccion"), ("Conclusiones", "tiene_conclusiones"))]
    counts = pd.DataFrame(rows)
    counts["Etiqueta"] = counts["Detectados"].astype(str) + " / " + counts["Total"].astype(str)
    chart(px.bar(counts, x="Sección", y="Detectados", color="Etapa", barmode="group", text="Etiqueta",
                 range_y=[0, len(metadata) + 8], title="La misma colección, distintas reglas de extracción"), 340)
    explanation("Las expresiones regulares detectan encabezados; al revisar los índices y la posición del texto, "
                "el número de conclusiones detectadas pasa de 14 a 73.",
                "La mejora refleja la segmentación del texto; no que las tesis hayan agregado conclusiones.")
    evidence("segmentation", [29, 31, 35], counts, "Recalculado desde los CSV de cada etapa.")
    stages = pd.DataFrame({"Etapa": ["Metadatos", "Conclusiones para similitud", "Extracción posicional", "Corpus único"],
                           "Documentos": [len(metadata), 73, 71, len(corpus)]})
    chart(go.Figure(go.Funnel(y=stages["Etapa"], x=stages["Documentos"], textinfo="value+percent initial")), 300)
    st.caption("79 → 73 → 71 → 70: cortes de distintas etapas. 71 es la salida histórica de la celda 35; "
               "el CSV definitivo ya contiene las 70 tesis tras la depuración del duplicado.")
    evidence("pipeline", [33, 35, 40], stages)
    st.header("¿Qué secciones detecta la auditoría en las tesis finales?")
    compliance = pd.DataFrame({"Sección": list(SECTION_LABELS.values()),
                               "Presentes": [int(corpus[c].sum()) for c in SECTION_LABELS]})
    compliance["Porcentaje"] = compliance["Presentes"] / len(corpus) * 100
    compliance["Etiqueta"] = compliance.apply(lambda r: f"{r['Presentes']} / {len(corpus)} · {r['Porcentaje']:.1f}%", axis=1)
    fig = px.bar(compliance.sort_values("Porcentaje"), x="Porcentaje", y="Sección", orientation="h",
                 text="Etiqueta", range_x=[0, 115], title="Cobertura reglamentaria · detección automática")
    fig.update_traces(textposition="outside")
    chart(fig, 420)
    explanation("Cada columna sec_* es una detección booleana. Su promedio se transforma en porcentaje.",
                "Presencia no implica calidad de la sección; una expresión regular puede producir falsos positivos.")
    evidence("compliance", [41], compliance)
    missing = corpus.loc[~corpus[list(SECTION_LABELS)].all(axis=1)]
    if not missing.empty:
        st.markdown("**Dónde se concentran las ausencias**")
        values = missing[list(SECTION_LABELS)].astype(int).to_numpy()
        fig = go.Figure(go.Heatmap(z=values, x=list(SECTION_LABELS.values()), y=missing["archivo_pdf"],
                                  zmin=0, zmax=1, colorscale=[[0, "#F0B86E"], [1, "#355DA8"]],
                                  text=np.where(values, "Presente", "Falta"), texttemplate="%{text}",
                                  showscale=False, hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
        chart(fig, max(350, len(missing) * 30))
        evidence("missing_sections", [41], missing[["archivo_pdf", *SECTION_LABELS]].rename(columns=SECTION_LABELS))
    st.header("¿Qué modalidades y menciones se extrajeron de las portadas?")
    categories = []
    for col, field in zip(st.columns(2), ("modalidad", "mencion")):
        counts = corpus[field].value_counts().rename_axis("Categoría").reset_index(name="Tesis")
        with col:
            chart(px.bar(counts, x="Tesis", y="Categoría", orientation="h", text="Tesis",
                         title="Modalidad" if field == "modalidad" else "Mención"), 300)
        categories.append(counts.assign(Variable=field))
    explanation("La auditoría busca expresiones de mención y modalidad en las primeras seis páginas de cada PDF.",
                "Son categorías detectadas en el documento; una etiqueta no especificada indica una extracción sin coincidencia.")
    evidence("categories", [41], pd.concat(categories, ignore_index=True))


def metadata_lesson(corpus, metadata, chart):
    st.header("¿Cómo cambia el ranking al limpiar los nombres?")
    mode = st.radio("Criterio de conteo", ["Sin limpieza · cuaderno", "Limpieza de sufijos · cuaderno", "Por persona · app"])
    ranking = tutor_counts(metadata, mode)
    display = ranking.head(10).iloc[::-1].copy()
    display["Etiqueta breve"] = [f"{rank}. " + (text if len(text) < 56 else text[:53] + "…")
                                 for rank, text in zip(range(len(display), 0, -1), display["Tutor / registro"])]
    fig = px.bar(display, x="Proyectos", y="Etiqueta breve", orientation="h", text="Proyectos",
                 hover_data=["Tutor / registro"], labels={"Etiqueta breve": "Tutor / registro"},
                 title=f"79 registros originales · {mode}")
    fig.update_traces(textposition="outside", cliponaxis=False)
    chart(fig, 480)
    explanation("value_counts() cuenta celdas iguales. La celda 28 limpia sufijos, pero no separa varias personas "
                "en una misma celda. La app también separa personas y unifica variantes inequívocas del nombre.",
                "Los rankings no cuentan exactamente lo mismo. En modo por persona se cuentan PDFs distintos "
                "por tutor; los asesores se excluyen. Por eso el 16 del cuaderno puede diferir del 29 de la app.")
    evidence("tutors", [24, 28], ranking, "La opción por persona es una extensión de la app.")
    left, right = st.columns(2)
    years = metadata["fecha"].value_counts().sort_index().rename_axis("Año").reset_index(name="Proyectos")
    with left:
        fig = px.bar(years, x="Año", y="Proyectos", text="Proyectos", title="Cobertura temporal del repositorio")
        fig.update_xaxes(type="category")
        chart(fig, 320)
    lengths = metadata[["archivo_pdf", "titulo"]].copy()
    lengths["Caracteres"] = metadata["resumen"].fillna("").str.len()
    lengths["Resumen ausente"] = metadata["resumen"].isna()
    with right:
        chart(px.box(lengths.loc[~lengths["Resumen ausente"]], x="Caracteres", points="all", hover_data=["archivo_pdf"],
                     title="Longitud de resúmenes disponibles"), 320)
    explanation("El conteo por fecha describe la cobertura del dataset; str.len() mide caracteres por resumen.",
                f"Un resumen largo no implica mayor rigor. Hay {int(lengths['Resumen ausente'].sum())} resumen ausente, "
                "que se muestra en los datos y se excluye del boxplot en lugar de tratarlo como texto de longitud cero.")
    evidence("years", [23], years)
    evidence("metadata", [23, 25], lengths)


def similarity_lesson(chart):
    st.header("¿Cómo pasa el código de palabras a similitud entre documentos?")
    left, right = st.columns(2)
    stage = left.selectbox("Dataset para vectorizar", ["Inicial · 73 documentos", "Final · 70 tesis"])
    min_df = right.selectbox("Mínimo de documentos que deben contener un término", [2, 1, 3, 5],
                             help="El cuaderno utiliza min_df=2. Las otras opciones son escenarios recalculados.")
    result = get_tfidf(stage, min_df)
    frame, matrix = result["frame"], result["matrix"]
    a, b, c = st.columns(3)
    a.metric("Filas · documentos", matrix.shape[0])
    b.metric("Columnas · términos", matrix.shape[1])
    c.metric("Pares distintos", len(result["pairs"]))
    explanation("TfidfVectorizer(max_features=1500, min_df=2) asigna pesos a las palabras. "
                "cosine_similarity compara los vectores normalizados: cada celda cruza dos documentos.",
                "La diagonal vale 1 por comparar cada tesis consigo misma. El análisis de pares la excluye. "
                "Cambiar min_df modifica el vocabulario y las similitudes; no se están mostrando embeddings semánticos.")
    options = frame["archivo_pdf"].tolist()
    shown = st.multiselect("Documentos visibles en el mapa · hasta 30", options, default=options[:20], max_selections=30)
    if shown:
        indexes = [options.index(name) for name in shown]
        heatmap = result["similarity"][np.ix_(indexes, indexes)]
        fig = px.imshow(heatmap, x=shown, y=shown, zmin=0, zmax=1, color_continuous_scale="Blues",
                        labels={"color": "Coseno"}, title=f"Cruces entre {len(shown)} documentos · {stage}",
                        text_auto=".2f" if len(shown) <= 10 else False)
        chart(fig, max(400, len(shown) * 22))
        evidence("similarity_matrix", [33, 36], pd.DataFrame(heatmap, index=shown, columns=shown).rename_axis("Documento").reset_index())
    threshold = st.slider("Similitud mínima de los pares listados", 0.0, 1.0, .35, .05)
    pairs = result["pairs"]
    fig = px.histogram(pairs, x="Similitud", nbins=30, range_x=[0, 1], title="Distribución de todos los pares · sin diagonal")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="Umbral elegido")
    chart(fig, 280)
    visible = pairs.loc[pairs["Similitud"].ge(threshold)]
    st.caption(f"{len(visible)} de {len(pairs)} pares superan el umbral. La selección del mapa no cambia este total.")
    st.dataframe(visible.head(20), hide_index=True, width="stretch")
    evidence("similarity_pairs", [33, 36], pairs, "Distribución y pares: visualización adicional de la app.")
    best = pairs.iloc[0]
    st.header("Desarmar un par: ¿qué palabras aportan a su similitud?")
    first = st.selectbox("Documento A", options, index=options.index(best["Documento A"]))
    second = st.selectbox("Documento B", options, index=options.index(best["Documento B"]))
    i, j = options.index(first), options.index(second)
    terms = term_contributions(result, i, j)
    st.metric("Coseno de este par", f"{result['similarity'][i, j]:.4f}")
    chart(px.bar(terms.head(12).iloc[::-1], x="Aporte al coseno", y="Término", orientation="h",
                 title="Los productos de pesos TF-IDF explican el coseno"), 380)
    explanation("Como los vectores TF-IDF tienen norma L2 igual a 1, la suma de Peso A × Peso B es el coseno.",
                "Los términos compartidos explican el parecido léxico. Una similitud alta exige revisar los textos; "
                "no demuestra plagio. El par 46514/36615 motivó la revisión del duplicado original.")
    with st.expander("Leer las conclusiones de este par"):
        for col, index in zip(st.columns(2), (i, j)):
            with col:
                st.write(frame.iloc[index]["archivo_pdf"])
                st.text(str(frame.iloc[index]["conclusiones_texto"])[:3500])
    evidence("similarity", [32, 33, 35, 36, 40], terms, "Recalculado con el CSV conservado de la etapa elegida; no se recupera el estado histórico de 71 filas.")


def coherence_lesson(corpus, chart):
    st.header("¿Qué cambia al evitar los índices y comparar párrafos?")
    comparison = st.selectbox("Relación semántica", ["Objetivos ↔ aplicativo", "Objetivos ↔ conclusiones"])
    before, after = (("coh_objetivos_aplicativo", "coh_obj_apl_v2") if comparison.endswith("aplicativo")
                     else ("coh_objetivos_conclusiones", "coh_obj_con_v2"))
    view = corpus[["archivo_pdf", "titulo", before, after]].dropna(subset=[before, after]).copy()
    view["Cambio"] = view[after] - view[before]
    fig = px.scatter(view, x=before, y=after, hover_data=["archivo_pdf", "titulo", "Cambio"],
                     labels={before: "Versión inicial", after: "Versión por párrafos"},
                     range_x=[0, 1], range_y=[0, 1], title=f"Cada punto es una tesis · {comparison}")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#6B7280"}, name="Sin cambio"))
    chart(fig, 410)
    explanation("La versión inicial codifica un fragmento por sección. La celda 44 evita el índice y promedia "
                "la mejor correspondencia de cada párrafo de objetivos con los párrafos de la otra sección.",
                f"Encima de la diagonal la nueva medición aumenta; debajo disminuye. Hay {len(view)} pares válidos "
                f"de {len(corpus)} tesis. Cambia el método de medición: un aumento no prueba mejor calidad.")
    evidence("coherence", [43, 44], view)
    citations = corpus[["archivo_pdf", "citas_apa", "citas_ieee", "total_citas_detectadas"]].copy()
    totals = pd.DataFrame({"Patrón": ["APA", "IEEE"], "Coincidencias": [citations["citas_apa"].sum(), citations["citas_ieee"].sum()]})
    left, right = st.columns(2)
    with left:
        chart(px.bar(totals, x="Patrón", y="Coincidencias", text="Coincidencias", title="Qué patrones detectó el código"), 300)
    with right:
        chart(px.histogram(citations, x="total_citas_detectadas", nbins=20,
                           labels={"total_citas_detectadas": "Coincidencias por tesis"}, title="La citación varía entre documentos"), 300)
    st.caption("Las expresiones regulares cuentan coincidencias APA/IEEE, no fuentes bibliográficas únicas ni corrección de las citas.")
    evidence("citations", [43], citations)


def clusters_lesson(corpus, chart):
    st.header("¿Por qué el código eligió K=6?")
    scores, groups = notebook_clusters()
    fig = px.line(scores, x="K", y="Silhouette", markers=True, text="Silhouette", range_y=[0, 1],
                  title="Selección de K · resultados guardados en la celda 45")
    fig.update_traces(texttemplate="%{y:.4f}", textposition="top center")
    best = scores.loc[scores["Silhouette"].idxmax()]
    fig.add_vline(x=best["K"], line_dash="dash", annotation_text=f"Mejor K={best['K']:.0f}")
    chart(fig, 330)
    explanation("TF-IDF de título duplicado + cuerpo → TruncatedSVD(15) → KMeans para K=3…7. "
                "Se guarda el K con mayor silhouette y se reajusta el modelo final.",
                f"El máximo es {best['Silhouette']:.4f}, pero está cerca de 0: los temas se solapan. "
                "K=6 fue el mejor entre los K probados, no una prueba de seis categorías académicas naturales.")
    evidence("silhouette", [45], scores, "Son resultados históricos, no un entrenamiento nuevo.")
    counts = corpus["cluster_tematico"].value_counts().sort_index().rename_axis("Clúster").reset_index(name="Tesis actuales")
    chart(px.bar(counts, x="Clúster", y="Tesis actuales", text="Tesis actuales", title="Tamaño de los grupos del corpus final"), 300)
    selected = st.selectbox("Clúster para inspeccionar", groups["Clúster"].tolist())
    group = groups.loc[groups["Clúster"].eq(selected)].iloc[0]
    st.write(f"**Términos del cuaderno:** {group['Términos']}")
    members = corpus.loc[corpus["cluster_tematico"].eq(selected), ["archivo_pdf", "titulo", "fecha", "indice_rigor_academico"]]
    st.dataframe(members, hide_index=True, width="stretch")
    explanation("El cuaderno calcula el promedio TF-IDF de cada clúster y toma sus diez términos dominantes.",
                "Las palabras orientan la interpretación del grupo; los números de clúster son etiquetas sin orden de calidad. "
                "Los miembros listados permiten contrastar la interpretación del grupo con los títulos de sus tesis.")
    evidence("clusters", [45], groups)


def rigor_lesson(corpus, chart):
    st.header("¿De dónde salen los puntos de una tesis?")
    selected = st.selectbox("Tesis para descomponer su índice", corpus["archivo_pdf"].tolist())
    row = corpus.loc[corpus["archivo_pdf"].eq(selected)].iloc[0]
    contributions = rigor_components(corpus)
    actual = contributions.loc[row.name]
    st.caption(row["titulo"])
    fig = go.Figure(go.Waterfall(x=COMPONENTS + ["Total"], y=actual.tolist() + [0],
                                measure=["relative"] * 5 + ["total"], text=[f"{x:.2f}" for x in actual] + [f"{actual.sum():.2f}"],
                                textposition="outside", connector={"line": {"color": "#64748B"}}))
    fig.update_layout(title="Cinco aportes suman el índice · puntos sobre 100")
    chart(fig, 370)
    explanation("Se promedian las ocho secciones. Coherencias ausentes se imputan con la mediana. "
                "Citas y páginas se dividen por su p95 y se recortan en 1. Después se aplican pesos 25/25/15/20/15.",
                f"La suma recalculada es {actual.sum():.2f}; el artefacto guarda {row['indice_rigor_academico']:.2f}. "
                "La escala p95 procede de las 70 tesis completas, no sólo de la tesis seleccionada.")
    points = pd.DataFrame({"Componente": COMPONENTS, "Peso máximo (puntos)": WEIGHTS * 100, "Aporte actual": actual.values})
    evidence("rigor", [46], points)
    simulated = st.slider("Escenario: citas detectadas de esta tesis", 0, max(200, int(corpus["total_citas_detectadas"].max())),
                          int(row["total_citas_detectadas"]), key=f"sim_cites_{selected}")
    scenario = actual.copy()
    scenario["Citas"] = 20 * min(1, simulated / corpus["total_citas_detectadas"].quantile(.95))
    st.metric("Índice simulado · otros componentes constantes", f"{scenario.sum():.2f}", f"{scenario.sum() - actual.sum():+.2f} puntos")
    st.caption("Simulación local: no modifica el corpus ni demuestra mejora real de calidad. "
               "Superar el p95 de citas ya no añade puntos a ese componente.")
    st.header("¿Qué calculan burstiness y MATTR?")
    lengths, windows = sentence_profile(str(row["texto_completo"]))
    left, right = st.columns(2)
    with left:
        chart(px.histogram(x=lengths, nbins=25, labels={"x": "Palabras por oración"}, title="Longitudes de las oraciones del cuerpo"), 300)
        st.caption(f"Burstiness guardado: {row['burstiness_oraciones']:.4f}; recalculado: {np.std(lengths):.4f}.")
    with right:
        chart(px.line(windows, x="Inicio de ventana", y="Diversidad léxica", range_y=[0, 1], title="MATTR · ventanas de 100 palabras"), 300)
        st.caption(f"MATTR guardado: {row['riqueza_lexica_mattr']:.4f}; media de ventanas: {windows['Diversidad léxica'].mean():.4f}.")
    explanation("El código usa caracteres 5.000–45.000 del texto, oraciones de al menos cuatro palabras "
                "y ventanas léxicas de 100 palabras con salto de 50.",
                "Burstiness es dispersión de longitud de oraciones; MATTR es diversidad de vocabulario. "
                "Ninguna métrica por sí sola identifica autoría ni calidad académica.")
    evidence("sentences", [46], pd.DataFrame({"Oración": range(1, len(lengths) + 1), "Palabras": lengths}))
    evidence("stylometry", [46], windows)
    st.header("¿Por qué hay siete anomalías?")
    anomalies = corpus[["archivo_pdf", "indice_rigor_academico", "score_anomalia_if", "anomalia"]].copy()
    anomalies["Estado"] = np.where(anomalies["anomalia"], "Atípico", "Resto del corpus")
    fig = px.scatter(anomalies, x="score_anomalia_if", y="indice_rigor_academico", color="Estado", symbol="Estado",
                     hover_data=["archivo_pdf"], title="Isolation Forest · puntuaciones guardadas")
    fig.add_vline(x=0, line_dash="dash", annotation_text="Frontera del modelo")
    chart(fig, 330)
    explanation("IsolationForest usa seis variables y contamination=0.10; decision_function() coloca "
                "la frontera en cero y fit_predict() asigna −1 a los casos atípicos.",
                f"Hay {int(corpus['anomalia'].sum())} de {len(corpus)} perfiles atípicos. La fracción del 10% "
                "fue configurada en el algoritmo: no es una tasa de fraude descubierta.")
    evidence("anomalies", [46], anomalies)


def benchmark_lesson(benchmark, chart):
    st.header("¿Qué aprendieron los siete modelos sobre la etiqueta de rigor?")
    if benchmark.empty:
        st.info("No se encontró el archivo de resultados del benchmark.")
        return
    values = benchmark_values(benchmark)
    metric = st.selectbox("Métrica del benchmark", ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"])
    chart(benchmark_figure(values, metric), 430)
    explanation("El objetivo es índice ≥ mediana (35/35 clases). Los modelos reciben embeddings y estilometría; "
                "StratifiedKFold crea cinco particiones. El escalado se ajusta al entrenamiento y SMOTE se aplica "
                "al entrenamiento ML/MLP; las redes secuenciales usan cuatro bloques de embeddings.",
                "ROC-AUC≈0,5 es poca discriminación. En Accuracy las barras son ± desviación entre folds, "
                "no intervalos de confianza. Las otras métricas sólo conservan su promedio.")
    heatmap = values.pivot(index="Modelo", columns="Métrica", values="Valor")
    chart(px.imshow(heatmap, zmin=0, zmax=1, text_auto=".3f", aspect="auto", color_continuous_scale="Blues",
                    title="Todas las métricas · misma escala 0–1"), 360)
    st.caption("No se trazan ROC, pérdidas de entrenamiento ni matrices de confusión de los siete modelos: "
               "el CSV no conserva las probabilidades, predicciones por fold ni historiales necesarios.")
    evidence("benchmark", [48], values)


def validation_lesson(corpus, chart):
    benchmark_lesson(load_benchmark(), chart)
    st.header("¿El índice coincide con la nota del tribunal?")
    stats = note_validation(corpus)
    notes = corpus.loc[corpus["nota_numeral"].notna(), ["archivo_pdf", "nota_numeral", "indice_rigor_academico"]]
    fig = px.scatter(notes, x="indice_rigor_academico", y="nota_numeral", text="archivo_pdf",
                     range_x=[0, 100], range_y=[0, 105], title=f"Sólo {len(notes)} tesis tienen nota extraída")
    fig.update_traces(textposition="top center")
    chart(fig, 400)
    explanation("spearmanr() compara el orden del índice con el orden de las notas disponibles.",
                f"Spearman={stats['spearman']:.4f}, p={stats['p_value']:.4f}, n={stats['n']}. "
                "La muestra pequeña y la falta de significación impiden concluir validez predictiva. "
                "El benchmark clasifica el índice; no predice estas notas.")
    evidence("validation", [43, 46], notes)


def originals_lesson():
    st.header("Cobertura del cuaderno y procedencia de los gráficos")
    inventory = coverage_inventory()
    st.dataframe(inventory, hide_index=True, width="stretch")
    st.write("Las cinco figuras originales se conservan aquí. Las etapas anteriores reconstruyen esas figuras "
             "e incorporan visualizaciones de resultados que el cuaderno sólo imprimía como tablas o texto.")
    available = cells()
    if not available:
        st.info("El cuaderno original no está disponible en esta instalación.")
        return
    images_only = st.toggle("Mostrar sólo celdas con gráficos", value=True)
    choices = [c for c in available if c["images"] or not images_only]
    selected = st.selectbox("Celda del cuaderno", [c["cell"] for c in choices],
                            format_func=lambda i: f"Celda {i} · {available[i - 1]['title']}")
    cell = available[selected - 1]
    for picture in cell["images"]:
        st.image(picture, caption=f"Figura guardada en la celda {selected}", width="stretch")
    st.code(cell["source"], language="python")
    if cell["output"]:
        st.text(cell["output"])
    st.download_button("Descargar cuaderno original", NOTEBOOK_PATH.read_bytes(), "repositorioUMSA.ipynb", "application/x-ipynb+json")


def experiment_dashboard(corpus, metadata, chart):
    st.title("El experimento, paso a paso")
    st.write("Cada etapa conecta una pregunta con la operación del código, su resultado y los datos que la sostienen.")
    stage = st.selectbox("Etapa del análisis", STAGES, key="experiment_stage")
    if stage == STAGES[0]:
        preparation(corpus, metadata, chart)
    elif stage == STAGES[1]:
        metadata_lesson(corpus, metadata, chart)
    elif stage == STAGES[2]:
        similarity_lesson(chart)
    elif stage == STAGES[3]:
        coherence_lesson(corpus, chart)
    elif stage == STAGES[4]:
        clusters_lesson(corpus, chart)
    elif stage == STAGES[5]:
        rigor_lesson(corpus, chart)
    elif stage == STAGES[6]:
        validation_lesson(corpus, chart)
    else:
        originals_lesson()
