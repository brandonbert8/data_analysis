"""UMSA Auditor — aplicación Streamlit del cuaderno repositorioUMSA.ipynb."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from umsa.analytics import teacher_rows
from umsa.audit import (
    audit_document,
    audit_to_json,
    audit_verdict_band,
    openalex_search,
    resolve_corpus_pdf,
)
from umsa.data import (
    DATASET_DIR,
    METRIC_LABELS,
    NUMERIC_METRICS,
    PDF_DIR,
    SECTION_LABELS,
    available_pdfs,
    build_similarity_index,
    cluster_terms,
    corpus_fingerprint,
    load_benchmark,
    load_bundle,
    load_corpus,
    load_metadata,
    percentile_table,
    similar_documents,
)
from umsa.lessons import benchmark_lesson
from umsa.modeling import (
    correlation_table,
    note_validation,
    train_rigor_model,
)
from umsa.presentation import chat_page, experiment_page
from umsa.storage import AuditStore

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="UMSA Auditor · Rigor académico", page_icon="◈", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#355DA8", "#248471", "#D9A441", "#C66B72", "#8C78B4", "#508CBB"]

PAGES = ["Resumen", "Experimento", "Exploración", "Corpus", "Chat RAG", "Auditoría PDF", "Modelos", "Registro"]


def style():
    st.html(
        """<style>
        .stApp {background:#F3F6FB;}
        .block-container {padding-top:2rem;max-width:1500px;padding-bottom:3rem;}
        h1 {font-size:clamp(2rem,1.5rem + 1.5vw,3rem)!important;font-weight:760!important;letter-spacing:-.055em;}
        h2,h3 {letter-spacing:-.025em;}
        [data-testid="stSidebar"] {border-right:1px solid #DFE6F1;background:#FFF;}
        [data-testid="stMetric"] {background:white;border:1px solid #DFE6F1;border-radius:16px;padding:1.1rem;}
        [data-testid="stMetricValue"] {color:#1D3B68;font-variant-numeric:tabular-nums;}
        .eyebrow {color:#486A9A;font-size:.8125rem;font-weight:760;letter-spacing:.15em;margin-bottom:.75rem;}
        .hero {border-radius:20px;background:linear-gradient(115deg,#152C52,#24497C 65%,#24685B);
               padding:clamp(1.25rem,1rem + 2vw,2.5rem);color:#FFF;margin-bottom:1.5rem;}
        .hero h2 {color:#FFF;margin:0;font-size:clamp(1.4rem,1rem + 1vw,2rem);}
        .hero p {color:#E0E9F8;max-width:960px;margin-bottom:0;line-height:1.7;}
        .step {background:white;border:1px solid #DFE6F1;border-top:3px solid #355DA8;
               border-radius:14px;padding:1.2rem;min-height:190px;}
        .step b {color:#355DA8;font-size:.875rem;}
        .step h4 {font-size:1.125rem;line-height:1.4;margin:.6rem 0;}
        .step p {color:#536680;line-height:1.6;}
        .verdict {border-radius:14px;padding:1rem 1.125rem;background:white;border:1px solid #DFE6F1;}
        [data-testid="stPlotlyChart"] {border:1px solid #DFE6F1;border-radius:16px;background:#FFF;overflow:hidden;}
        [data-testid="stChatMessage"] {border:1px solid #DFE6F1;border-radius:16px;background:white;}
        button:focus-visible,a:focus-visible {outline:3px solid #355DA8!important;outline-offset:3px;}
        @media(max-width:640px) {
            .block-container {padding-left:1rem;padding-right:1rem;}
            .step {min-height:0;}
            [data-baseweb="tab-list"] {overflow-x:auto;}
        }
        </style>"""
    )


def chart(fig, height=350):
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 25, "t": 65, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif", "color": "#172B4D", "size": 14},
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": False})


@st.cache_data(show_spinner=False)
def get_corpus() -> pd.DataFrame:
    return load_corpus()


@st.cache_data(show_spinner=False)
def get_metadata() -> pd.DataFrame:
    return load_metadata()


@st.cache_data(show_spinner=False)
def get_benchmark() -> pd.DataFrame:
    return load_benchmark()


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_bundle()


@st.cache_resource(show_spinner=False)
def get_index() -> dict:
    return build_similarity_index(get_corpus())


@st.cache_data(show_spinner=False)
def get_percentiles() -> dict:
    return percentile_table(get_corpus())


@st.cache_data(show_spinner=False)
def get_cluster_terms() -> dict:
    return cluster_terms(get_corpus())


@st.cache_data(show_spinner=False)
def audit_corpus_pdf(filename: str) -> dict:
    return audit_document(
        resolve_corpus_pdf(filename, PDF_DIR), filename, get_corpus(), get_index(), get_percentiles()
    )


def get_store() -> AuditStore:
    return AuditStore(os.environ.get("UMSA_DATA_DIR", str(ROOT / "runtime")))


def hero(eyebrow: str, title: str, subtitle: str, headline: str):
    st.html(f'<div class="eyebrow">{eyebrow}</div>')
    st.title(title)
    st.html(f'<div class="hero"><h2>{headline}</h2><p>{subtitle}</p></div>')


def summary(corpus: pd.DataFrame, store: AuditStore):
    hero(
        "OBSERVATORIO / RESUMEN EJECUTIVO",
        "Del expediente a la evidencia.",
        "Explora el experimento PLN del repositorio de Informática de la UMSA: "
        "del corpus y sus gráficos a los modelos y a un chat que responde con evidencia recuperada.",
        "Un laboratorio de lenguaje y evidencia académica.",
    )
    percentiles = get_percentiles()
    bundle = get_bundle()
    notes = note_validation(corpus)
    cols = st.columns(4)
    cols[0].metric("Tesis en el corpus", f"{len(corpus)}")
    cols[1].metric("Mediana de rigor", f"{corpus['indice_rigor_academico'].median():.2f} / 100")
    cols[2].metric("Mediana de citas", f"{percentiles['total_citas_detectadas']['p50']:.0f}")
    cols[3].metric("Anomalías estructurales", int(corpus["anomalia"].sum()))
    st.caption(
        f"Corpus: {corpus['fecha'].min()} → {corpus['fecha'].max()} · "
        f"{corpus['autores'].nunique()} autores · {corpus['tutores'].nunique()} tutores · "
        f"huella {corpus_fingerprint(corpus)[:16]}"
    )

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Distribución del índice de rigor")
        fig = px.histogram(
            corpus,
            x="indice_rigor_academico",
            nbins=22,
            labels={"indice_rigor_academico": "Índice de rigor académico"},
            title="70 tesis · umbral empírico de la mediana",
        )
        fig.add_vline(x=60, line_dash="dash", line_color="#A24E5A", annotation_text="60 · aceptable")
        fig.add_vline(x=72, line_dash="dash", line_color="#4E8B62", annotation_text="72 · sólido")
        fig.add_vline(x=float(corpus["indice_rigor_academico"].median()), line_dash="dot", line_color="#355DA8",
                      annotation_text="Mediana del corpus", annotation_position="bottom right")
        chart(fig, 320)
        st.caption(
            "El índice combina estructura reglamentaria (25%), coherencia objetivos↔aplicativo (25%), "
            "coherencia objetivos↔conclusiones (15%), citas (20%) y extensión (15%)."
        )
    with right:
        st.subheader("Estado del proyecto")
        st.success("Corpus y paquete de producción cargados")
        st.write(
            "**Paquete de producción:**",
            f"{len(bundle['metadata_umsa'])} fichas · embeddings {bundle['embeddings_umsa'].shape[1]}D",
        )
        st.write("**Mediana de referencia:**", f"{corpus['indice_rigor_academico'].median():.2f}")
        if notes["spearman"] is not None:
            st.metric(
                "Spearman rigor vs. nota real",
                f"{notes['spearman']:.3f}",
                help=f"n={notes['n']} tesis con nota numeral extraída del acta",
            )
        st.write(
            "**Fuente:** `df_maestro_final_modelos_70.parquet` + "
            "`auditor_umsa_produccion.joblib`."
        )
        st.write(
            "**Protocolo:** 8 secciones reglamentarias, citas APA/IEEE, coherencia léxica TF-IDF "
            "y estilometría (burstiness + MATTR)."
        )
        st.info(
            "El índice es una señal de apoyo, no una calificación oficial. Las notas del tribunal "
            "sólo están disponibles para una muestra pequeña."
        )

    st.subheader("Un proyecto, cuatro formas de explorarlo")
    cards = [
        (
            "EXPERIMENTO",
            "¿Cómo se construyó el estudio?",
            "Depuración del corpus, laboratorio PLN y gráficos originales del cuaderno.",
        ),
        (
            "01 / EXPLORAR",
            "¿Cómo se distribuye el rigor?",
            "Años, modalidades, tutores, clústeres temáticos, cumplimiento de secciones y anomalías.",
        ),
        (
            "02 / AUDITAR",
            "¿Qué le falta a un PDF nuevo?",
            "Diagnóstico explicable por sección, citas, coherencia y redacción, con tesis similares.",
        ),
        (
            "CHAT CON RAG",
            "¿Qué dice la evidencia?",
            "Consulta la metodología y las tesis con GPT-4o mini y fragmentos verificables.",
        ),
    ]
    for col, (label, title, body) in zip(st.columns(4), cards):
        with col:
            st.html(f'<div class="step"><b>{label}</b><h4>{title}</h4><p>{body}</p></div>')

    recent = store.audits()[:5]
    if recent:
        st.subheader("Últimas auditorías guardadas")
        st.dataframe(pd.DataFrame(recent), hide_index=True, width="stretch")


def exploration(corpus: pd.DataFrame):
    st.html('<div class="eyebrow">01 / EXPLORACIÓN</div>')
    st.title("Conocer el corpus antes de auditar")
    left, middle, right = st.columns([2, 2, 2])
    years = left.multiselect(
        "Año de publicación", sorted(corpus["fecha"].dropna().unique().tolist()), default=[]
    )
    modalities = middle.multiselect(
        "Modalidad", sorted(corpus["modalidad"].unique().tolist()), default=[]
    )
    clusters = right.multiselect(
        "Clúster temático", sorted(corpus["cluster_tematico"].dropna().unique().tolist()), default=[]
    )
    subset = corpus.copy()
    if years:
        subset = subset[subset["fecha"].isin(years)]
    if modalities:
        subset = subset[subset["modalidad"].isin(modalities)]
    if clusters:
        subset = subset[subset["cluster_tematico"].isin(clusters)]
    st.caption(f"{len(subset)} tesis en la vista. Los filtros sólo afectan la exploración.")
    if subset.empty:
        st.info("No hay tesis para estos filtros.")
        return

    cols = st.columns(4)
    cols[0].metric("Tesis", len(subset))
    cols[1].metric("Rigor medio", f"{subset['indice_rigor_academico'].mean():.2f}")
    cols[2].metric("Citas medianas", f"{subset['total_citas_detectadas'].median():.0f}")
    cols[3].metric("Páginas medianas", f"{subset['num_paginas'].median():.0f}")

    tab_dist, tab_topics, tab_structure = st.tabs(
        ["Distribución y correlaciones", "Tutores y temas", "Cumplimiento estructural"]
    )
    with tab_dist:
        left, right = st.columns(2)
        with left:
            chart(
                px.box(
                    subset,
                    x="modalidad",
                    y="indice_rigor_academico",
                    color="modalidad",
                    points="all",
                    labels={"modalidad": "Modalidad", "indice_rigor_academico": "Índice de rigor"},
                    title="Rigor por modalidad",
                )
            )
            chart(
                px.scatter(
                    subset,
                    x="total_citas_detectadas",
                    y="indice_rigor_academico",
                    color=subset["cluster_tematico"].astype(str),
                    size="num_paginas",
                    hover_data=["titulo", "autores", "fecha"],
                    labels={
                        "total_citas_detectadas": "Citas detectadas",
                        "indice_rigor_academico": "Índice de rigor",
                        "color": "Clúster",
                    },
                    title="Citas vs. rigor",
                )
            )
        with right:
            chart(
                px.bar(
                    subset.groupby("fecha").size().rename("Tesis").reset_index(),
                    x="fecha",
                    y="Tesis",
                    text="Tesis",
                    labels={"fecha": "Año"},
                    title="Tesis por año",
                ),
                300,
            )
            corr_cols = [
                "num_paginas",
                "total_citas_detectadas",
                "coh_obj_apl_v2",
                "coh_obj_con_v2",
                "burstiness_oraciones",
                "riqueza_lexica_mattr",
                "indice_rigor_academico",
            ]
            short_labels = dict(zip(corr_cols, ["Páginas", "Citas", "Coh. aplicativo", "Coh. conclusiones",
                                                 "Burstiness", "MATTR", "Rigor"]))
            correlation = subset[corr_cols].corr().rename(index=short_labels, columns=short_labels)
            chart(
                px.imshow(
                    correlation,
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlación de Pearson entre métricas",
                ),
                470,
            )
            st.caption("Pearson utiliza los pares disponibles de cada variable. Una correlación no demuestra "
                       "causalidad; el índice contiene varias de estas métricas en su propia fórmula. "
                       "Las celdas vacías indican datos insuficientes o variables sin variación.")
        st.download_button(
            "Descargar vista explorada",
            subset.to_csv(index=False).encode("utf-8"),
            "exploracion_umsa.csv",
            "text/csv",
        )
    with tab_topics:
        left, right = st.columns(2)
        with left:
            teachers = teacher_rows(subset)
            top_tutors = (teachers.loc[teachers["rol"].isin(["tutor", "sin especificar"])]
                          .groupby("docente")["archivo_pdf"].nunique().sort_values(ascending=False).head(12))
            chart(
                px.bar(
                    x=top_tutors.values,
                    y=top_tutors.index,
                    orientation="h",
                    labels={"x": "Tesis dirigidas", "y": "Tutor"},
                    title="Tutores con más tesis en la vista",
                ),
                420,
            )
            st.caption("Docentes separados por persona, con nombres normalizados. "
                       "Se cuentan PDFs distintos por tutor; los asesores se excluyen de este gráfico.")
        with right:
            cluster_counts = subset["cluster_tematico"].value_counts().sort_index()
            chart(
                px.bar(
                    x=cluster_counts.index.astype(str),
                    y=cluster_counts.values,
                    labels={"x": "Clúster", "y": "Tesis"},
                    title="Tamaño de los clústeres temáticos",
                ),
                240,
            )
            terms = get_cluster_terms()
            if terms:
                st.write("**Términos dominantes por clúster**")
                st.dataframe(
                    pd.DataFrame(
                        [{"clúster": key, "términos": ", ".join(value)} for key, value in terms.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
        st.write("**Rigor medio por clúster**")
        cluster_summary = (
            subset.groupby("cluster_tematico", dropna=False)["indice_rigor_academico"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .rename(columns={"count": "tesis", "mean": "media", "median": "mediana", "std": "desv."})
        )
        st.dataframe(cluster_summary, hide_index=True, width="stretch")
    with tab_structure:
        compliance = (subset[list(SECTION_LABELS)].mean() * 100).rename(
            {key: label for key, label in SECTION_LABELS.items()}
        )
        chart(
            px.bar(
                x=compliance.values,
                y=compliance.index,
                orientation="h",
                labels={"x": "% de tesis que cumplen", "y": ""},
                title="Cumplimiento de secciones reglamentarias",
                range_x=[0, 100],
            ),
            420,
        )
        st.dataframe(
            subset[
                ["archivo_pdf", "titulo", "autores", "fecha", "modalidad", "indice_rigor_academico", "anomalia"]
            ].sort_values("indice_rigor_academico", ascending=False),
            hide_index=True,
            width="stretch",
        )


def corpus_page(corpus: pd.DataFrame):
    st.html('<div class="eyebrow">02 / CORPUS</div>')
    st.title("Cada tesis, con su expediente")
    search = st.text_input("Buscar por título, autor o tutor", "")
    view = corpus.copy()
    if search.strip():
        needle = search.strip().lower()
        mask = (
            view["titulo"].str.lower().str.contains(needle, na=False, regex=False)
            | view["autores"].str.lower().str.contains(needle, na=False, regex=False)
            | view["tutores"].str.lower().str.contains(needle, na=False, regex=False)
        )
        view = view[mask]
    only_anomalies = st.checkbox("Mostrar sólo anomalías estructurales", value=False)
    if only_anomalies:
        view = view[view["anomalia"]]
    st.caption(f"{len(view)} tesis coinciden.")
    st.dataframe(
        view[
            [
                "archivo_pdf",
                "titulo",
                "autores",
                "fecha",
                "tutores",
                "modalidad",
                "cluster_tematico",
                "indice_rigor_academico",
                "nota_numeral",
                "anomalia",
            ]
        ].sort_values("indice_rigor_academico", ascending=False),
        hide_index=True,
        width="stretch",
    )
    if view.empty:
        return
    options = {
        f"{row.archivo_pdf} · {row.titulo[:78]}": row.archivo_pdf for row in view.itertuples()
    }
    selected = st.selectbox("Abrir ficha de tesis", list(options))
    filename = options[selected]
    row = corpus.loc[corpus["archivo_pdf"] == filename].iloc[0]
    st.divider()
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader(row["titulo"])
        st.write(f"**Autor:** {row['autores']} · **Año:** {row['fecha']} · **Tutor:** {row['tutores']}")
        st.write(f"**Mención:** {row['mencion']} · **Modalidad:** {row['modalidad']} · **Clúster:** {row['cluster_tematico']}")
        metadata = get_metadata()
        meta_row = metadata.loc[metadata["archivo_pdf"] == filename]
        if not meta_row.empty:
            meta = meta_row.iloc[0]
            if pd.notna(meta.get("materias")):
                st.write(f"**Materias:** {meta['materias']}")
            if pd.notna(meta.get("resumen")):
                with st.expander("Resumen del repositorio"):
                    st.write(str(meta["resumen"])[:2500])
        if pd.notna(row["nota_numeral"]):
            st.success(f"Nota numeral del acta: {row['nota_numeral']:.0f}")
        st.metric("Índice de rigor académico", f"{row['indice_rigor_academico']:.2f} / 100")
        sections = pd.DataFrame(
            [
                {"sección": SECTION_LABELS[key], "presente": bool(row[key])}
                for key in SECTION_LABELS
            ]
        )
        st.dataframe(sections, hide_index=True, width="stretch")
        with st.expander("Conclusiones extraídas del PDF"):
            st.write(str(row.get("conclusiones_texto", ""))[:4000])
        if st.button("Auditar este PDF en la pestaña Auditoría"):
            st.session_state["corpus_pdf_choice"] = filename
            st.info("Selección guardada. Abre «Auditoría PDF» y pulsa Ejecutar auditoría.")
    with right:
        radar_metrics = [metric for metric in NUMERIC_METRICS if metric in corpus]
        maxima = {metric: float(corpus[metric].quantile(0.95)) for metric in radar_metrics}
        values = [
            min(1.0, float(row[metric]) / maxima[metric]) if maxima[metric] else 0.0
            for metric in radar_metrics
        ]
        fig = go.Figure(
            go.Scatterpolar(
                r=values + values[:1],
                theta=[METRIC_LABELS.get(metric, metric) for metric in radar_metrics]
                + [METRIC_LABELS.get(radar_metrics[0], radar_metrics[0])],
                fill="toself",
                name=filename,
            )
        )
        fig.update_layout(
            title="Perfil relativo (p95 del corpus = 1)",
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=False,
        )
        chart(fig, 380)
        similar = similar_documents(
            " ".join(
                [
                    str(row.get("titulo", "")),
                    str(row.get("texto_completo", ""))[2000:9000],
                    str(row.get("conclusiones_texto", ""))[:2500],
                ]
            ),
            corpus,
            get_index(),
            top_n=4,
            exclude_pdf=filename,
        )
        st.write("**Tesis más similares**")
        st.dataframe(
            similar.assign(similitud=lambda frame: (frame["similitud"] * 100).round(1))
            .rename(columns={"similitud": "similitud_%"}),
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        "Descargar ficha (JSON)",
        json.dumps(
            {
                "archivo_pdf": filename,
                "titulo": row["titulo"],
                "autores": row["autores"],
                "fecha": int(row["fecha"]) if pd.notna(row["fecha"]) else None,
                "indice_rigor_academico": float(row["indice_rigor_academico"]),
                "secciones": {key: bool(row[key]) for key in SECTION_LABELS},
            },
            ensure_ascii=False,
            indent=2,
        ),
        f"ficha_{filename}.json",
        "application/json",
    )


def render_audit(audit: dict, percentiles: dict):
    score = audit["puntaje"]
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": " / 100"},
                title={"text": "Puntaje de rigor"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#155B66"},
                    "steps": [
                        {"range": [0, 60], "color": "#F3D6D9"},
                        {"range": [60, 72], "color": "#F7E7C6"},
                        {"range": [72, 100], "color": "#D6EBDD"},
                    ],
                    "threshold": {"line": {"color": "#102A43", "width": 3}, "value": score},
                },
            )
        )
        chart(fig, 300)
        st.html(
            f'<div class="verdict"><b>{audit["veredicto"]}</b><br>'
            f'<span style="color:#63758B">Banda: {audit_verdict_band(score)} · '
            f'{audit["paginas"]} páginas · {audit["total_citas"]} citas</span></div>'
        )
    with right:
        cols = st.columns(4)
        cols[0].metric("Páginas", audit["paginas"])
        cols[1].metric("Citas", audit["total_citas"])
        cols[2].metric("Estructura", f"{audit['pct_estructura'] * 100:.0f}%")
        coherence = audit["coh_aplicativo"]
        cols[3].metric("Coh. aplicativo", "s/d" if np.isnan(coherence) else f"{coherence:.2f}")
        cols = st.columns(4)
        conclusion = audit["coh_conclusiones"]
        cols[0].metric("Coh. conclusiones", "s/d" if np.isnan(conclusion) else f"{conclusion:.2f}")
        cols[1].metric("Burstiness", "s/d" if np.isnan(audit["burstiness"]) else f"{audit['burstiness']:.2f}")
        cols[2].metric("MATTR", "s/d" if np.isnan(audit["riqueza_lexica"]) else f"{audit['riqueza_lexica']:.3f}")
        cols[3].metric("Citas APA / IEEE", f"{audit['citas_apa']} / {audit['citas_ieee']}")
        st.write("**Diagnóstico explicable**")
        if audit["diagnosticos"]:
            for message in audit["diagnosticos"]:
                st.warning(message)
        else:
            st.success("Cumple los umbrales estructurales, bibliográficos y de coherencia del corpus.")

    st.subheader("Secciones reglamentarias")
    sections = pd.DataFrame(
        [
            {"sección": SECTION_LABELS[key], "presente": present}
            for key, present in audit["secciones"].items()
        ]
    )
    left, right = st.columns([1, 1.4])
    with left:
        st.dataframe(sections, hide_index=True, width="stretch")
    with right:
        fig = px.bar(
            sections,
            x="sección",
            y=[1 if value else 0 for value in sections["presente"]],
            color="presente",
            color_discrete_map={True: "#4E8B62", False: "#A24E5A"},
            labels={"y": "Presente", "sección": ""},
            title="Cobertura de secciones",
        )
        fig.update_yaxes(range=[0, 1.1], tickvals=[0, 1], ticktext=["Falta", "Presente"])
        chart(fig, 300)

    st.subheader("Comparación con el corpus")
    st.caption(
        "La coherencia se calcula con TF-IDF y se calibra por cuantiles a la escala de embeddings del "
        "cuaderno. Así el puntaje es comparable con los percentiles históricos sin instalar torch."
    )
    left, right = st.columns(2)
    with left:
        metrics = [
            ("num_paginas", audit["paginas"]),
            ("total_citas_detectadas", audit["total_citas"]),
            ("coh_obj_apl_v2", audit["coh_aplicativo"]),
            ("coh_obj_con_v2", audit["coh_conclusiones"]),
            ("burstiness_oraciones", audit["burstiness"]),
            ("riqueza_lexica_mattr", audit["riqueza_lexica"]),
        ]
        rows = []
        for key, value in metrics:
            if key not in percentiles or value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            reference = percentiles[key]
            rows.append(
                {
                    "métrica": METRIC_LABELS.get(key, key),
                    "documento": round(float(value), 3),
                    "p25": round(reference["p25"], 3),
                    "mediana": round(reference["p50"], 3),
                    "p75": round(reference["p75"], 3),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Coherencia léxica sin calibrar: aplicativo "
            f"{audit['coh_aplicativo_lexico']:.3f} · conclusiones {audit['coh_conclusiones_lexico']:.3f}."
        )
    with right:
        if not np.isnan(audit["coh_aplicativo"]) or not np.isnan(audit["coh_conclusiones"]):
            labels = ["Objetivos ↔ aplicativo", "Objetivos ↔ conclusiones"]
            values = [
                0.0 if np.isnan(audit["coh_aplicativo"]) else audit["coh_aplicativo"],
                0.0 if np.isnan(audit["coh_conclusiones"]) else audit["coh_conclusiones"],
            ]
            medians = [
                percentiles["coh_obj_apl_v2"]["p50"],
                percentiles["coh_obj_con_v2"]["p50"],
            ]
            fig = go.Figure(
                data=[
                    go.Bar(name="Documento", x=labels, y=values, marker_color="#155B66"),
                    go.Bar(name="Mediana UMSA", x=labels, y=medians, marker_color="#C48B2E"),
                ]
            )
            fig.update_layout(barmode="group", title="Coherencia léxica frente a la referencia")
            chart(fig, 320)
        else:
            st.info("No se pudieron extraer secciones suficientes para medir coherencia.")

    st.subheader("Tesis similares en el repositorio")
    if audit["similares"]:
        similar = pd.DataFrame(audit["similares"])
        similar["similitud"] = (similar["similitud"] * 100).round(1)
        st.dataframe(
            similar.rename(
                columns={
                    "archivo_pdf": "archivo",
                    "similitud": "similitud_%",
                    "indice_rigor_academico": "rigor",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Sin coincidencias internas para este documento.")

    with st.expander("Fragmentos extraídos para la auditoría"):
        for key, label in (
            ("fragmento_objetivos", "Objetivos / planteamiento"),
            ("fragmento_aplicativo", "Marco aplicativo"),
            ("fragmento_conclusiones", "Conclusiones"),
        ):
            st.write(f"**{label}**")
            st.caption(audit.get(key, "") or "No se detectó la sección.")

    st.subheader("Contraste con literatura abierta")
    st.caption("La consulta usa OpenAlex y requiere conexión a internet. Es opcional y no altera el puntaje.")
    if st.button("Buscar referentes en OpenAlex"):
        with st.spinner("Consultando OpenAlex…"):
            query, papers = openalex_search(audit.get("_texto", ""))
        st.session_state["openalex"] = {"query": query, "papers": papers, "archivo": audit["archivo"]}
    result = st.session_state.get("openalex")
    if result and result.get("archivo") == audit["archivo"]:
        st.write(f"**Consulta:** {result['query']}")
        if result["papers"]:
            st.dataframe(pd.DataFrame(result["papers"]), hide_index=True, width="stretch")
        else:
            st.info("Sin resultados externos o sin conexión.")

    st.download_button(
        "Descargar informe (JSON)",
        json.dumps(audit_to_json(audit), ensure_ascii=False, indent=2, default=str),
        f"auditoria_{Path(audit['archivo']).stem}.json",
        "application/json",
    )


def audit_page(corpus: pd.DataFrame, store: AuditStore):
    st.html('<div class="eyebrow">03 / AUDITORÍA DE PDF</div>')
    st.title("¿Qué le falta a este documento?")
    st.write(
        "Sube un PDF nuevo o elige uno del repositorio. El informe verifica secciones reglamentarias, "
        "citas, coherencia objetivos↔aplicativo↔conclusiones, estilometría y tesis similares."
    )
    tab_upload, tab_corpus = st.tabs(["Subir PDF", "Elegir del corpus"])
    source = None
    filename = None
    cache_key = None
    with tab_upload:
        uploaded = st.file_uploader("Documento PDF", type="pdf", key="pdf_upload")
        if uploaded is not None:
            source = uploaded.getvalue()
            filename = uploaded.name
            cache_key = f"upload::{filename}::{len(source)}"
    with tab_corpus:
        pdfs = available_pdfs()
        if pdfs:
            default = st.session_state.get("corpus_pdf_choice")
            index = pdfs.index(default) if default in pdfs else 0
            choice = st.selectbox("PDF del corpus", pdfs, index=index)
            if st.button("Usar este PDF"):
                st.session_state["corpus_pdf_choice"] = choice
                st.session_state.pop("audit_result", None)
            if st.session_state.get("corpus_pdf_choice"):
                source = "corpus"
                filename = st.session_state["corpus_pdf_choice"]
                cache_key = f"corpus::{filename}"
        else:
            st.info("La carpeta dataset/PDFs no está disponible.")

    if source is None:
        st.info("Selecciona un PDF para comenzar.")
        return

    st.caption(f"Documento listo: **{filename}**")
    run = st.button("Ejecutar auditoría", type="primary")
    if run:
        try:
            with st.spinner("Extrayendo texto y calculando métricas…"):
                if source == "corpus":
                    result = audit_corpus_pdf(filename)
                else:
                    result = audit_document(source, filename, corpus, get_index(), get_percentiles())
            store.save_audit(audit_to_json(result))
            st.session_state["audit_result"] = result
            st.session_state["audit_key"] = cache_key
            st.success("Auditoría calculada y guardada en el registro local.")
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            st.error(f"No se pudo auditar el PDF: {exc}")
    result = st.session_state.get("audit_result")
    if result and st.session_state.get("audit_key") == cache_key:
        st.divider()
        render_audit(result, get_percentiles())


def models_page(corpus: pd.DataFrame):
    st.html('<div class="eyebrow">04 / MODELOS Y EVIDENCIA</div>')
    st.title("Medir antes de decidir")
    benchmark_lesson(get_benchmark(), chart)

    st.divider()
    st.subheader("Modelo sustituto del índice de rigor")
    st.write(
        "Para que la app no dependa de torch ni de sentence-transformers, se entrena una regresión "
        "logística sobre las siete métricas auditables. Es un **modelo sustituto**: reconstruye el índice "
        "para mostrar la dirección de los pesos, no una validación predictiva independiente."
    )
    try:
        model = train_rigor_model(corpus)
    except ValueError as exc:
        st.warning(str(exc))
        return
    cols = st.columns(3)
    cols[0].metric("ROC-AUC (CV)", f"{model.auc:.3f}")
    cols[1].metric("Exactitud (CV)", f"{model.accuracy:.3f}")
    cols[2].metric("Umbral de alto rigor", f"{model.umbral:.2f}")
    st.warning(
        "El AUC es alto por construcción: la etiqueta se deriva del mismo índice que usa estas variables. "
        "No debe interpretarse como capacidad de predecir la nota del tribunal."
    )
    left, right = st.columns(2)
    with left:
        coefficients = model.coefficients.copy()
        chart(
            px.bar(
                coefficients.sort_values("coeficiente"),
                x="coeficiente",
                y="etiqueta",
                orientation="h",
                labels={"coeficiente": "Coeficiente estandarizado", "etiqueta": ""},
                title="Peso de cada métrica en la probabilidad de alto rigor",
            ),
            340,
        )
    with right:
        chart(
            px.histogram(
                model.predictions,
                x="probabilidad_alto_rigor",
                color=model.predictions["rigor_alto"].astype(str),
                nbins=18,
                labels={"probabilidad_alto_rigor": "Probabilidad estimada", "color": "Alto rigor real"},
                title="Probabilidades fuera de muestra (CV)",
            ),
            340,
        )
    with st.expander("Predicciones y auditoría del modelo"):
        st.dataframe(
            model.predictions.sort_values("probabilidad_alto_rigor", ascending=False),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(model.coefficients, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Validación contra notas reales del tribunal")
    validation = note_validation(corpus)
    if validation["spearman"] is None:
        st.info("No hay suficientes notas numerales para calcular correlaciones.")
    else:
        cols = st.columns(4)
        cols[0].metric("Tesis con nota", validation["n"])
        cols[1].metric("Spearman", f"{validation['spearman']:.3f}")
        cols[2].metric("p-value", f"{validation['p_value']:.3f}")
        cols[3].metric("Pearson", f"{validation['pearson']:.3f}")
        notes = corpus[corpus["nota_numeral"].notna()]
        chart(
            px.scatter(
                notes,
                x="indice_rigor_academico",
                y="nota_numeral",
                hover_data=["archivo_pdf", "titulo"],
                labels={
                    "indice_rigor_academico": "Índice de rigor",
                    "nota_numeral": "Nota numeral del acta",
                },
                title="Índice de rigor vs. nota real",
            ),
            340,
        )
        st.caption(
            "La muestra es pequeña: la correlación es exploratoria y no demuestra validez predictiva. "
            "Las notas provienen de las actas cuando el texto permitió extraerlas."
        )

    st.subheader("Correlaciones y anomalías")
    st.dataframe(correlation_table(corpus), hide_index=True, width="stretch")
    anomalies = corpus[corpus["anomalia"]][
        ["archivo_pdf", "titulo", "indice_rigor_academico", "score_anomalia_if"]
    ].sort_values("score_anomalia_if")
    st.write("**Documentos marcados por Isolation Forest**")
    st.dataframe(anomalies, hide_index=True, width="stretch")
    st.caption(
        "Isolation Forest se entrenó en el cuaderno con estructura, coherencia, citas, páginas y burstiness. "
        "Una anomalía no implica fraude ni mala calidad: sólo un perfil atípico frente al corpus."
    )


def registry_page(store: AuditStore):
    st.html('<div class="eyebrow">05 / REGISTRO</div>')
    st.title("Cada auditoría tiene una versión")
    records = store.audits()
    if not records:
        st.info("Todavía no hay auditorías guardadas. Ejecuta una desde «Auditoría PDF».")
    else:
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        choice = st.selectbox(
            "Revisar auditoría",
            [record["id"] for record in records],
            format_func=lambda ident: next(
                f"{record['created_at'][:19]} · {record['filename']} · {record['score']:.2f}"
                for record in records
                if record["id"] == ident
            ),
        )
        payload = store.audit(choice)
        cols = st.columns(4)
        cols[0].metric("Puntaje", f"{payload.get('puntaje', 0):.2f}")
        cols[1].metric("Páginas", payload.get("paginas", 0))
        cols[2].metric("Citas", payload.get("total_citas", 0))
        cols[3].metric("Estructura", f"{float(payload.get('pct_estructura', 0)) * 100:.0f}%")
        st.write(f"**Veredicto:** {payload.get('veredicto', '')}")
        st.json(payload, expanded=False)
        left, right = st.columns(2)
        left.download_button(
            "Descargar auditoría (JSON)",
            json.dumps(payload, ensure_ascii=False, indent=2),
            f"auditoria_{Path(payload.get('archivo', 'documento')).stem}.json",
            "application/json",
        )
        if right.button("Eliminar auditoría del registro"):
            store.delete_audit(choice)
            st.rerun()

    st.subheader("Procedencia y despliegue")
    st.write(
        "El corpus, el paquete de producción y los PDFs se leen desde `dataset/`. Las auditorías se "
        "persisten en SQLite dentro de `runtime/`. Define `UMSA_DATA_DIR` para usar un volumen distinto."
    )
    st.code("uv sync --frozen\nuv run streamlit run app.py", language="bash")
    st.caption(
        f"Directorio de datos: {DATASET_DIR} · "
        f"almacén: {store.database} · "
        f"auditorías: {len(records)}"
    )


def main():
    style()
    corpus = get_corpus()
    store = get_store()
    with st.sidebar:
        st.html(
            '<div class="eyebrow">INFORMÁTICA · UMSA</div>'
            '<h2 style="color:#1D3B68;letter-spacing:-1px">◈ UMSA Lab</h2>'
            '<p style="color:#536680">Observatorio de lenguaje y rigor académico</p>'
        )
        page = st.radio("Recorrido analítico", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("CORPUS DE REFERENCIA")
        st.metric("Tesis", len(corpus))
        st.metric("Rigor mediano", f"{corpus['indice_rigor_academico'].median():.2f}")
        st.metric("Citas medianas", f"{corpus['total_citas_detectadas'].median():.0f}")
        with st.expander("Archivos utilizados"):
            st.write(
                "- `dataset/df_maestro_final_modelos_70.parquet`\n"
                "- `dataset/auditor_umsa_produccion.joblib`\n"
                "- `dataset/benchmark_ml_vs_dl_umsa.csv`\n"
                "- `dataset/PDFs/`"
            )
            st.caption("Generados por repositorioUMSA.ipynb.")
        st.divider()
        st.caption("Auditoría explicable · Sin decisiones automáticas de aprobación")

    if page == "Resumen":
        summary(corpus, store)
    elif page == "Experimento":
        experiment_page(corpus, get_metadata(), chart)
    elif page == "Chat RAG":
        chat_page(corpus)
    elif page == "Exploración":
        exploration(corpus)
    elif page == "Corpus":
        corpus_page(corpus)
    elif page == "Auditoría PDF":
        audit_page(corpus, store)
    elif page == "Modelos":
        models_page(corpus)
    elif page == "Registro":
        registry_page(store)
    st.divider()
    st.caption(
        "UMSA Auditor · Migración del cuaderno repositorioUMSA.ipynb · "
        "El índice de rigor es una señal de apoyo, no una calificación oficial."
    )


if __name__ == "__main__":
    main()
