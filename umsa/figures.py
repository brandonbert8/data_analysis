"""Datos reproducibles y figuras para explicar las operaciones del cuaderno."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from umsa.analytics import teacher_rows
from umsa.data import DATASET_DIR, SECTION_LABELS
from umsa.notebook import notebook_cells

COMPONENTS = ["Estructura", "Objetivos ↔ aplicativo", "Objetivos ↔ conclusiones", "Citas", "Páginas"]
WEIGHTS = np.array([.25, .25, .15, .20, .15])


def tutor_counts(metadata: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Por persona · app":
        people = teacher_rows(metadata)
        counts = (people.loc[people["rol"].isin(["tutor", "sin especificar"])]
                  .groupby("docente")["archivo_pdf"].nunique())
    else:
        values = metadata["tutores"].fillna("Sin dato")
        if mode == "Limpieza de sufijos · cuaderno":
            values = values.str.replace(r",\s*tutor|\s*tutor|\s*asesor|\s*asesora", "", regex=True, case=False).str.strip()
        counts = values.value_counts()
    return counts.rename_axis("Tutor / registro").reset_index(name="Proyectos").sort_values(
        ["Proyectos", "Tutor / registro"], ascending=[False, True]).reset_index(drop=True)


def tfidf_experiment(stage: str = "Final · 70 tesis", min_df: int = 2) -> dict:
    path = DATASET_DIR / ("dataset_nlp_limpio.csv" if stage.startswith("Inicial") else "dataset_final_definitivo.csv")
    frame = pd.read_csv(path)
    frame = frame.loc[frame["tiene_conclusiones"].eq(True)].dropna(subset=["conclusiones_limpias"]).reset_index(drop=True)
    vectorizer = TfidfVectorizer(max_features=1500, min_df=min_df)
    matrix = vectorizer.fit_transform(frame["conclusiones_limpias"])
    similarity = np.clip(cosine_similarity(matrix), 0, 1)
    a, b = np.triu_indices(len(frame), 1)
    pairs = pd.DataFrame({"Documento A": frame["archivo_pdf"].to_numpy()[a],
                          "Documento B": frame["archivo_pdf"].to_numpy()[b], "Similitud": similarity[a, b]})
    return {"frame": frame, "terms": vectorizer.get_feature_names_out(), "matrix": matrix,
            "similarity": similarity, "pairs": pairs.sort_values("Similitud", ascending=False).reset_index(drop=True)}


def term_contributions(experiment: dict, first: int, second: int) -> pd.DataFrame:
    a = experiment["matrix"].getrow(first).toarray().ravel()
    b = experiment["matrix"].getrow(second).toarray().ravel()
    return pd.DataFrame({"Término": experiment["terms"], "Peso A": a, "Peso B": b,
                         "Aporte al coseno": a * b}).sort_values("Aporte al coseno", ascending=False)


def rigor_components(corpus: pd.DataFrame) -> pd.DataFrame:
    """Reproduce celda 46, incluida imputación y recorte p95 calculados en el corpus completo."""
    citations = corpus["total_citas_detectadas"] / corpus["total_citas_detectadas"].quantile(.95)
    pages = corpus["num_paginas"] / corpus["num_paginas"].quantile(.95)
    normalized = pd.DataFrame({
        COMPONENTS[0]: corpus[list(SECTION_LABELS)].mean(axis=1),
        COMPONENTS[1]: corpus["coh_obj_apl_v2"].fillna(corpus["coh_obj_apl_v2"].median()),
        COMPONENTS[2]: corpus["coh_obj_con_v2"].fillna(corpus["coh_obj_con_v2"].median()),
        COMPONENTS[3]: citations.clip(0, 1), COMPONENTS[4]: pages.clip(0, 1),
    })
    return normalized.mul(WEIGHTS * 100, axis=1)


def sentence_profile(text: str) -> tuple[list[int], pd.DataFrame]:
    body = text[5000:45000] if len(text) > 10000 else text
    lengths = [len(sentence.split()) for sentence in re.split(r"[.!?]+", body) if len(sentence.split()) >= 4]
    words = re.findall(r"\b[a-záéíóúñ]{3,}\b", body.lower())
    windows = [(start, len(set(words[start:start + 100])) / 100)
               for start in range(0, len(words) - 100, 50)] if len(words) >= 100 else []
    return lengths, pd.DataFrame(windows, columns=["Inicio de ventana", "Diversidad léxica"])


def notebook_clusters() -> tuple[pd.DataFrame, pd.DataFrame]:
    output = next((cell["output"] for cell in notebook_cells() if "SILHOUETTE" in cell["output"].upper()), "")
    scores = pd.DataFrame([(int(k), float(score)) for k, score in re.findall(
        r"K\s*=\s*(\d+)\s*->\s*Silhouette Score\s*=\s*([\d.]+)", output)], columns=["K", "Silhouette"])
    groups = pd.DataFrame([{"Clúster": int(k), "Tesis": int(n), "Términos": words}
                           for k, n, words in re.findall(r"Clúster (\d+) \(n=(\d+) tesis\): ([^\n]+)", output)])
    return scores, groups


def benchmark_values(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in benchmark.iterrows():
        for metric in ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"):
            values = str(row[metric]).split("±")
            rows.append({"Modelo": row["Modelo"], "Métrica": metric, "Valor": float(values[0].strip()),
                         "Desviación": float(values[1].strip()) if len(values) > 1 else np.nan})
    return pd.DataFrame(rows)


def benchmark_figure(values: pd.DataFrame, metric: str) -> go.Figure:
    view = values.loc[values["Métrica"].eq(metric)].sort_values("Valor")
    fig = px.scatter(view, x="Valor", y="Modelo", text="Valor",
                     error_x="Desviación" if metric == "Accuracy" else None,
                     labels={"Valor": metric}, title=f"{metric} · los siete modelos del cuaderno")
    fig.update_traces(marker={"size": 12, "color": "#355DA8"}, texttemplate="%{x:.4f}", textposition="top center")
    fig.update_xaxes(range=[0, 1])
    if metric in ("Accuracy", "ROC-AUC"):
        fig.add_vline(x=.5, line_dash="dash", line_color="#6B7280", annotation_text="0,5 · referencia")
    return fig


def coverage_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        ("23", "Proyectos por año", "Tutores y metadatos", "Figura original + reconstrucción"),
        ("24", "Ranking de tutores sin limpieza", "Tutores y metadatos", "Figura original + reconstrucción"),
        ("25", "Longitud de resúmenes", "Tutores y metadatos", "Figura original + reconstrucción"),
        ("28", "Ranking con limpieza de sufijos", "Tutores y metadatos", "Figura original + reconstrucción"),
        ("33", "Heatmap TF-IDF · primeros 20 de 73", "TF-IDF y similitud", "Figura original + reconstrucción"),
        ("41", "Secciones, menciones y modalidades", "Datos y depuración", "Tabla del cuaderno visualizada"),
        ("43–44", "Citas y cambio de coherencia", "Coherencia y citación", "Tabla del cuaderno visualizada"),
        ("45", "Silhouette y términos por clúster", "Clústeres temáticos", "Salida textual visualizada"),
        ("46", "Rigor, estilometría y anomalías", "Rigor y estilometría", "Salida textual visualizada"),
        ("48", "Benchmark ML/DL", "Modelos y validación", "Tabla del cuaderno visualizada"),
    ], columns=["Celdas", "Análisis", "Etapa en la app", "Procedencia"])
