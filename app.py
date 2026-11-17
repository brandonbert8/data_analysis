"""SismoLab — aplicación Streamlit del pipeline sismos_integrado.ipynb."""

from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import streamlit as st

from sismo.data import REQUIRED, clean_catalog, csv_bytes, fingerprint, read_csv
from sismo.features import CATS, F19, F25, FeatureConfig, build_features, temporal_partitions
from sismo.modeling import (
    FAMILIES,
    TrainConfig,
    block_intervals,
    importance,
    metrics,
    score_new,
    train_experiment,
)
from sismo.storage import Store

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="SismoLab · Inteligencia sísmica", page_icon="◉", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0F8B8D", "#F4A259", "#5278BA", "#E36C6C", "#9B77B5"]


def style():
    st.html("""<style>
    .stApp {background: #F6F8FB;}
    .block-container {padding-top: 2rem; max-width: 1500px;}
    h1 {font-weight: 750 !important; letter-spacing: -1.5px;}
    h2,h3 {letter-spacing: -.5px;}
    [data-testid="stSidebar"] {border-right: 1px solid #DDE5EE;}
    [data-testid="stMetric"] {background:white;border:1px solid #E1E7EE;border-radius:12px;padding:18px;}
    .eyebrow {color:#0F8B8D;font-size:12px;font-weight:750;letter-spacing:2px;margin-bottom:12px;}
    .hero {border-radius:16px;background:linear-gradient(120deg,#142D48,#155B66);padding:30px;color:#FFF;margin-bottom:22px;}
    .hero h2 {color:white;margin:0;font-size:29px;}
    .hero p {color:#CFE4EC;max-width:850px;margin-bottom:0;}
    .step {background:white;border:1px solid #E1E7EE;border-radius:12px;padding:18px;min-height:150px;}
    .step b {color:#0F8B8D;}
    </style>""")


def chart(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#172B4D"),
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner=False)
def featured(catalog, config):
    return build_features(catalog, config)


def set_catalog(raw, source):
    st.session_state.raw = raw
    st.session_state.source = source
    st.session_state.pop("bundle", None)
    st.session_state.pop("external", None)
    st.session_state.pop("external_id", None)


def get_bundle():
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("Primero ejecuta un entrenamiento o abre un experimento guardado en Registro.")
        return None
    return bundle


def summary(catalog, frame, config):
    st.html('<div class="eyebrow">OBSERVATORIO / RESUMEN EJECUTIVO</div>')
    st.title("Del catálogo a la evidencia.")
    st.html(
        '<div class="hero"><h2>Inteligencia sísmica, con trazabilidad.</h2><p>Explora el catálogo, '
        "comprueba cada transformación y contrasta modelos con datos que todavía no han visto. "
        "Todo el ciclo analítico en un solo lugar.</p></div>"
    )
    eligible = frame.loc[frame.observable & frame.warmup_complete]
    cols = st.columns(4)
    cols[0].metric("Eventos en catálogo", f"{len(catalog):,}")
    cols[1].metric("Eventos M ≥ umbral", int((catalog.mag >= config.magnitude).sum()))
    cols[2].metric("Casos positivos elegibles", int(eligible.target.sum()))
    cols[3].metric("Futuro pendiente", int((~frame.observable).sum()))
    st.caption(
        f"Cobertura UTC: {catalog.time.min():%d %b %Y} → {catalog.time.max():%d %b %Y} · "
        f"Fuente: {st.session_state.source} · {catalog.country.nunique()} regiones de referencia"
    )
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Una pregunta medible")
        st.write(
            f"Dado un evento del catálogo, ¿aparecerá otro de magnitud **≥ {config.magnitude:g}** "
            f"en su misma celda de **{config.grid:g}° × {config.grid:g}°** dentro de los siguientes "
            f"**{config.horizon} días**?"
        )
        monthly = catalog.set_index("time").resample("MS").size().rename("eventos").reset_index()
        chart(
            px.area(monthly, x="time", y="eventos", labels={"time": "Mes (UTC)", "eventos": "Eventos"}), 290
        )
    with right:
        st.subheader("Estado del proyecto")
        bundle = st.session_state.get("bundle")
        if bundle:
            st.success("Experimento disponible")
            st.write("**Seleccionado por CV:**", bundle["metadata"]["selected"])
            row = bundle["results"].iloc[0]
            st.metric("AP en prueba reservada", f"{row.AP:.3f}" if pd.notna(row.AP) else "No definida")
        else:
            st.info("Catálogo listo. Abre Modelo para generar la primera comparación.")
        st.write(
            "**Protocolo:** 60% entrenamiento / 20% validación / 20% prueba, con embargo temporal en ambas fronteras."
        )
    st.subheader("Un recorrido para la evaluación")
    cards = [
        (
            "01 / PREPROCESAMIENTO",
            "EDA, calidad y conversiones",
            "Inspecciona distribuciones, nulos, tipos y registros rechazados.",
        ),
        (
            "02 / MODELO",
            "Aprender sólo del pasado",
            "Compara F19/F25 y seis familias con validación de origen expansivo.",
        ),
        (
            "03 / EVALUACIÓN",
            "Medir antes de decidir",
            "AP, curvas, matriz de confusión y comparación con reglas simples.",
        ),
        (
            "04 / DESPLIEGUE",
            "Un flujo que sigue creciendo",
            "Carga nuevos eventos, conserva el modelo y espera etiquetas observables.",
        ),
    ]
    for col, (label, title, body) in zip(st.columns(4), cards):
        with col:
            st.html(f'<div class="step"><b>{label}</b><h4>{title}</h4><p>{body}</p></div>')
    st.info(
        "Alcance: evaluación estadística experimental de un catálogo. Los scores no constituyen una "
        "predicción sísmica validada ni deben utilizarse para decisiones de emergencia."
    )


def preprocessing(raw, cleaned, frame, config):
    st.html('<div class="eyebrow">01 / PREPROCESAMIENTO</div>')
    st.title("Conocer antes de modelar")
    tab_eda, tab_clean, tab_features = st.tabs(
        ["Exploración", "Limpieza y conversión", "Variables y etiqueta"]
    )
    catalog = cleaned.frame
    with tab_eda:
        c1, c2 = st.columns(2)
        countries = c1.multiselect(
            "País / referencia", sorted(catalog.country.unique()), default=sorted(catalog.country.unique())
        )
        dates = c2.date_input(
            "Intervalo de exploración (UTC)",
            (catalog.time.min().date(), catalog.time.max().date()),
            min_value=catalog.time.min().date(),
            max_value=catalog.time.max().date(),
        )
        subset = catalog.loc[catalog.country.isin(countries)]
        if len(dates) == 2:
            subset = subset.loc[(subset.time.dt.date >= dates[0]) & (subset.time.dt.date <= dates[1])]
        st.caption(f"{len(subset):,} eventos en la vista. Estos filtros sólo afectan la exploración.")
        if subset.empty:
            st.info("Sin eventos para estos filtros.")
        else:
            left, right = st.columns(2)
            with left:
                chart(
                    px.histogram(
                        subset,
                        x="mag",
                        nbins=30,
                        title="Distribución de magnitud",
                        labels={"mag": "Magnitud"},
                    )
                )
                chart(
                    px.scatter(
                        subset,
                        x="longitude",
                        y="latitude",
                        color="depth",
                        size="mag",
                        size_max=14,
                        hover_data=["time", "place", "mag"],
                        title="Epicentros · coordenadas",
                        color_continuous_scale="Teal",
                        labels={"longitude": "Longitud", "latitude": "Latitud", "depth": "Profundidad (km)"},
                    )
                ) if subset.mag.min() >= 0 else chart(
                    px.scatter(subset, x="longitude", y="latitude", color="depth")
                )
            with right:
                chart(
                    px.box(
                        subset,
                        x="country",
                        y="depth",
                        title="Profundidad y valores extremos",
                        labels={"country": "País", "depth": "Profundidad (km)"},
                    )
                )
                corr_cols = ["mag", "depth", "latitude", "longitude", "nst", "gap", "rms", "magError"]
                chart(
                    px.imshow(
                        subset[corr_cols].corr(),
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        title="Correlación de Pearson",
                    )
                )
            with st.expander("Gutenberg–Richter: análisis descriptivo"):
                mc = float(subset.mag.round(1).mode().iloc[0])
                tail = subset.loc[subset.mag >= mc, "mag"]
                b = np.log10(np.e) / (tail.mean() - (mc - 0.05))
                mags = np.sort(subset.mag.unique())
                chart(
                    px.line(
                        x=mags,
                        y=[(subset.mag >= m).sum() for m in mags],
                        log_y=True,
                        labels={"x": "Magnitud", "y": "N acumulado"},
                        title=f"Mc modal ≈ {mc:.1f} · b descriptivo ≈ {b:.2f}",
                    )
                )
                st.caption(
                    "Mc modal es una aproximación exploratoria; no demuestra completitud del catálogo."
                )
            st.dataframe(subset, hide_index=True, width="stretch")
            st.download_button(
                "Descargar vista explorada", csv_bytes(subset), "vista_catalogo.csv", "text/csv"
            )
    with tab_clean:
        cols = st.columns(4)
        for col, key in zip(cols, ["entrada", "válidos", "rechazados", "duplicados"]):
            col.metric(key.capitalize(), cleaned.audit[key])
        st.write(
            "Las fechas se convierten a UTC y los números inválidos a nulos. Se rechazan eventos sin fecha, "
            "coordenadas, magnitud o profundidad válidas. Los duplicados se resuelven por ID y evento; "
            "si hay revisiones se conserva la última actualización conocida."
        )
        st.write(
            "**Reglas físicas:** latitud [−90, 90], longitud [−180, 180], profundidad [−10, 800] km, magnitud [−2, 10]. "
            "Los indicadores de calidad negativos se convierten a nulos. Los extremos válidos se conservan."
        )
        st.dataframe(cleaned.schema, hide_index=True, width="stretch")
        st.caption(
            "Columnas constantes (documentadas, sin eliminación del contrato): "
            + ", ".join(cleaned.audit["constantes"])
        )
        missing = cleaned.schema.query("nulos > 0")
        if len(missing):
            chart(px.bar(missing, x="columna", y="nulos", title="Faltantes antes de imputar"), 270)
        st.info(
            "La imputación mediana, los indicadores de ausencia, el escalado y el one-hot se ajustan "
            "dentro de cada fold de entrenamiento. Ningún valor de prueba determina estas transformaciones."
        )
        with st.expander("Comparar registros originales y limpios"):
            st.write("Originales")
            st.dataframe(raw.head(25), hide_index=True)
            st.write("Limpios")
            st.dataframe(catalog.head(25), hide_index=True)
        if len(cleaned.rejected):
            st.dataframe(cleaned.rejected, hide_index=True)
            st.download_button(
                "Descargar rechazos con motivo", csv_bytes(cleaned.rejected), "rechazos.csv", "text/csv"
            )
        st.download_button("Descargar catálogo limpio", csv_bytes(catalog), "catalogo_limpio.csv", "text/csv")
        st.download_button(
            "Descargar auditoría", json.dumps(cleaned.audit, ensure_ascii=False, indent=2), "auditoria.json"
        )
    with tab_features:
        st.write(
            f"**Etiqueta:** otro evento M ≥ {config.magnitude:g} en la misma celda ({config.grid:g}°), "
            f"en el intervalo abierto por la izquierda (t, t + {config.horizon} días]."
        )
        st.write(
            "Los eventos simultáneos no cuentan como pasado ni como futuro. Las ventanas de actividad miran "
            "sólo hacia atrás. Las cuentas globales incluyen el evento actual; las de celda cuentan sólo el pasado."
        )
        st.write(
            "**F19:** ubicación, profundidad, calidad de medición, ciclos temporales y actividad global. "
            "**F25:** añade magnitud actual y actividad histórica de la celda. Ambos usan tipo de magnitud y país."
        )
        st.code(
            "F19 = "
            + ", ".join(F19)
            + "\n\nF25 añade = "
            + ", ".join(F25[len(F19) :])
            + "\n\nCategóricas = "
            + ", ".join(CATS),
            language=None,
        )
        cols = st.columns(3)
        cols[0].metric("Positivos observables", int(frame.target.sum()))
        cols[1].metric("Futuro sin observar", int((~frame.observable).sum()))
        cols[2].metric("Historia inicial < 30 días", int((~frame.warmup_complete).sum()))
        st.warning(
            "La etiqueta supone un catálogo completo hasta su último evento. No distingue una ausencia real "
            "de sismos de una interrupción de captura. Los primeros 30 días se excluyen del modelado."
        )
        st.dataframe(
            frame[
                [
                    "time",
                    "id",
                    "mag",
                    "lat_grid",
                    "lon_grid",
                    "cell_n7",
                    "cell_n30",
                    "events_last_30d",
                    "observable",
                    "warmup_complete",
                    "target",
                ]
            ],
            hide_index=True,
            width="stretch",
        )


def model_page(catalog, frame, config, store):
    st.html('<div class="eyebrow">02 / MODELO</div>')
    st.title("Aprender del pasado")
    st.write(
        "Los candidatos se comparan por **Average Precision (AP)** en CV temporal del bloque de entrenamiento. "
        "Después se fija el umbral en validación y se reporta la prueba reservada."
    )
    try:
        parts = temporal_partitions(frame, config.horizon)
    except ValueError as exc:
        st.error(str(exc))
        return
    rows = [
        {
            "bloque": k,
            "eventos": len(v),
            "positivos": int(v.target.sum()),
            "desde": v.time.min(),
            "hasta": v.time.max(),
        }
        for k, v in parts.items()
        if len(v)
    ]
    chart(
        px.timeline(
            pd.DataFrame(rows),
            x_start="desde",
            x_end="hasta",
            y="bloque",
            color="bloque",
            title="Fechas de separación y embargo",
        ),
        250,
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Embargo estricto de {config.horizon} días en cada frontera y en cada fold. Los porcentajes son "
        "aproximados por fechas únicas; eventos simultáneos siempre permanecen juntos."
    )
    if any(int(parts[k].target.sum()) < 10 for k in ["Entrenamiento", "Validación", "Prueba"]):
        st.warning(
            "Hay menos de 10 positivos en al menos un bloque. Las estimaciones pueden variar mucho; "
            "revisa los conteos y los folds omitidos antes de interpretar el ranking."
        )
    with st.form("training"):
        left, right = st.columns([2, 1])
        models = left.multiselect(
            "Familias a comparar", FAMILIES, default=["Regresión logística", "Random Forest", "Boosting"]
        )
        sets = right.multiselect("Conjuntos de variables", ["F19", "F25"], default=["F19", "F25"])
        folds = right.selectbox("Folds temporales", [3, 4], index=0)
        tune = st.checkbox("Comparar dos configuraciones por familia (mayor tiempo de entrenamiento)")
        st.caption(
            "Boosting usa HistGradientBoosting de scikit-learn. Stacking combina logística y Random Forest "
            "con predicciones temporales fuera de muestra. Siempre se incluyen dos referencias simples."
        )
        submitted = st.form_submit_button("Entrenar y evaluar", type="primary", width="stretch")
    if submitted:
        if not models or not sets:
            st.error("Selecciona al menos una familia y un conjunto de variables.")
            return
        bar = st.progress(0, text="Preparando experimento…")
        try:
            bundle = train_experiment(
                catalog,
                config,
                TrainConfig(tuple(sets), tuple(models), folds, tune),
                lambda fraction, text: bar.progress(fraction, text=text),
            )
            bundle["metadata"]["source"] = st.session_state.source
            store.save_experiment(bundle)
            st.session_state.bundle = bundle
            st.session_state.pop("external", None)
            st.success("Entrenamiento completado y guardado. Abre Evaluación para revisar las métricas.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    bundle = st.session_state.get("bundle")
    if bundle:
        st.subheader("Selección del experimento activo")
        st.write("**Candidato elegido por CV:**", bundle["metadata"]["selected"])
        st.dataframe(
            bundle["results"][["candidato", "CV_AP", "folds_evaluables", "seleccionable", "umbral"]],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Auditoría de folds y parámetros"):
            st.dataframe(bundle["folds"], hide_index=True, width="stretch")
            st.json(bundle["metadata"]["parameters"], expanded=False)
            for failure in bundle["metadata"]["failures"]:
                st.warning(failure)


def evaluation_page():
    st.html('<div class="eyebrow">03 / EVALUACIÓN</div>')
    st.title("Evidencia antes de decisiones")
    bundle = get_bundle()
    if bundle is None:
        return
    results = bundle["results"]
    st.caption(
        "Orden por elegibilidad y AP de CV. Sólo se seleccionan candidatos con todos los folds comparables evaluados. "
        "Las columnas AP/ROC/recall corresponden a prueba y no eligen al ganador. "
        "AP es Average Precision, una síntesis de precisión–recall; no es el área trapezoidal."
    )
    st.dataframe(results.drop(columns="probability"), hide_index=True, width="stretch")
    key = st.selectbox("Inspeccionar candidato", results.candidato.tolist())
    pset = bundle["predictions"][key]
    test = bundle["partitions"]["Prueba"]
    y, p = test.target.astype(int).to_numpy(), pset["test"]
    row = results.set_index("candidato").loc[key]
    st.caption(pset["threshold_note"] + f" Umbral guardado: {pset['threshold']:.6f}.")
    threshold = st.slider(
        "Simular umbral en prueba (exploratorio)",
        0.0,
        1.0,
        float(pset["threshold"]),
        0.001,
        key="sim_threshold_" + key,
    )
    st.caption(
        "La simulación no modifica el modelo guardado ni su umbral. Usar repetidamente la prueba para "
        "decidir el umbral convierte este bloque en exploratorio; confirma luego con datos externos."
    )
    current = metrics(y, p, threshold, bool(row.probability))
    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        ["AP", "Recall", "Precisión", "F2", "Tasa_señales"],
        [current[k] for k in ["AP", "Recall", "Precisión", "F2", "Tasa_señales"]],
    ):
        col.metric(label.replace("_", " "), "No definida" if value is None else f"{value:.3f}")
    left, right = st.columns(2)
    with left:
        if y.sum():
            precision, recall, _ = precision_recall_curve(y, p)
            fig = px.line(
                x=recall, y=precision, labels={"x": "Recall", "y": "Precisión"}, title="Precisión–recall"
            )
            fig.add_hline(y=float(y.mean()), line_dash="dash", annotation_text="Prevalencia de prueba")
            chart(fig)
        else:
            st.info("Prueba sin positivos: AP y recall no definidos.")
        matrix = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
        chart(
            px.imshow(
                matrix,
                text_auto=True,
                x=["Sin señal", "Con señal"],
                y=["Real negativo", "Real positivo"],
                color_continuous_scale="Teal",
                title="Matriz de confusión · umbral simulado",
            ),
            300,
        )
    with right:
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, p)
            fig = px.line(
                x=fpr, y=tpr, labels={"x": "Tasa de falsos positivos", "y": "Recall"}, title="Curva ROC"
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#9AA6B2"))
            chart(fig)
        weekly = pd.DataFrame({"fecha": test.time, "score": p, "real": y})
        chart(
            px.scatter(
                weekly,
                x="fecha",
                y="score",
                color="real",
                title="Scores a lo largo de la prueba",
                labels={"fecha": "Fecha UTC"},
            ),
            300,
        )
    constant_ap = results.loc[results.modelo == "Constante", "AP"].iloc[0]
    activity_ap = results.loc[results.modelo == "Actividad reciente", "AP"].iloc[0]
    if pd.notna(row.AP) and pd.notna(activity_ap):
        st.info(
            f"AP del candidato: {row.AP:.3f} · referencia constante: {constant_ap:.3f} · "
            f"actividad reciente: {activity_ap:.3f}. "
            + (
                "El candidato no supera a la regla de actividad en esta prueba."
                if row.AP <= activity_ap
                else "La mejora observada necesita confirmación en un catálogo externo."
            )
        )
    with st.expander("Incertidumbre y calibración"):
        if st.button("Calcular IC 95% por bloques temporales"):
            ci = block_intervals(y, p, test.time, bundle["metadata"]["feature_config"]["horizon"])
            st.json(ci)
            st.caption(
                "200 réplicas; se omiten réplicas de una sola clase. Bloques de al menos 7 días y del tamaño "
                "del horizonte. IC exploratorios con pocos positivos y posible dependencia espacial."
            )
        if row.probability:
            observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
            fig = go.Figure(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="Observado"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash")))
            fig.update_layout(
                title="Diagnóstico de calibración",
                xaxis_title="Salida media",
                yaxis_title="Frecuencia observada",
            )
            chart(fig, 280)
            st.caption(
                f"Brier: {current['Brier']:.4f}. El balanceo de clases puede distorsionar las probabilidades; "
                "la app presenta salidas como scores, sin garantizar calibración."
            )
        else:
            st.info(
                "Este candidato produce un score de ranking. Se omite Brier y el diagnóstico probabilístico."
            )
    with st.expander("Explicar variables en validación"):
        if row.modelo in ["Constante", "Actividad reciente"]:
            st.write(
                "Referencia constante: prevalencia de entrenamiento. Actividad reciente: transformación monótona "
                "de eventos en la celda durante los 30 días anteriores, escalada con entrenamiento."
            )
        elif st.button("Calcular importancia por permutación"):
            try:
                with st.spinner("Permutando variables de validación…"):
                    imp = importance(bundle, key)
                chart(
                    px.bar(
                        imp.head(15).sort_values("importancia"),
                        y="variable",
                        x="importancia",
                        orientation="h",
                        title="Caída de AP al permutar · validación",
                    ),
                    420,
                )
                st.dataframe(imp, hide_index=True)
            except ValueError as exc:
                st.warning(str(exc))
    export = test[["id", "time", "target"]].copy()
    export["score"], export["señal"] = p, p >= threshold
    st.download_button(
        "Descargar predicciones de prueba", csv_bytes(export), "predicciones_prueba.csv", "text/csv"
    )
    st.download_button(
        "Descargar comparación",
        csv_bytes(results.drop(columns="probability")),
        "comparacion_modelos.csv",
        "text/csv",
    )


def new_data_page(catalog, store):
    st.html('<div class="eyebrow">04 / NUEVOS DATOS</div>')
    st.title("El proyecto sigue aprendiendo")
    st.write(
        "Carga eventos posteriores al catálogo de un experimento para evaluarlo con el modelo congelado. "
        "Si incorporas historia adicional al catálogo, podrás crear después una nueva versión del modelo."
    )
    st.download_button(
        "Descargar plantilla CSV",
        csv_bytes(pd.DataFrame(columns=REQUIRED + ["id", "place", "magType", "magSource"])),
        "plantilla_eventos.csv",
        "text/csv",
    )
    st.caption(
        "time en ISO 8601 (UTC), coordenadas en grados, depth en km y mag numérica. "
        "La plantilla no contiene datos sintéticos. Opcionales: nst, gap, dmin, rms y errores de medición."
    )
    upload_tab, manual_tab = st.tabs(["Cargar lote CSV", "Registrar un evento"])
    with upload_tab:
        upload = st.file_uploader("Nuevos eventos", type="csv", key="new_upload")
        if upload:
            try:
                import hashlib

                digest = hashlib.sha256(upload.getvalue()).hexdigest()
                raw_new = read_csv(upload.getvalue())
                clean_new = clean_catalog(raw_new)
                if st.session_state.get("pending_upload_sha") != digest:
                    st.session_state.pending_new = clean_new.frame
                    st.session_state.pending_upload_sha = digest
                st.caption(
                    f"Válidos: {len(clean_new.frame)} · rechazos: {len(clean_new.rejected)} · duplicados: {clean_new.audit['duplicados']}"
                )
                if len(clean_new.rejected):
                    st.dataframe(clean_new.rejected, hide_index=True)
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("pending_new", None)
    with manual_tab:
        with st.form("event"):
            left, middle, right = st.columns(3)
            date = left.date_input("Fecha UTC", value=catalog.time.max().date())
            time = left.time_input("Hora UTC", value=datetime.min.time())
            lat = middle.number_input("Latitud", -90.0, 90.0, -21.0, step=0.01)
            lon = middle.number_input("Longitud", -180.0, 180.0, -68.0, step=0.01)
            depth = right.number_input("Profundidad (km)", -10.0, 800.0, 100.0)
            mag = right.number_input("Magnitud", -2.0, 10.0, 4.0, step=0.1)
            place = st.text_input("Referencia geográfica", "Bolivia")
            if st.form_submit_button("Preparar evento"):
                timestamp = datetime.combine(date, time).replace(tzinfo=timezone.utc).isoformat()
                new = pd.DataFrame(
                    [dict(time=timestamp, latitude=lat, longitude=lon, depth=depth, mag=mag, place=place)]
                )
                st.session_state.pending_new = clean_catalog(new).frame
                st.success("Evento preparado. Revisa el lote antes de evaluarlo o incorporarlo.")
    pending = st.session_state.get("pending_new")
    if pending is None:
        st.info("Carga un CSV o prepara un evento para continuar.")
        return
    st.subheader("Lote preparado")
    st.dataframe(
        pending[["id", "time", "latitude", "longitude", "depth", "mag", "country"]],
        hide_index=True,
        width="stretch",
    )
    duplicate_ids = int(pending.id.isin(catalog.id).sum())
    if duplicate_ids:
        st.warning(
            f"{duplicate_ids} IDs ya existen en el catálogo activo. La incorporación conserva esos eventos originales."
        )
    left, right = st.columns(2)
    bundle = st.session_state.get("bundle")
    if left.button("Evaluar con modelo congelado", type="primary", disabled=bundle is None):
        try:
            external = score_new(bundle, pending)
            store.save_validation(bundle, pending, *external)
            st.session_state.external = external
            st.session_state.external_id = fingerprint(pending) + bundle["metadata"]["id"]
            st.success("Validación guardada en el registro local.")
        except ValueError as exc:
            st.error(str(exc))
    if right.button("Incorporar al catálogo y guardar"):
        unique = pending.loc[~pending.id.isin(catalog.id)]
        try:
            merged = clean_catalog(pd.concat([catalog, unique], ignore_index=True)).frame
            store.save_catalog(merged)
            # Conserva el experimento congelado para comparaciones; nunca lo reentrena implícitamente.
            st.session_state.raw = merged
            st.session_state.source = "Catálogo local actualizado"
            st.success(
                f"Catálogo guardado con {len(merged):,} eventos. Crea un nuevo experimento en Modelo para reentrenar."
            )
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if bundle is None:
        st.caption("Entrena o abre un experimento para habilitar la evaluación externa.")
    external = st.session_state.get("external")
    if (
        external
        and bundle
        and st.session_state.get("external_id") == fingerprint(pending) + bundle["metadata"]["id"]
    ):
        rows, result, drift = external
        st.subheader("Resultado del modelo congelado")
        st.caption("Candidato: " + bundle["metadata"]["selected"])
        st.dataframe(
            rows[["id", "time", "mag", "score", "señal", "estado", "target"]],
            hide_index=True,
            width="stretch",
        )
        if result:
            st.write("Métricas sólo de los eventos con horizonte completamente observado:")
            st.dataframe(pd.DataFrame([result]), hide_index=True)
        else:
            st.info(
                "Se generaron scores. Aún no hay etiquetas evaluables: carga el catálogo posterior que cubra "
                "todo el horizonte. Un evento sin futuro observable permanece pendiente."
            )
        st.download_button(
            "Descargar validación externa",
            csv_bytes(rows.drop(columns="future_max")),
            "validacion_externa.csv",
            "text/csv",
        )
        with st.expander("Cobertura y cambios respecto al entrenamiento"):
            st.dataframe(drift, hide_index=True, width="stretch")
            missing = (
                pending[["nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError", "magNst"]]
                .isna()
                .mean()
            )
            st.write("Fracción ausente en indicadores de calidad:")
            st.dataframe(missing.rename_axis("variable").reset_index(name="ausencia"), hide_index=True)
            gap = pending.time.min() - bundle["catalog"].time.max()
            st.write(
                f"Distancia entre fin histórico y primer evento nuevo: {gap.total_seconds() / 86400:.1f} días."
            )
            st.warning(
                "Estas comprobaciones describen cobertura y rangos; no prueban estabilidad del modelo. "
                "Para validar, el lote debe contener todos los eventos capturados durante el intervalo, "
                "con la misma región y criterio de selección del catálogo original."
            )


def registry_page(store):
    st.html('<div class="eyebrow">05 / REGISTRO Y DESPLIEGUE</div>')
    st.title("Cada resultado tiene una versión")
    records = store.experiments()
    if records:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "fecha_UTC": r["created_at"],
                        "seleccionado": r["selected"],
                        "eventos": r["catalog_rows"],
                        "catálogo_SHA256": r["catalog_sha256"][:16],
                    }
                    for r in records
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        choice = st.selectbox(
            "Experimento guardado",
            [r["id"] for r in records],
            format_func=lambda ident: next(
                r["created_at"][:19] + " · " + r["selected"] for r in records if r["id"] == ident
            ),
        )
        if st.button("Abrir experimento congelado"):
            try:
                st.session_state.bundle = store.load_experiment(choice)
                st.session_state.pop("external", None)
                st.success("Experimento cargado. Evaluación usa su propio catálogo y sus bloques originales.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    else:
        st.info("Los entrenamientos se guardan automáticamente en el registro local.")
    bundle = st.session_state.get("bundle")
    if bundle:
        st.subheader("Ficha del experimento activo")
        st.json(bundle["metadata"], expanded=False)
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(bundle["metadata"], ensure_ascii=False, indent=2))
            archive.writestr("resultados.csv", csv_bytes(bundle["results"].drop(columns="probability")))
            archive.writestr("folds.csv", csv_bytes(bundle["folds"]))
            archive.writestr("catalogo_snapshot.csv", csv_bytes(bundle["catalog"]))
        st.download_button(
            "Descargar evidencia del experimento (.zip)",
            payload.getvalue(),
            "experimento.zip",
            "application/zip",
        )
    validations = store.validations()
    if validations:
        st.subheader("Validaciones externas guardadas")
        st.dataframe(
            pd.DataFrame(
                [{k: v for k, v in r.items() if k not in ("metrics", "catalog_sha256")} for r in validations]
            ),
            hide_index=True,
            width="stretch",
        )
        validation_id = st.selectbox(
            "Revisar validación",
            [v["id"] for v in validations],
            format_func=lambda ident: next(
                v["created_at"][:19] + " · " + str(v["eventos"]) + " eventos"
                for v in validations
                if v["id"] == ident
            ),
        )
        detail = next(v for v in validations if v["id"] == validation_id)
        report = store.validation_report(validation_id)
        st.write(
            "**Métricas:**",
            detail["metrics"] if detail["metrics"] else "Futuro pendiente: métricas todavía no disponibles.",
        )
        st.dataframe(
            pd.DataFrame(report["predictions"])[["id", "time", "score", "señal", "estado", "target"]],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Descargar informe de validación (.json)",
            json.dumps({"summary": detail, **report}, ensure_ascii=False, indent=2),
            "validacion_externa.json",
            "application/json",
        )
    st.subheader("Operación y despliegue")
    st.write(
        "El catálogo persistido y los modelos se guardan en SQLite y archivos locales dentro de `runtime/`. "
        "Configura `SISMO_DATA_DIR` para un volumen persistente. Los CSV se validan antes de incorporarlos; "
        "los modelos se cargan exclusivamente del registro creado por esta app."
    )
    st.code("uv sync --frozen\nuv run streamlit run app.py", language="bash")
    st.write(
        "La entrega incluye Docker, un volumen persistente y comprobación de salud. "
        "Esta instancia comparte el catálogo entre usuarios: para uso empresarial externo se necesita "
        "autenticación, control de acceso, respaldo y un proceso de aprobación de modelos."
    )


def main():
    style()
    store = Store(os.environ.get("SISMO_DATA_DIR", str(ROOT / "runtime")))
    if "raw" not in st.session_state:
        saved = store.load_catalog()
        set_catalog(
            saved if saved is not None else read_csv((ROOT / "datasets" / "query (2).csv").read_bytes()),
            "Catálogo local guardado" if saved is not None else "Catálogo incluido · USGS CSV",
        )
    with st.sidebar:
        st.html(
            '<h2 style="color:#0F8B8D;letter-spacing:-1px">◉ SismoLab</h2><p style="color:#63758B">Inteligencia sísmica / Laboratorio</p>'
        )
        page = st.radio(
            "Recorrido analítico",
            ["Resumen", "Preprocesamiento", "Modelo", "Evaluación", "Nuevos datos", "Registro"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("DEFINICIÓN DEL EXPERIMENTO")
        horizon = st.select_slider("Horizonte (días)", options=[1, 3, 7, 14, 30], value=7)
        magnitude = st.slider("Magnitud objetivo ≥", 3.0, 8.0, 5.0, 0.1)
        grid = st.select_slider("Celda geográfica (grados)", options=[0.25, 0.5, 1.0, 2.0], value=0.5)
        with st.expander("Catálogo de trabajo"):
            upload = st.file_uploader("Reemplazar catálogo en sesión", type="csv", key="catalog_upload")
            if st.button("Utilizar archivo", disabled=upload is None):
                try:
                    content = read_csv(upload.getvalue())
                    clean_catalog(content)
                    set_catalog(content, upload.name)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if st.button("Volver al catálogo incluido"):
                set_catalog(
                    read_csv((ROOT / "datasets" / "query (2).csv").read_bytes()),
                    "Catálogo incluido · USGS CSV",
                )
                st.rerun()
            if st.button("Guardar catálogo local"):
                store.save_catalog(clean_catalog(st.session_state.raw).frame)
                st.success("Catálogo guardado para próximas sesiones.")
        st.divider()
        st.caption("Estudio experimental · Región andina\n\nFechas y ventanas en UTC")
    config = FeatureConfig(horizon, magnitude, grid)
    try:
        cleaned = clean_catalog(st.session_state.raw)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    catalog = cleaned.frame
    frame = featured(catalog, config)
    bundle = st.session_state.get("bundle")
    if bundle and (
        bundle["metadata"]["feature_config"] != asdict(config)
        or bundle["metadata"]["catalog_sha256"] != fingerprint(catalog)
    ):
        st.warning(
            "El catálogo o la definición actual difieren del experimento activo. Evaluación y Nuevos datos "
            "usan la versión congelada indicada en Registro. Entrena de nuevo para comparar la configuración actual."
        )
    if page == "Resumen":
        summary(catalog, frame, config)
    elif page == "Preprocesamiento":
        preprocessing(st.session_state.raw, cleaned, frame, config)
    elif page == "Modelo":
        model_page(catalog, frame, config, store)
    elif page == "Evaluación":
        evaluation_page()
    elif page == "Nuevos datos":
        new_data_page(catalog, store)
    elif page == "Registro":
        registry_page(store)
    st.divider()
    st.caption(
        "SismoLab · Pipeline reproducible a partir de sismos_integrado.ipynb · Los resultados describen este catálogo y su cobertura."
    )


if __name__ == "__main__":
    main()
