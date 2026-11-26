"""Modelos interpretables y validación estadística del corpus UMSA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from umsa.data import METRIC_LABELS, NUMERIC_METRICS

FEATURES = [
    "num_paginas",
    "total_citas_detectadas",
    "coh_obj_apl_v2",
    "coh_obj_con_v2",
    "burstiness_oraciones",
    "riqueza_lexica_mattr",
    "pct_estructura",
]


@dataclass
class RigorModel:
    model: Pipeline
    features: list[str]
    target: str
    auc: float
    accuracy: float
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    umbral: float


def feature_matrix(corpus: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = corpus[FEATURES].copy()
    frame = frame.fillna(frame.median(numeric_only=True))
    target = corpus["rigor_alto"].to_numpy(dtype=int)
    return frame, target


def train_rigor_model(corpus: pd.DataFrame) -> RigorModel:
    """Clasificador logístico interpretable para estimar alto rigor académico."""
    frame, target = feature_matrix(corpus)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    splits = min(5, int(np.bincount(target).min()))
    if splits < 2:
        raise ValueError("El corpus necesita al menos dos casos por clase para validar el modelo.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    probabilities = cross_val_predict(model, frame, target, cv=cv, method="predict_proba")[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    model.fit(frame, target)
    coefficients = pd.DataFrame(
        {
            "variable": FEATURES,
            "etiqueta": [METRIC_LABELS.get(feature, "Estructura reglamentaria") for feature in FEATURES],
            "coeficiente": model.named_steps["logit"].coef_[0],
        }
    ).sort_values("coeficiente", key=np.abs, ascending=False)
    predictions = corpus[["archivo_pdf", "titulo", "indice_rigor_academico", "nota_numeral", "rigor_alto"]].copy()
    predictions["probabilidad_alto_rigor"] = probabilities
    predictions["prediccion"] = predicted
    return RigorModel(
        model=model,
        features=FEATURES,
        target="rigor_alto",
        auc=float(roc_auc_score(target, probabilities)),
        accuracy=float((predicted == target).mean()),
        predictions=predictions,
        coefficients=coefficients,
        umbral=float(corpus["indice_rigor_academico"].median()),
    )


def correlation_table(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = corpus[corpus["nota_numeral"].notna()]
    for metric in NUMERIC_METRICS + ["indice_rigor_academico"]:
        if metric not in corpus:
            continue
        series = corpus[metric]
        if series.notna().sum() >= 3:
            rho, p_value = spearmanr(corpus[metric], corpus["indice_rigor_academico"], nan_policy="omit")
            rows.append(
                {
                    "variable": METRIC_LABELS.get(metric, "Índice de rigor"),
                    "spearman_vs_indice": float(rho),
                    "p_value": float(p_value),
                    "n": int(series.notna().sum()),
                }
            )
        if len(valid) >= 3 and metric != "indice_rigor_academico":
            rho, p_value = spearmanr(valid[metric], valid["nota_numeral"], nan_policy="omit")
            rows.append(
                {
                    "variable": METRIC_LABELS.get(metric, metric) + " vs. nota real",
                    "spearman_vs_indice": float(rho),
                    "p_value": float(p_value),
                    "n": len(valid),
                }
            )
    return pd.DataFrame(rows)


def note_validation(corpus: pd.DataFrame) -> dict:
    valid = corpus[corpus["nota_numeral"].notna()]
    if len(valid) < 3:
        return {"n": len(valid), "spearman": None, "p_value": None, "pearson": None}
    rho, p_value = spearmanr(valid["nota_numeral"], valid["indice_rigor_academico"])
    pearson_r, _ = pearsonr(valid["nota_numeral"], valid["indice_rigor_academico"])
    return {
        "n": len(valid),
        "spearman": float(rho),
        "p_value": float(p_value),
        "pearson": float(pearson_r),
    }


def feature_medians(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in FEATURES:
        series = corpus[metric].dropna()
        rows.append(
            {
                "variable": METRIC_LABELS.get(metric, "Estructura reglamentaria"),
                "mediana": float(series.median()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)
