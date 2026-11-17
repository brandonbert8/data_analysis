"""Selección por CV temporal, umbral en validación y prueba reservada."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import platform
import uuid

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sismo import PIPELINE_VERSION

from sismo.data import fingerprint
from sismo.features import CATS, F19, F25, FeatureConfig, build_features, temporal_folds, temporal_partitions

SEED = 42
FAMILIES = [
    "Regresión logística",
    "SVM RBF",
    "Random Forest",
    "Bagging de árboles",
    "Boosting",
    "Stacking temporal",
]
BASELINES = ["Constante", "Actividad reciente"]


@dataclass(frozen=True)
class TrainConfig:
    feature_sets: tuple = ("F19", "F25")
    models: tuple = ("Regresión logística", "Random Forest", "Boosting")
    folds: int = 3
    tune: bool = False


def make_pipeline(name, features, variant=0):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Sin dato")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer([("numeric", numeric, features), ("categorical", categorical, CATS)])
    classifiers = {
        "Regresión logística": LogisticRegression(
            C=[0.1, 1][variant], class_weight="balanced", max_iter=1500, random_state=SEED
        ),
        # Sin probability=True: evita la calibración interna por CV aleatoria del SVC.
        "SVM RBF": SVC(C=[0.3, 3][variant], gamma="scale", class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=[4, 8][variant],
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=SEED,
        ),
        "Bagging de árboles": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=[3, 5][variant], class_weight="balanced", min_samples_leaf=5
            ),
            n_estimators=50,
            max_samples=0.8,
            random_state=SEED,
            n_jobs=1,
        ),
        "Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=[7, 15][variant],
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=SEED,
        ),
    }
    return Pipeline([("preprocess", pre), ("classifier", classifiers[name])])


def scores(estimator, frame):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(frame)[:, 1]
    return expit(estimator.decision_function(frame))


class TemporalStack(ClassifierMixin, BaseEstimator):
    """Metamodelo ajustado con predicciones fuera de muestra y embargo real."""

    def __init__(self, features=tuple(F25), horizon=7):
        self.features = features
        self.horizon = horizon

    def fit(self, X, y):
        y = np.asarray(y)
        names = ["Regresión logística", "Random Forest"]
        oof = np.full((len(X), len(names)), np.nan)
        for a, b in temporal_folds(X, self.horizon, 3):
            if not len(a) or not len(b) or len(np.unique(y[a])) < 2:
                continue
            for j, name in enumerate(names):
                model = make_pipeline(name, list(self.features)).fit(X.iloc[a], y[a])
                oof[b, j] = scores(model, X.iloc[b])
        valid = np.isfinite(oof).all(axis=1)
        if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
            raise ValueError(
                "Stacking sin suficientes etiquetas de ambas clases en sus folds temporales internos."
            )
        self.meta_ = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            oof[valid], y[valid]
        )
        self.bases_ = [make_pipeline(name, list(self.features)).fit(X, y) for name in names]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(np.column_stack([scores(model, X) for model in self.bases_]))


class Baseline(ClassifierMixin, BaseEstimator):
    def __init__(self, activity=False):
        self.activity = activity

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y))
        self.scale_ = max(float(X.cell_n30.mean()), 1)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (
            1 - np.exp(-X.cell_n30.to_numpy() / self.scale_)
            if self.activity
            else np.full(len(X), self.prevalence_)
        )
        return np.column_stack([1 - p, p])


def metrics(y, p, threshold, probability=True):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    prev = float(y.mean())
    precision = float(precision_score(y, pred, zero_division=0))
    return {
        "AP": float(average_precision_score(y, p)) if y.sum() else None,
        "ROC_AUC": float(roc_auc_score(y, p)) if both else None,
        "Recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else None,
        "Precisión": precision,
        "F2": float(fbeta_score(y, pred, beta=2, zero_division=0)) if y.sum() else None,
        "MCC": float(matthews_corrcoef(y, pred)) if both else None,
        "Balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "Brier": float(brier_score_loss(y, p)) if probability else None,
        "Lift": precision / prev if prev else None,
        "Tasa_señales": float(pred.mean()),
        "Prevalencia": prev,
        "N": len(y),
        "Positivos": int(y.sum()),
    }


def pick_threshold(y, p):
    """F2 solamente en validación. No finge optimización si hay una única clase."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5, "Validación con una sola clase: umbral fijo 0.5, sin optimización."
    precision, recall, thresholds = precision_recall_curve(y, p)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f2, f2.max()))[-1]
    return float(thresholds[best]), "Umbral que maximiza F2 en validación (prioriza recall)."


def train_experiment(catalog, feature_config=FeatureConfig(), config=TrainConfig(), progress=None):
    frame = build_features(catalog, feature_config)
    parts = temporal_partitions(frame, feature_config.horizon)
    train, val, test = [parts[k] for k in ("Entrenamiento", "Validación", "Prueba")]
    y_train = train.target.astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Entrenamiento tiene una sola clase. Amplía el catálogo o revisa horizonte/celda/magnitud."
        )
    if not config.feature_sets or not config.models:
        raise ValueError("Selecciona al menos un conjunto de variables y un modelo.")
    if set(config.models) - set(FAMILIES) or set(config.feature_sets) - {"F19", "F25"}:
        raise ValueError("Configuración de modelos no reconocida.")
    folds = list(temporal_folds(train, feature_config.horizon, config.folds))
    expected_cv = sum(
        bool(len(a) and len(b) and len(np.unique(y_train[a])) == 2 and y_train[b].sum()) for a, b in folds
    )
    if not expected_cv:
        raise ValueError("Ningún fold temporal permite medir AP. Se necesitan más positivos históricos.")
    candidates = [("—", n, 0, Baseline(activity=n == "Actividad reciente")) for n in BASELINES]
    for feature_set in config.feature_sets:
        features = F19 if feature_set == "F19" else F25
        for name in config.models:
            variants = range(2) if config.tune and name != "Stacking temporal" else range(1)
            for variant in variants:
                estimator = (
                    TemporalStack(tuple(features), feature_config.horizon)
                    if name == "Stacking temporal"
                    else make_pipeline(name, features, variant)
                )
                candidates.append((feature_set, name, variant, estimator))
    fold_rows, candidate_rows, fitted, predictions = [], [], {}, {}
    failures = []
    with threadpool_limits(limits=1):
        for step, (feature_set, name, variant, estimator) in enumerate(candidates):
            key = f"{feature_set} · {name} · v{variant + 1}"
            if progress:
                progress(step / len(candidates), key)
            aps = []
            for fold, (a, b) in enumerate(folds, 1):
                row = {
                    "candidato": key,
                    "fold": fold,
                    "train_n": len(a),
                    "valid_n": len(b),
                    "train_positivos": int(y_train[a].sum()),
                    "valid_positivos": int(y_train[b].sum()),
                    "AP": None,
                }
                if len(a) and len(b):
                    row.update(
                        train_hasta=str(train.iloc[a].time.max()), valid_desde=str(train.iloc[b].time.min())
                    )
                if not len(a) or not len(b) or len(np.unique(y_train[a])) < 2:
                    row["estado"] = "Omitido: entrenamiento sin ambas clases o bloque vacío"
                    fold_rows.append(row)
                    continue
                try:
                    model = clone(estimator).fit(train.iloc[a], y_train[a])
                    p = scores(model, train.iloc[b])
                    row["AP"] = float(average_precision_score(y_train[b], p)) if y_train[b].sum() else None
                    row["estado"] = "OK" if row["AP"] is not None else "Sin positivos: AP no definida"
                    if row["AP"] is not None:
                        aps.append(row["AP"])
                except ValueError as exc:
                    row["estado"] = str(exc)
                fold_rows.append(row)
            eligible = len(aps) == expected_cv
            if not eligible:
                failures.append(
                    f"{key}: CV incompleta ({len(aps)}/{expected_cv} folds comparables); no participa en la selección."
                )
            try:
                model = clone(estimator).fit(train, y_train)
            except ValueError as exc:
                failures.append(f"{key}: {exc}")
                continue
            fitted[key] = model
            pv, pt = scores(model, val), scores(model, test)
            threshold, threshold_note = pick_threshold(val.target, pv)
            predictions[key] = {
                "validation": pv,
                "test": pt,
                "threshold": threshold,
                "threshold_note": threshold_note,
            }
            # Actividad y SVM son scores; no se calcula Brier como si fueran probabilidades.
            probability = name not in ("Actividad reciente", "SVM RBF")
            candidate_rows.append(
                {
                    "candidato": key,
                    "variables": feature_set,
                    "modelo": name,
                    "variante": variant + 1,
                    "CV_AP": float(np.mean(aps)) if aps else np.nan,
                    "seleccionable": eligible,
                    "folds_evaluables": len(aps),
                    "umbral": threshold,
                    "probability": probability,
                    **metrics(test.target, pt, threshold, probability),
                }
            )
    if not candidate_rows:
        raise ValueError("Ningún candidato evaluable. " + " ".join(failures))
    # Ganador por CV de ENTRENAMIENTO, jamás por métricas de prueba o por ajustes manuales del umbral.
    table = (
        pd.DataFrame(candidate_rows)
        .sort_values(["seleccionable", "CV_AP"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    best = table.iloc[0].candidato
    now = datetime.now(timezone.utc).isoformat()
    parameters = {
        key: {
            k: str(v)
            for k, v in model.get_params().items()
            if k.startswith("classifier__") or k in ("activity", "horizon", "features")
        }
        for key, model in fitted.items()
    }
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "id": str(uuid.uuid4()),
        "created_at": now,
        "catalog_sha256": fingerprint(catalog),
        "feature_config": asdict(feature_config),
        "train_config": asdict(config),
        "selected": best,
        "catalog_rows": len(catalog),
        "coverage_start": str(catalog.time.min()),
        "coverage_end": str(catalog.time.max()),
        "fit_until": str(train.time.max()),
        "validation_until": str(val.time.max()),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "seed": SEED,
        "failures": failures,
        "parameters": parameters,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "selected_test_metrics": json.loads(table.iloc[0].to_json()),
    }
    if progress:
        progress(1.0, "Experimento completado")
    return {
        "metadata": metadata,
        "models": fitted,
        "predictions": predictions,
        "results": table,
        "folds": pd.DataFrame(fold_rows),
        "partitions": parts,
        "catalog": catalog.copy(),
    }


def block_intervals(y, p, times, horizon=7, repetitions=200):
    """IC percentil por bloques temporales; inferencia exploratoria, no garantía."""
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    days = pd.to_datetime(times).astype("int64").to_numpy() // 86_400_000_000_000
    blocks = days // max(7, horizon)
    groups = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    rng = np.random.default_rng(SEED)
    aps, rocs = [], []
    for _ in range(repetitions):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            aps.append(average_precision_score(y[idx], p[idx]))
            rocs.append(roc_auc_score(y[idx], p[idx]))
    return {
        "AP": np.quantile(aps, [0.025, 0.975]).tolist() if aps else None,
        "ROC_AUC": np.quantile(rocs, [0.025, 0.975]).tolist() if rocs else None,
        "réplicas_válidas": len(aps),
        "bloques": len(groups),
    }


def importance(bundle, key):
    val = bundle["partitions"]["Validación"]
    if val.target.sum() == 0:
        raise ValueError("No hay positivos en validación para calcular importancia por AP.")
    cols = (F19 if bundle["results"].set_index("candidato").loc[key, "variables"] == "F19" else F25) + CATS

    def scorer(model, X, y):
        return average_precision_score(y, scores(model, X))

    with threadpool_limits(limits=1):
        result = permutation_importance(
            bundle["models"][key],
            val[cols + ["time"]],
            val.target.astype(int),
            scoring=scorer,
            n_repeats=3,
            random_state=SEED,
            n_jobs=1,
        )
    return (
        pd.DataFrame(
            {
                "variable": cols + ["time"],
                "importancia": result.importances_mean,
                "desviación": result.importances_std,
            }
        )
        .query("variable != 'time'")
        .sort_values("importancia", ascending=False)
    )


def score_new(bundle, new_catalog):
    """Mantiene el modelo congelado; usa historia anterior para variables y futuro sólo para etiquetas."""
    history = bundle["catalog"]
    if new_catalog.id.isin(history.id).any():
        raise ValueError("El lote contiene IDs del catálogo del experimento; usa eventos nuevos.")
    # Un lote externo debe estar DESPUÉS de la cobertura conocida para no contaminar la prueba histórica.
    if (new_catalog.time <= history.time.max()).any():
        raise ValueError(
            "La validación externa requiere fechas posteriores al último evento del experimento."
        )
    from sismo.data import clean_catalog

    combined = clean_catalog(pd.concat([history, new_catalog], ignore_index=True)).frame
    config = FeatureConfig(**bundle["metadata"]["feature_config"])
    frame = build_features(combined, config)
    rows = frame.loc[frame.id.isin(new_catalog.id)].copy()
    key = bundle["metadata"]["selected"]
    with threadpool_limits(limits=1):
        rows["score"] = scores(bundle["models"][key], rows)
    threshold = bundle["predictions"][key]["threshold"]
    rows["señal"] = rows.score >= threshold
    rows["estado"] = np.where(rows.observable, "Evaluable", "Pendiente: futuro incompleto")
    comparable = rows.loc[rows.observable]
    probability = bool(bundle["results"].set_index("candidato").loc[key, "probability"])
    result = metrics(comparable.target, comparable.score, threshold, probability) if len(comparable) else None
    drift = []
    train = bundle["partitions"]["Entrenamiento"]
    for col in ["latitude", "longitude", "depth", "mag", "events_last_30d", "cell_n30"]:
        outside = ~rows[col].between(train[col].min(), train[col].max())
        drift.append(
            {
                "variable": col,
                "train_min": train[col].min(),
                "train_max": train[col].max(),
                "nuevos_min": rows[col].min(),
                "nuevos_max": rows[col].max(),
                "fuera_de_rango_%": 100 * outside.mean(),
            }
        )
    return rows, result, pd.DataFrame(drift)
