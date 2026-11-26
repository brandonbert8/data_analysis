"""Carga, normalización y métricas del corpus de tesis de Informática UMSA.

El corpus de referencia vive en ``dataset/``. La app no depende de Google Colab ni
de rutas absolutas: todo se resuelve desde la raíz del repositorio.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
PDF_DIR = DATASET_DIR / "PDFs"
CORPUS_PARQUET = DATASET_DIR / "df_maestro_final_modelos_70.parquet"
CORPUS_CSV = DATASET_DIR / "resultados_modelos_umsa_70.csv"
BUNDLE_PATH = DATASET_DIR / "auditor_umsa_produccion.joblib"
METADATA_PATH = DATASET_DIR / "metadatos_informatica.csv"
BENCHMARK_PATH = DATASET_DIR / "benchmark_ml_vs_dl_umsa.csv"

SECTION_LABELS = {
    "sec_introduccion": "Introducción / Capítulo I",
    "sec_planteamiento": "Planteamiento del problema",
    "sec_objetivos": "Objetivos",
    "sec_marco_teorico": "Marco teórico",
    "sec_marco_aplicativo": "Marco aplicativo / Ingeniería",
    "sec_pruebas": "Pruebas y resultados",
    "sec_conclusiones": "Conclusiones",
    "sec_bibliografia": "Bibliografía / Referencias",
}

SECTION_PATTERNS = {
    "sec_introduccion": r"\b(INTRODUCCI[ÓO]N|CAP[ÍI]TULO\s*(I|1))\b",
    "sec_planteamiento": r"PLANTEAMIENTO\s+DEL\s+PROBLEMA",
    "sec_objetivos": r"\bOBJETIVOS?\b",
    "sec_marco_teorico": r"MARCO\s+TE[ÓO]RICO",
    "sec_marco_aplicativo": (
        r"(MARCO\s+APLICATIVO|INGENIER[ÍI]A\s+DEL\s+PROYECTO|DESARROLLO\s+DEL\s+SISTEMA|"
        r"DISE[ÑN]O\s+E\s+IMPLEMENTACI[ÓO]N)"
    ),
    "sec_pruebas": r"\b(PRUEBAS|RESULTADOS|VALIDACI[ÓO]N|EVALUACI[ÓO]N)\b",
    "sec_conclusiones": r"\bCONCLUSIONES\b",
    "sec_bibliografia": r"\b(BIBLIOGRAF[ÍI]A|REFERENCIAS\s+BIBLIOGR[ÁA]FICAS)\b",
}

SECTION_WEIGHTS = {
    "sec_introduccion": 0.15,
    "sec_planteamiento": 0.15,
    "sec_objetivos": 0.15,
    "sec_marco_teorico": 0.15,
    "sec_marco_aplicativo": 0.15,
    "sec_pruebas": 0.10,
    "sec_conclusiones": 0.10,
    "sec_bibliografia": 0.05,
}

NUMERIC_METRICS = [
    "num_paginas",
    "total_citas_detectadas",
    "coh_obj_apl_v2",
    "coh_obj_con_v2",
    "burstiness_oraciones",
    "riqueza_lexica_mattr",
]

LEXICAL_METRICS = ["coh_lex_apl", "coh_lex_con"]

OBJECTIVE_PATTERN = r"(OBJETIVO\s+GENERAL|PLANTEAMIENTO\s+DEL\s+PROBLEMA)"
APPLICATIVE_PATTERN = (
    r"(MARCO\s+APLICATIVO|INGENIER[ÍI]A\s+DEL\s+PROYECTO|DESARROLLO\s+DEL\s+SISTEMA|"
    r"DISE[ÑN]O\s+E\s+IMPLEMENTACI[ÓO]N)"
)
CONCLUSION_PATTERN = r"\bCONCLUSIONES\b"

METRIC_LABELS = {
    "num_paginas": "Páginas",
    "total_citas_detectadas": "Citas detectadas",
    "coh_obj_apl_v2": "Coherencia objetivos ↔ aplicativo",
    "coh_obj_con_v2": "Coherencia objetivos ↔ conclusiones",
    "burstiness_oraciones": "Burstiness de oraciones",
    "riqueza_lexica_mattr": "Riqueza léxica (MATTR)",
    "coh_lex_apl": "Coherencia léxica objetivos ↔ aplicativo",
    "coh_lex_con": "Coherencia léxica objetivos ↔ conclusiones",
}

STOPWORDS_ES = {
    "universidad", "mayor", "san", "andres", "andrés", "facultad", "ciencias", "puras",
    "naturales", "carrera", "informatica", "informática", "proyecto", "grado", "tesis",
    "capitulo", "capítulo", "tabla", "figura", "cuadro", "pagina", "página", "sistema",
    "informacion", "información", "desarrollo", "mediante", "través", "cada", "asi", "así",
    "ser", "puede", "pueden", "parte", "forma", "manera", "caso", "bolivia", "paz", "datos",
    "usuario", "usuarios", "proceso", "procesos", "gestión", "gestion", "control", "the",
    "and", "for", "with", "from", "that", "this", "are", "was", "were", "para", "como",
    "entre", "sobre", "desde", "hasta", "donde", "cuando", "porque", "también", "tambien",
}


def _read_corpus_file() -> pd.DataFrame:
    if CORPUS_PARQUET.exists():
        return pd.read_parquet(CORPUS_PARQUET)
    if CORPUS_CSV.exists():
        return pd.read_csv(CORPUS_CSV)
    raise FileNotFoundError(
        "No se encontró el corpus en dataset/. Se esperaba "
        "df_maestro_final_modelos_70.parquet o resultados_modelos_umsa_70.csv."
    )


def load_corpus() -> pd.DataFrame:
    """Carga el corpus final de 70 tesis y normaliza tipos y columnas derivadas."""
    df = _read_corpus_file().copy()
    df["fecha"] = pd.to_numeric(df.get("fecha"), errors="coerce").astype("Int64")
    df["nota_numeral"] = pd.to_numeric(df.get("nota_numeral"), errors="coerce")
    df["cluster_tematico"] = pd.to_numeric(df.get("cluster_tematico"), errors="coerce").astype("Int64")
    df["indice_rigor_academico"] = pd.to_numeric(df["indice_rigor_academico"], errors="coerce")
    for column in SECTION_LABELS:
        if column not in df:
            df[column] = False
        df[column] = df[column].fillna(False).astype(bool)
    for column in NUMERIC_METRICS:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("titulo", "autores", "tutores", "mencion", "modalidad", "archivo_pdf"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    if "texto_completo" not in df:
        df["texto_completo"] = ""
    df["texto_completo"] = df["texto_completo"].fillna("").astype(str)
    df["pct_estructura"] = df[list(SECTION_LABELS)].mean(axis=1)
    umbral = float(df["indice_rigor_academico"].median())
    df["rigor_alto"] = (df["indice_rigor_academico"] >= umbral).astype(int)
    df["anomalia"] = df.get("es_anomalia_estructural", 1).fillna(1).astype(int).eq(-1)
    df = add_lexical_coherence(df)
    return df.reset_index(drop=True)


def load_bundle() -> dict:
    """Carga el paquete de producción exportado por el cuaderno."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(BUNDLE_PATH)


def load_benchmark() -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        return pd.DataFrame(columns=["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    return pd.read_csv(BENCHMARK_PATH)


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        return pd.DataFrame(columns=["id", "titulo", "autores", "tutores", "fecha", "resumen", "archivo_pdf"])
    return pd.read_csv(METADATA_PATH, encoding="utf-8-sig")


def corpus_fingerprint(df: pd.DataFrame) -> str:
    columns = ["id", "archivo_pdf", "indice_rigor_academico", "num_paginas", "total_citas_detectadas"]
    payload = df[[c for c in columns if c in df]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def available_pdfs() -> list[str]:
    if not PDF_DIR.exists():
        return []
    return sorted(path.name for path in PDF_DIR.glob("*.pdf"))


def structure_presence(text: str) -> dict[str, bool]:
    upper = (text or "").upper()
    return {key: bool(re.search(pattern, upper)) for key, pattern in SECTION_PATTERNS.items()}


def weighted_structure(presence: dict[str, bool]) -> float:
    return float(sum(SECTION_WEIGHTS[key] for key, present in presence.items() if present))


def count_citations(text: str) -> tuple[int, int]:
    apa = len(
        re.findall(
            r"\([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ\s,&\.]+,\s*(19|20)\d{2}\)",
            text or "",
        )
    )
    narrative = len(
        re.findall(
            r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:et\s+al\.?|y\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+))?\s+\((19|20)\d{2}\)",
            text or "",
        )
    )
    ieee = len(re.findall(r"\[\d{1,3}\]", text or ""))
    return apa + narrative, ieee


def extract_section(text: str, pattern: str, max_chars: int = 4000) -> str:
    """Extrae una sección evitando la tabla de contenido (puntos guía)."""
    for match in re.finditer(pattern, text or "", re.IGNORECASE):
        fragment = text[match.start() : match.start() + max_chars]
        if fragment.count("....") < 3 and len(fragment.strip()) > 400:
            return fragment.strip()
    return ""


def lexical_coherence(text_a: str, text_b: str) -> float:
    """Aproximación léxica (TF-IDF) a la coherencia semántica del cuaderno."""
    if not text_a or not text_b or len(text_a.strip()) < 120 or len(text_b.strip()) < 120:
        return float("nan")
    try:
        vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words=list(STOPWORDS_ES),
            token_pattern=r"\b[a-záéíóúñ]{3,24}\b",
        )
        matrix = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        return float("nan")


def stylometry(text: str) -> tuple[float, float]:
    body = text[5000:45000] if len(text) > 10000 else text
    sentences = [s.strip() for s in re.split(r"[.!?]+", body or "") if len(s.split()) >= 4]
    if len(sentences) < 10:
        return float("nan"), float("nan")
    lengths = [len(sentence.split()) for sentence in sentences]
    burstiness = float(np.std(lengths))
    words = re.findall(r"\b[a-záéíóúñ]{3,}\b", body.lower())
    if len(words) < 100:
        mattr = len(set(words)) / max(1, len(words))
    else:
        windows = [len(set(words[i : i + 100])) / 100.0 for i in range(0, len(words) - 100, 50)]
        mattr = float(np.mean(windows))
    return round(burstiness, 4), round(mattr, 4)


def normalize_metric_name(value: float, metric: str, percentiles: dict) -> float:
    """Normaliza una métrica contra los percentiles del corpus de referencia."""
    p95 = percentiles.get(metric, {}).get("p95")
    if p95 is None:
        return float("nan")
    return float(np.clip(value / p95, 0, 1)) if p95 else float("nan")


def add_lexical_coherence(corpus: pd.DataFrame) -> pd.DataFrame:
    """Calcula coherencia léxica con el mismo método que la auditoría de PDFs.

    Estas columnas permiten calibrar los valores TF-IDF de un documento nuevo a la
    escala de los embeddings del corpus original (coh_obj_apl_v2 / coh_obj_con_v2).
    """
    applicative: list[float] = []
    conclusions: list[float] = []
    for text in corpus.get("texto_completo", []):
        objective = extract_section(str(text), OBJECTIVE_PATTERN, 4000)
        applicative.append(lexical_coherence(objective, extract_section(str(text), APPLICATIVE_PATTERN, 6000)))
        conclusions.append(lexical_coherence(objective, extract_section(str(text), CONCLUSION_PATTERN, 4000)))
    corpus["coh_lex_apl"] = applicative
    corpus["coh_lex_con"] = conclusions
    return corpus


def calibrate_coherence(value: float, lexical_reference: pd.Series, target_reference: pd.Series) -> float:
    """Mapea una coherencia léxica a la distribución de embeddings del corpus.

    El cuaderno mide coherencia con sentence-transformers; la app usa TF-IDF para
    evitar torch. La calibración por cuantiles conserva el orden y permite comparar
    un documento nuevo contra los percentiles históricos del cuaderno.
    """
    if value is None or np.isnan(value):
        return float("nan")
    lexical = pd.to_numeric(lexical_reference, errors="coerce").dropna()
    target = pd.to_numeric(target_reference, errors="coerce").dropna()
    if lexical.empty or target.empty:
        return float(value)
    percentile = float((lexical < value).mean())
    return float(target.quantile(percentile))


def rigor_score(
    presence: dict[str, bool],
    coherence_applicative: float,
    coherence_conclusions: float,
    total_citations: int,
    pages: int,
) -> float:
    coherence_applicative = 0.0 if np.isnan(coherence_applicative) else coherence_applicative
    coherence_conclusions = 0.0 if np.isnan(coherence_conclusions) else coherence_conclusions
    citations_norm = min(1.0, total_citations / 55.0)
    pages_norm = min(1.0, pages / 150.0)
    structure_ratio = sum(presence.values()) / len(presence) if presence else 0.0
    score = (
        0.25 * structure_ratio
        + 0.25 * coherence_applicative
        + 0.15 * coherence_conclusions
        + 0.20 * citations_norm
        + 0.15 * pages_norm
    ) * 100
    return round(float(score), 2)


def verdict(score: float, missing: list[str]) -> str:
    if score >= 72 and not missing:
        return "SÓLIDO / LISTO PARA DEFENSA (Nivel Top 25% UMSA)"
    if score >= 60 and len(missing) <= 1:
        return "ACEPTABLE CON OBSERVACIONES MENORES (Nivel Estándar UMSA)"
    return "POBRE / DEFICIENTE (Requiere reestructuración antes de entrega)"


def diagnostics(
    presence: dict[str, bool],
    total_citations: int,
    coherence_applicative: float,
    coherence_conclusions: float,
    pages: int,
    burstiness: float,
    percentiles: dict,
) -> list[str]:
    missing = [SECTION_LABELS[key] for key, present in presence.items() if not present]
    messages: list[str] = []
    if missing:
        messages.append(
            f"ESTRUCTURA INCOMPLETA: faltan {len(missing)} secciones reglamentarias ({', '.join(missing)})."
        )

    def p(metric: str, key: str, default: float = 0.0) -> float:
        return float(percentiles.get(metric, {}).get(key, default))

    if total_citations < p("total_citas_detectadas", "p25"):
        messages.append(
            f"SUSTENTO BIBLIOGRÁFICO POBRE: {total_citations} citas detectadas; el percentil 25 del "
            f"corpus tiene {int(p('total_citas_detectadas', 'p25'))} y la mediana "
            f"{int(p('total_citas_detectadas', 'p50'))}."
        )
    if not np.isnan(coherence_applicative) and coherence_applicative < p("coh_obj_apl_v2", "p25"):
        messages.append(
            f"BAJA COHERENCIA TÉCNICA ({coherence_applicative:.2f} vs. mediana "
            f"{p('coh_obj_apl_v2', 'p50'):.2f}): el marco aplicativo no responde con claridad a los objetivos."
        )
    if not np.isnan(coherence_conclusions) and coherence_conclusions < p("coh_obj_con_v2", "p25"):
        messages.append(
            f"CONCLUSIONES DESCONECTADAS ({coherence_conclusions:.2f} vs. mediana "
            f"{p('coh_obj_con_v2', 'p50'):.2f}): no se evidencia el cierre de los objetivos."
        )
    if pages < p("num_paginas", "p25"):
        messages.append(
            f"EXTENSIÓN INSUFICIENTE ({pages} páginas): por debajo del percentil 25 "
            f"({int(p('num_paginas', 'p25'))} páginas)."
        )
    if not np.isnan(burstiness) and burstiness < 10.5:
        messages.append(
            f"ALERTA ESTILOMÉTRICA (burstiness={burstiness:.2f}): redacción excesivamente uniforme; "
            f"el estándar humano del corpus ronda {p('burstiness_oraciones', 'p50'):.2f}."
        )
    return messages


def build_similarity_index(corpus: pd.DataFrame) -> dict:
    """Índice TF-IDF del corpus para comparar tesis nuevas sin depender de torch."""
    documents = []
    for _, row in corpus.iterrows():
        text = " ".join(
            part
            for part in (
                str(row.get("titulo", "")),
                str(row.get("texto_completo", ""))[2000:9000],
                str(row.get("conclusiones_texto", ""))[:2500],
            )
            if part
        )
        documents.append(text)
    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
        stop_words=list(STOPWORDS_ES),
        token_pattern=r"\b[a-záéíóúñ]{3,24}\b",
    )
    matrix = vectorizer.fit_transform(documents)
    return {"vectorizer": vectorizer, "matrix": matrix, "documents": documents}


def similar_documents(
    query: str,
    corpus: pd.DataFrame,
    index: dict,
    top_n: int = 3,
    exclude_pdf: str | None = None,
) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame(columns=["archivo_pdf", "titulo", "autores", "fecha", "tutores", "similitud", "indice_rigor_academico"])
    query_vector = index["vectorizer"].transform([query])
    similarities = cosine_similarity(query_vector, index["matrix"])[0]
    order = similarities.argsort()[::-1]
    rows = []
    for position in order:
        row = corpus.iloc[int(position)]
        if exclude_pdf and row["archivo_pdf"] == exclude_pdf:
            continue
        rows.append(
            {
                "archivo_pdf": row["archivo_pdf"],
                "titulo": row["titulo"],
                "autores": row["autores"],
                "fecha": row["fecha"],
                "tutores": row["tutores"],
                "similitud": float(similarities[position]),
                "indice_rigor_academico": float(row["indice_rigor_academico"]),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def percentile_table(corpus: pd.DataFrame) -> dict:
    table = {}
    for metric in NUMERIC_METRICS + LEXICAL_METRICS:
        if metric not in corpus:
            continue
        series = corpus[metric].dropna()
        if series.empty:
            continue
        table[metric] = {
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "mean": float(series.mean()),
        }
    return table


def cluster_terms(corpus: pd.DataFrame, top_n: int = 6) -> dict[int, list[str]]:
    """Términos dominantes por clúster temático para etiquetar la exploración."""
    if "cluster_tematico" not in corpus:
        return {}
    texts = (
        corpus["titulo"].fillna("")
        + " "
        + corpus["conclusiones_texto"].fillna("").astype(str).str[:3000]
    ).tolist()
    try:
        vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words=list(STOPWORDS_ES),
            token_pattern=r"\b[a-záéíóúñ]{4,24}\b",
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {}
    terms = vectorizer.get_feature_names_out()
    result: dict[int, list[str]] = {}
    for cluster in sorted(corpus["cluster_tematico"].dropna().unique()):
        mask = (corpus["cluster_tematico"] == cluster).to_numpy()
        if not mask.any():
            continue
        centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
        result[int(cluster)] = [terms[i] for i in centroid.argsort()[::-1][:top_n]]
    return result
