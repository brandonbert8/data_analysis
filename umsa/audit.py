"""Auditoría de un PDF nuevo contra el corpus de referencia UMSA."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

from umsa.data import (
    APPLICATIVE_PATTERN,
    CONCLUSION_PATTERN,
    OBJECTIVE_PATTERN,
    SECTION_LABELS,
    calibrate_coherence,
    count_citations,
    diagnostics,
    extract_section,
    lexical_coherence,
    rigor_score,
    similar_documents,
    structure_presence,
    stylometry,
    verdict,
    weighted_structure,
)


def read_pdf(source: str | Path | bytes | BytesIO) -> tuple[str, int]:
    """Extrae texto y número de páginas de una ruta o de bytes subidos."""
    if isinstance(source, (bytes, bytearray)):
        reader = PdfReader(BytesIO(source))
    elif isinstance(source, BytesIO):
        reader = PdfReader(source)
    else:
        reader = PdfReader(str(source))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - una página corrupta no debe abortar el informe
            pages.append("")
    return "\n".join(pages), len(reader.pages)


def audit_document(
    source: str | Path | bytes | BytesIO,
    filename: str,
    corpus: pd.DataFrame,
    index: dict,
    percentiles: dict,
) -> dict:
    """Calcula el informe completo de un documento sin depender de Colab."""
    text, pages = read_pdf(source)
    presence = structure_presence(text)
    missing = [SECTION_LABELS[key] for key, present in presence.items() if not present]

    citations_apa, citations_ieee = count_citations(text)
    total_citations = citations_apa + citations_ieee

    objective = extract_section(text, OBJECTIVE_PATTERN, 4000)
    applicative = extract_section(text, APPLICATIVE_PATTERN, 6000)
    conclusions = extract_section(text, CONCLUSION_PATTERN, 4000)

    coherence_applicative_lexical = lexical_coherence(objective, applicative)
    coherence_conclusions_lexical = lexical_coherence(objective, conclusions)
    coherence_applicative = calibrate_coherence(
        coherence_applicative_lexical, corpus.get("coh_lex_apl", pd.Series(dtype=float)), corpus["coh_obj_apl_v2"]
    )
    coherence_conclusions = calibrate_coherence(
        coherence_conclusions_lexical, corpus.get("coh_lex_con", pd.Series(dtype=float)), corpus["coh_obj_con_v2"]
    )
    burstiness, mattr = stylometry(text)

    score = rigor_score(
        presence,
        coherence_applicative,
        coherence_conclusions,
        total_citations,
        pages,
    )
    label = verdict(score, missing)
    findings = diagnostics(
        presence,
        total_citations,
        coherence_applicative,
        coherence_conclusions,
        pages,
        burstiness,
        percentiles,
    )

    query_text = " ".join(
        part for part in (str(objective)[:3000], str(conclusions)[:2000], text[1500:6500]) if part
    )
    similar = similar_documents(query_text, corpus, index, top_n=3, exclude_pdf=filename)

    return {
        "archivo": filename,
        "paginas": pages,
        "caracteres": len(text),
        "palabras": len(text.split()),
        "secciones": presence,
        "faltantes": missing,
        "pct_estructura": round(sum(presence.values()) / len(presence), 4),
        "pct_estructura_ponderado": round(weighted_structure(presence), 4),
        "citas_apa": citations_apa,
        "citas_ieee": citations_ieee,
        "total_citas": total_citations,
        "coh_aplicativo": coherence_applicative,
        "coh_conclusiones": coherence_conclusions,
        "coh_aplicativo_lexico": coherence_applicative_lexical,
        "coh_conclusiones_lexico": coherence_conclusions_lexical,
        "burstiness": burstiness,
        "riqueza_lexica": mattr,
        "puntaje": score,
        "veredicto": label,
        "diagnosticos": findings,
        "similares": similar.to_dict(orient="records"),
        "fragmento_objetivos": objective[:700],
        "fragmento_aplicativo": applicative[:700],
        "fragmento_conclusiones": conclusions[:700],
        "_texto": text[:40000],
    }


def audit_verdict_band(score: float) -> str:
    if score >= 72:
        return "Alto"
    if score >= 60:
        return "Medio"
    return "Bajo"


def extract_keywords(text: str, top_n: int = 4) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer

    from umsa.data import STOPWORDS_ES

    try:
        vectorizer = TfidfVectorizer(
            max_features=40,
            stop_words=list(STOPWORDS_ES),
            token_pattern=r"\b[a-záéíóúñ]{4,20}\b",
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform([text[250:15000]])
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        words = vectorizer.get_feature_names_out()
        return " ".join(words[i] for i in scores.argsort()[::-1][:top_n])
    except ValueError:
        return "software engineering information system"


def openalex_search(text: str, top_n: int = 3, timeout: int = 10) -> tuple[str, list[dict]]:
    """Consulta OpenAlex para contrastar el documento con literatura abierta."""
    query = extract_keywords(text)
    params = {"search": query, "per-page": top_n, "sort": "relevance_score:desc"}
    headers = {"User-Agent": "UMSA-Auditor-Academico/1.0 (mailto:investigacion.informatica@umsa.bo)"}
    try:
        response = requests.get(
            "https://api.openalex.org/works", params=params, headers=headers, timeout=timeout
        )
        if response.status_code != 200:
            return query, []
        papers = []
        for item in response.json().get("results", []):
            inverted = item.get("abstract_inverted_index")
            abstract = ""
            if inverted:
                positions = sorted((position, word) for word, places in inverted.items() for position in places)
                abstract = " ".join(word for _, word in positions)[:280] + "..."
            papers.append(
                {
                    "titulo": item.get("title", "Sin título"),
                    "anio": item.get("publication_year", "N/A"),
                    "citas": item.get("cited_by_count", 0),
                    "doi": item.get("doi") or item.get("id", "Sin enlace"),
                    "resumen": abstract,
                }
            )
        return query, papers
    except (requests.RequestException, ValueError):
        return query, []


def audit_to_json(audit: dict) -> dict:
    """Serializa un informe a tipos JSON simples."""
    payload = dict(audit)
    payload.pop("fragmento_objetivos", None)
    payload.pop("fragmento_aplicativo", None)
    payload.pop("fragmento_conclusiones", None)
    payload.pop("_texto", None)
    for key in (
        "coh_aplicativo",
        "coh_conclusiones",
        "coh_aplicativo_lexico",
        "coh_conclusiones_lexico",
        "burstiness",
        "riqueza_lexica",
    ):
        value = payload.get(key)
        if isinstance(value, float) and np.isnan(value):
            payload[key] = None
    return payload


def corpus_audit_rows(corpus: pd.DataFrame) -> pd.DataFrame:
    """Resumen tabular del corpus para la vista de exploración."""
    frame = corpus[
        [
            "archivo_pdf",
            "titulo",
            "autores",
            "fecha",
            "tutores",
            "mencion",
            "modalidad",
            "num_paginas",
            "total_citas_detectadas",
            "coh_obj_apl_v2",
            "coh_obj_con_v2",
            "indice_rigor_academico",
            "cluster_tematico",
            "nota_numeral",
        ]
    ].copy()
    frame["anomalia"] = corpus["anomalia"].to_numpy()
    return frame


def score_distribution(corpus: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "indice": corpus["indice_rigor_academico"],
            "veredicto": pd.cut(
                corpus["indice_rigor_academico"],
                bins=[-np.inf, 60, 72, np.inf],
                labels=["Bajo", "Medio", "Alto"],
                right=False,
            ),
        }
    )


def resolve_corpus_pdf(filename: str, pdf_dir: Path) -> Path:
    path = pdf_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el PDF {filename} en {pdf_dir}.")
    return path


def clean_snippet(value: str, limit: int = 400) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]
