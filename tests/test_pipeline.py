
import numpy as np

from umsa.audit import audit_document, audit_to_json, extract_keywords, read_pdf
from umsa.data import (
    PDF_DIR,
    SECTION_LABELS,
    corpus_fingerprint,
    count_citations,
    similar_documents,
    structure_presence,
    stylometry,
)
from umsa.modeling import note_validation, train_rigor_model
from umsa.storage import AuditStore


def test_corpus_has_expected_shape_and_metrics(corpus):
    assert len(corpus) == 70
    assert corpus["indice_rigor_academico"].between(0, 100).all()
    assert set(SECTION_LABELS).issubset(corpus.columns)
    assert corpus["archivo_pdf"].is_unique
    assert corpus["pct_estructura"].between(0, 1).all()
    assert len(corpus_fingerprint(corpus)) == 64


def test_structure_and_citations():
    text = (
        "INTRODUCCIÓN\nPLANTEAMIENTO DEL PROBLEMA\nOBJETIVOS GENERAL Y ESPECÍFICOS\n"
        "MARCO TEÓRICO\nMARCO APLICATIVO\nPRUEBAS Y RESULTADOS\nCONCLUSIONES\nBIBLIOGRAFÍA\n"
        "Según Pérez (2020) y [1] el sistema cumple."
    )
    presence = structure_presence(text)
    assert all(presence.values())
    apa, ieee = count_citations(text)
    assert apa == 1
    assert ieee == 1


def test_stylometry_returns_finite_values():
    text = " ".join(["Esta oración tiene varias palabras para medir la variación del texto."] * 40)
    burstiness, mattr = stylometry(text * 3)
    assert np.isfinite(burstiness)
    assert 0 < mattr <= 1


def test_similar_documents_excludes_self(corpus, index):
    row = corpus.iloc[0]
    query = row["titulo"] + " " + row["texto_completo"][2000:9000]
    result = similar_documents(query, corpus, index, top_n=3, exclude_pdf=row["archivo_pdf"])
    assert len(result) == 3
    assert row["archivo_pdf"] not in set(result["archivo_pdf"])
    assert result["similitud"].between(0, 1).all()


def test_audit_real_pdf(corpus, index, percentiles):
    filename = "46610.pdf"
    text, pages = read_pdf(PDF_DIR / filename)
    assert pages > 0
    assert len(text) > 5000
    audit = audit_document(PDF_DIR / filename, filename, corpus, index, percentiles)
    assert 0 <= audit["puntaje"] <= 100
    assert audit["veredicto"]
    assert isinstance(audit["secciones"], dict)
    assert set(audit["secciones"]) == set(SECTION_LABELS)
    assert all(row["archivo_pdf"] != filename for row in audit["similares"])
    payload = audit_to_json(audit)
    assert "_texto" not in payload
    assert "diagnosticos" in payload


def test_keywords_are_extracted():
    query = extract_keywords("Sistema web para control de inventario y ventas con PHP y MySQL. " * 20)
    assert isinstance(query, str)
    assert len(query) > 0


def test_interpretable_model(corpus):
    model = train_rigor_model(corpus)
    assert 0.5 <= model.auc <= 1.0
    assert 0 <= model.accuracy <= 1
    assert len(model.coefficients) == 7
    validation = note_validation(corpus)
    assert validation["n"] == int(corpus["nota_numeral"].notna().sum())


def test_audit_store_roundtrip(tmp_path):
    store = AuditStore(tmp_path)
    ident = store.save_audit(
        {"archivo": "demo.pdf", "puntaje": 68.5, "veredicto": "ACEPTABLE", "diagnosticos": []}
    )
    records = store.audits()
    assert records[0]["id"] == ident
    assert store.audit(ident)["puntaje"] == 68.5
    store.delete_audit(ident)
    assert store.audits() == []
