"""Regresiones de preguntas cuantitativas y consultas sobre datos reales."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from umsa.analytics import DatasetCatalog, DatasetQueryError, teacher_rows
from umsa.rag import Settings, chat_with_data


@pytest.fixture(scope="module")
def catalog():
    return DatasetCatalog()


def test_teacher_ranking_uses_all_rows_and_distinguishes_populations(catalog):
    sql = "SELECT docente, COUNT(DISTINCT archivo_pdf) AS proyectos FROM {} " \
          "WHERE rol IN ('tutor','sin especificar') GROUP BY docente ORDER BY proyectos DESC, docente LIMIT 3"
    original = catalog.query(sql.format("docentes_original"))["rows"]
    final = catalog.query(sql.format("docentes_final"))["rows"]
    assert original == [
        {"docente": "Valdez Alvarado, Aldo Ramiro", "proyectos": 29},
        {"docente": "Cuevas Quiroz, Franz", "proyectos": 10},
        {"docente": "Rodríguez Ramírez, Grover Alex", "proyectos": 8},
    ]
    assert final[0]["proyectos"] == 25
    assert catalog.query("SELECT COUNT(*) AS n FROM metadatos")["rows"] == [{"n": 79}]
    assert catalog.query("SELECT COUNT(*) AS n FROM corpus")["rows"] == [{"n": 70}]


def test_teacher_normalization_handles_roles_coauthors_and_ambiguity():
    frame = pd.DataFrame([
        {"archivo_pdf": "a.pdf", "tutores": "Pérez, Ana, tutora; Gómez, Luis, asesor"},
        {"archivo_pdf": "b.pdf", "tutores": "Pérez, Ana María"},
        {"archivo_pdf": "c.pdf", "tutores": "Gómez, Luis Alberto"},
        {"archivo_pdf": "d.pdf", "tutores": "Gómez, Luis Antonio"},
    ])
    rows = teacher_rows(frame)
    assert rows.iloc[0]["docente"] == "Pérez, Ana María"
    assert rows.iloc[0]["nombre_original"] == "Pérez, Ana"
    assert rows.iloc[1]["docente"] == "Gómez, Luis"  # Dos extensiones posibles: no se fusionan.
    assert rows.iloc[1]["rol"] == "asesor"


def test_catalog_includes_intermediate_datasets_and_can_join(catalog):
    assert "ds_dataset_segmentado_v3" in catalog.tables
    result = catalog.query("SELECT m.archivo_pdf, m.titulo FROM metadatos m "
                           "JOIN ds_dataset_final_definitivo d USING (archivo_pdf) WHERE m.fecha=2025")
    assert result["rows"]
    assert len(result["files"]) == 2


def test_document_scope_is_applied_before_query(catalog):
    result = catalog.query("SELECT archivo_pdf FROM corpus", filename="46610.pdf")
    assert result["rows"] == [{"archivo_pdf": "46610.pdf"}]


@pytest.mark.parametrize("sql", [
    "DELETE FROM metadatos", "PRAGMA database_list", "SELECT load_extension('anything')",
    "SELECT * FROM sqlite_master", "SELECT 1; DROP TABLE metadatos", "SELECT randomblob(1000000)",
])
def test_sql_cannot_mutate_or_access_server_resources(catalog, sql):
    with pytest.raises(DatasetQueryError):
        catalog.query(sql)


def test_teacher_question_executes_sql_before_generating_answer(catalog, monkeypatch):
    sql = "SELECT docente, COUNT(DISTINCT archivo_pdf) AS proyectos FROM docentes_original " \
          "WHERE rol IN ('tutor','sin especificar') GROUP BY docente ORDER BY proyectos DESC LIMIT 3"
    response = Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": "consultar_datasets", "arguments": json.dumps({"sql": sql})}}]}}]},
        {"choices": [{"message": {"content": "Aldo Ramiro Valdez Alvarado: 29 proyectos [F1]."}}]},
    ])
    monkeypatch.setattr("umsa.rag.openrouter_request", response)
    index = Mock(side_effect=AssertionError("Un conteo no requiere recuperación textual"))
    result = chat_with_data("cuales son los docentes con mas proyectos", index, catalog, Settings(api_key="test"), [])
    assert result["sources"][0]["rows"][0]["proyectos"] == 29
    assert result["sources"][0]["sql"] == sql
    sent_result = response.call_args_list[1].args[1]["messages"][-1]
    assert sent_result["role"] == "tool"
    assert '"proyectos": 29' in sent_result["content"]
    index.assert_not_called()


def test_failed_sql_can_be_corrected_in_the_same_turn(catalog, monkeypatch):
    def tool(sql, call_id):
        return {"choices": [{"message": {"tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "consultar_datasets", "arguments": json.dumps({"sql": sql})}}]}}]}
    monkeypatch.setattr("umsa.rag.openrouter_request", Mock(side_effect=[
        tool("SELECT missing_column FROM metadatos", "bad"),
        tool("SELECT COUNT(*) AS n FROM metadatos", "good"),
        {"choices": [{"message": {"content": "Hay 79 registros [F1]."}}]},
    ]))
    result = chat_with_data("¿Cuántos registros hay?", None, catalog, Settings(api_key="test"), [])
    assert result["sources"][0]["rows"] == [{"n": 79}]


def test_benchmark_methodology_searches_experiment_not_individual_theses(catalog, monkeypatch):
    index = Mock()
    index.retrieve.return_value = ([{"title": "Benchmark", "source": "repositorioUMSA.ipynb · celda 48",
                                    "text": "Validación cruzada 5-fold", "kind": "experimento",
                                    "offset": 0, "score": 0.5}], "léxica TF-IDF")
    monkeypatch.setattr("umsa.rag.openrouter_request", Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": "buscar_documentos", "arguments": json.dumps({"pregunta": "evaluación de modelos",
                                                                    "ambito": "experimento"})}}]}}]},
        {"choices": [{"message": {"content": "Se usó validación cruzada 5-fold [F1]."}}]},
    ]))
    result = chat_with_data("¿Cómo se evaluaron los modelos?", lambda: index, catalog, Settings(api_key="test"), [])
    assert index.retrieve.call_args.args[1] == "experimento"
    assert result["sources"][0]["kind"] == "experimento"
