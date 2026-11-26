"""Búsquedas, fichas completas, páginas verificables y referencias de seguimiento."""

import json
from unittest.mock import Mock

import pytest

from umsa.analytics import DatasetCatalog, DatasetQueryError
from umsa.rag import Settings, chat_with_data, conversation_context
from umsa.research import ProjectResearch, pdf_pages


@pytest.fixture(scope="module")
def research():
    return ProjectResearch(DatasetCatalog())


def test_search_includes_projects_outside_modeling_corpus(research):
    result = research.search("evaluación formato tesis")
    assert any(row["archivo_pdf"] == "46521.pdf" for row in result["rows"])
    project = research.profile("46521.pdf")["project"]
    assert project["en_corpus_final"] is False
    assert project["indice_rigor_academico"] is None
    assert project["uri"].startswith("https://repositorio.umsa.bo/")


def test_project_search_handles_plural_filters_and_pagination(research):
    result = research.search("inventarios", autor="Yampara", anio=2025)
    assert [row["archivo_pdf"] for row in result["rows"]] == ["46610.pdf"]
    first = research.search(limit=4)
    second = research.search(limit=4, offset=first["next_offset"])
    assert first["total_matches"] == 79
    assert not {row["archivo_pdf"] for row in first["rows"]} & {row["archivo_pdf"] for row in second["rows"]}
    assert research.search(docente="Aldo", anio=2025)["rows"]


def test_profiles_preserve_people_roles_and_missing_metrics(research):
    profile = research.profile("46610.pdf")
    assert profile["project"]["autores"] == "Yampara Ichuta, Ruddy Carlos"
    assert profile["teachers"][0]["docente"] == "Felipez Andrade, Jhonny Roberto"
    assert profile["project"]["pdf_disponible"] is True
    assert profile["project"]["motivo_exclusion"] is None
    assert "duplicidad" in research.profile("36615.pdf")["project"]["motivo_exclusion"]
    assert "no documentado" in research.profile("46521.pdf")["project"]["motivo_exclusion"]


def test_pdf_evidence_cites_actual_physical_pages(research):
    result = research.read_pdf("46610.pdf", "objetivo general objetivos especificos")
    goal = next(source for source in result if source["page"] == 26)
    assert "Objetivo General" in goal["text"]
    path = research.catalog.directory / "PDFs" / "46610.pdf"
    pages = pdf_pages(str(path), path.stat().st_mtime_ns)
    assert goal["text"] == pages[25][goal["offset"]:goal["offset"] + 2400]
    assert "página física 26" in goal["source"]


def test_pdf_page_ranges_missing_files_and_document_scopes(research, tmp_path, monkeypatch):
    with pytest.raises(DatasetQueryError, match="rango"):
        research.read_pdf("46610.pdf", page=99999)
    monkeypatch.setattr(research.catalog, "directory", tmp_path)
    with pytest.raises(DatasetQueryError, match="no está disponible"):
        research.read_pdf("46610.pdf")
    with pytest.raises(DatasetQueryError, match="limitado"):
        research.profile("46610.pdf", selected_filename="46609.pdf")
    with pytest.raises(DatasetQueryError, match="no encontrado"):
        research.profile("../../.env")


def test_notebook_reading_is_paginated_and_preserves_source(research):
    first = research.read_notebook(48)
    assert "celda 48" in first["source"]
    assert len(first["text"]) <= 6500
    if first["next_offset"]:
        second = research.read_notebook(48, first["next_offset"])
        assert second["offset"] == 6500
        assert second["text"] != first["text"]


def test_app_documentation_comes_from_readme_not_theses(research):
    result = research.documentation("Docker ejecutar aplicación compose modelo chat")
    assert all(source["source"] == "README.md" for source in result)
    assert any("docker compose up --build" in source["text"] for source in result)
    assert "compose.yaml" in result[0]["project_files"]


def test_quantitative_functions_ignore_missing_values(research):
    row = research.catalog.query("SELECT median(indice_rigor_academico) AS mediana, "
                                 "stddev(indice_rigor_academico) AS desviacion FROM proyectos")["rows"][0]
    assert row["mediana"] == pytest.approx(67.955)
    assert row["desviacion"] == pytest.approx(7.82958357)


def test_followup_retains_project_identifiers_and_reads_the_correct_project(research, monkeypatch):
    history = [{"role": "assistant", "content": "Estos son los dos primeros proyectos.", "sources": [{
        "rows": [{"archivo_pdf": "46609.pdf", "titulo": "Microservicios"},
                 {"archivo_pdf": "46610.pdf", "titulo": "Seshirya"}]}]}]
    assert "46610.pdf" in conversation_context(history)
    api = Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "profile", "type": "function", "function": {
            "name": "consultar_proyecto", "arguments": json.dumps({"archivo_pdf": "46610.pdf"})}}]}}]},
        {"choices": [{"message": {"content": "El segundo proyecto es Seshirya, de Ruddy Carlos Yampara Ichuta [F1]."}}]},
    ])
    monkeypatch.setattr("umsa.rag.openrouter_request", api)
    result = chat_with_data("Cuéntame más del segundo", None, research.catalog, Settings(api_key="test"), history)
    system = api.call_args_list[0].args[1]["messages"][0]["content"]
    assert "entidades_en_orden" in system and "46610.pdf" in system
    assert result["sources"][0]["rows"][0]["archivo_pdf"] == "46610.pdf"


def test_document_retrieval_limits_sources_to_identified_projects(research, monkeypatch):
    index = Mock()
    index.retrieve.return_value = ([], "léxica TF-IDF")
    api = Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "search", "type": "function", "function": {
            "name": "buscar_documentos", "arguments": json.dumps({"pregunta": "conclusiones",
                            "ambito": "tesis", "archivos": ["46610.pdf"]})}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"id": "profile", "type": "function", "function": {
            "name": "consultar_proyecto", "arguments": json.dumps({"archivo_pdf": "46610.pdf"})}}]}}]},
        {"choices": [{"message": {"content": "La ficha contiene conclusiones del proyecto [F1]."}}]},
    ])
    monkeypatch.setattr("umsa.rag.openrouter_request", api)
    result = chat_with_data("Conclusiones de Seshirya", lambda: index, research.catalog, Settings(api_key="test"), [])
    assert index.retrieve.call_args.args[2] == "46610.pdf"
    assert result["sources"]


def test_pdf_search_tool_does_not_get_stuck_on_the_cover(research, monkeypatch):
    read = Mock(return_value=[{"title": "Objetivos", "source": "46610.pdf · página física 26",
                               "text": "Objetivo General: desarrollar un sistema de inventario", "page": 26,
                               "kind": "pagina", "offset": 0, "score": 1.0}])
    monkeypatch.setattr("umsa.rag.ProjectResearch.read_pdf", read)
    monkeypatch.setattr("umsa.rag.openrouter_request", Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "pdf", "type": "function", "function": {
            "name": "leer_pdf", "arguments": json.dumps({"archivo_pdf": "46610.pdf",
                                                         "pregunta": "objetivos", "pagina": 1})}}]}}]},
        {"choices": [{"message": {"content": "El objetivo es desarrollar un inventario [F1]."}}]},
    ]))
    result = chat_with_data("¿Cuáles son sus objetivos?", None, research.catalog, Settings(api_key="test"), [])
    assert read.call_args.args[2] is None
    assert result["sources"][0]["page"] == 26


def test_app_documentation_includes_active_model_without_credentials(research, monkeypatch):
    api = Mock(side_effect=[
        {"choices": [{"message": {"tool_calls": [{"id": "guide", "type": "function", "function": {
            "name": "consultar_documentacion", "arguments": json.dumps({"pregunta": "modelo del chat"})}}]}}]},
        {"choices": [{"message": {"content": "El chat usa GPT-4o mini [F1]."}}]},
    ])
    monkeypatch.setattr("umsa.rag.openrouter_request", api)
    result = chat_with_data("¿Qué modelo usa el chat?", None, research.catalog, Settings(api_key="secret"), [])
    source = result["sources"][0]
    assert source["runtime_config"]["chat_model"] == "openai/gpt-4o-mini"
    assert "secret" not in json.dumps(source)
