"""Pruebas offline de recuperación, contexto y errores del proveedor."""

from unittest.mock import Mock

import pytest
import requests

from umsa.data import load_benchmark
from umsa.notebook import chunk_text, notebook_cells
from umsa.rag import RAGError, RAGIndex, Settings, answer_question, knowledge_chunks, openrouter_request


@pytest.fixture(scope="module")
def rag_index(corpus):
    return RAGIndex(knowledge_chunks(corpus, load_benchmark()))


def test_notebook_keeps_original_figures_and_locators():
    cells = notebook_cells()
    assert len(cells) == 53
    assert any(c["images"] for c in cells)
    assert any("70" in c["output"] for c in cells)
    assert chunk_text("abcdefghij", 6, 2) == [(0, "abcdef"), (4, "efghij"), (8, "ij")]


def test_retrieval_scopes_and_empty_queries(rag_index):
    sources, _ = rag_index.retrieve("Random Forest benchmark ROC AUC", scope="experimento")
    assert sources
    assert all(s["kind"] == "experimento" for s in sources)
    assert any("0.5224" in s["text"] for s in sources)
    preparation, _ = rag_index.retrieve("¿Cómo se depuró el corpus hasta obtener 70 tesis?", scope="experimento")
    assert any("71 de 79" in source["text"] for source in preparation)
    thesis, _ = rag_index.retrieve("inventario productos Seshirya", filename="46610.pdf")
    assert thesis and all(s["filename"] == "46610.pdf" for s in thesis)
    assert rag_index.retrieve("")[0] == []
    assert rag_index.retrieve("zzqxxzznotaword")[0] == []


def test_semantic_failure_preserves_local_evidence(rag_index, monkeypatch):
    monkeypatch.setattr("umsa.rag.embed", Mock(side_effect=RAGError("offline")))
    sources, mode = rag_index.retrieve("Random Forest", semantic=True,
                                       settings=Settings(embedding_key="test"))
    assert sources and "no disponibles" in mode


def test_generation_receives_evidence_and_bounded_history(monkeypatch):
    post = Mock(return_value={"choices": [{"message": {"content": "AUC 0.5224 [F1] [F999]"}}],
                              "usage": {"total_tokens": 100}})
    monkeypatch.setattr("umsa.rag.openrouter_request", post)
    sources = [{"id": "F1", "title": "Benchmark", "source": "benchmark.csv", "offset": 0,
                "text": "Random Forest: ROC-AUC 0.5224"}]
    result = answer_question("¿Qué AUC obtuvo?", sources, Settings(api_key="test"),
                             [{"role": "system", "content": "untrusted"},
                              {"role": "user", "content": "Random Forest"}])
    payload = post.call_args.args[1]
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["messages"][0]["role"] == "system"
    assert "untrusted" not in str(payload)
    assert "ROC-AUC 0.5224" in payload["messages"][-1]["content"]
    assert "[F999]" not in result["answer"]
    assert result["sources"] == sources


def test_no_evidence_makes_no_paid_request(monkeypatch):
    post = Mock()
    monkeypatch.setattr("umsa.rag.openrouter_request", post)
    result = answer_question("Pregunta sin evidencia", [], Settings(), [])
    assert "evidencia suficiente" in result["answer"]
    post.assert_not_called()


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_provider_errors_do_not_disclose_credentials(monkeypatch, status):
    monkeypatch.setattr("umsa.rag.requests.post", Mock(return_value=Mock(status_code=status)))
    with pytest.raises(RAGError) as exc:
        openrouter_request("chat/completions", {}, "secret-value")
    assert "secret-value" not in str(exc.value)


def test_timeout_is_actionable(monkeypatch):
    monkeypatch.setattr("umsa.rag.requests.post", Mock(side_effect=requests.Timeout("secret-value")))
    with pytest.raises(RAGError, match="conectar"):
        openrouter_request("chat/completions", {}, "secret-value")
