"""Recorrido de la interfaz Streamlit sobre el corpus real."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from umsa.lessons import STAGES
from umsa.rag import Settings


def navigate(app, name):
    app.sidebar.radio[0].set_value(name).run()
    assert not app.exception, [e.message for e in app.exception]


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_pages_and_pdf_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("UMSA_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Del expediente a la evidencia."

    for page in ["Experimento", "Exploración", "Corpus", "Chat RAG", "Modelos", "Registro", "Auditoría PDF"]:
        navigate(app, page)
        assert not app.exception

    button(app, "Usar este PDF").click().run()
    assert not app.exception
    button(app, "Ejecutar auditoría").click().run(timeout=120)
    assert not app.exception
    assert "audit_result" in app.session_state
    result = app.session_state["audit_result"]
    assert 0 <= result["puntaje"] <= 100
    assert result["veredicto"]

    navigate(app, "Registro")
    assert not app.exception
    assert len(app.dataframe) >= 1


def test_chat_retrieval_generation_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("UMSA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("umsa.presentation.load_settings", lambda: Settings(api_key="test"))
    calls = []

    def generate(question, index, catalog, settings, history, **kwargs):
        result = catalog.query('SELECT "Modelo", "ROC-AUC" FROM benchmark WHERE "Modelo" = \'Random Forest\'')
        sources = [{"id": "F1", "kind": "consulta", "title": "Benchmark", "source": result["files"][0],
                    "sql": result["sql"], "rows": result["rows"], "text": str(result["rows"])}]
        calls.append((question, sources))
        return {"answer": "Random Forest obtuvo ROC-AUC 0.5224 [F1].", "sources": sources,
                "usage": {"total_tokens": 42}, "mode": "consulta de datasets"}

    monkeypatch.setattr("umsa.presentation.chat_with_data", generate)
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=120).run()
    navigate(app, "Chat RAG")
    app.chat_input[0].set_value("¿Qué resultados obtuvo Random Forest en el benchmark?").run(timeout=120)
    assert not app.exception
    assert len(calls) == 1 and calls[0][1]
    assert any("0.5224" in source["text"] for source in calls[0][1])
    assert len(app.session_state["rag_messages"]) == 2
    assert "0.5224" in app.session_state["rag_messages"][-1]["content"]
    button(app, "Nueva conversación").click().run()
    assert app.session_state["rag_messages"] == []
    # Búsqueda local: no debe llegar al cliente de generación.
    next(t for t in app.toggle if t.label == "Sólo buscar evidencia").set_value(True).run()
    app.chat_input[0].set_value("¿Qué modelos se compararon en el benchmark?").run(timeout=120)
    assert not app.exception
    assert len(calls) == 1
    assert app.session_state["rag_messages"][-1]["sources"]


def test_experiment_stages_and_interactive_chart_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("UMSA_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=120).run()
    navigate(app, "Experimento")
    for stage in STAGES:
        next(widget for widget in app.selectbox if widget.key == "experiment_stage").set_value(stage).run()
        assert not app.exception, [error.message for error in app.exception]
        assert app.code  # El código original es accesible en todas las etapas.
        if stage == "TF-IDF y similitud":
            next(w for w in app.selectbox if w.label == "Dataset para vectorizar").set_value("Final · 70 tesis").run()
            assert not app.exception
            assert app.metric[0].value == "70"
            next(w for w in app.multiselect if w.label.startswith("Documentos visibles")).set_value([]).run()
            assert not app.exception
        if stage == "Rigor y estilometría":
            slider = next(w for w in app.slider if w.label.startswith("Escenario:"))
            slider.set_value(0).run()
            assert not app.exception
