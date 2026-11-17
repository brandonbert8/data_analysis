"""Recorrido sobre el catálogo real mediante el runtime oficial de Streamlit."""

from streamlit.testing.v1 import AppTest
from pathlib import Path


def navigate(app, name):
    app.sidebar.radio[0].set_value(name).run()
    assert not app.exception, [e.message for e in app.exception]


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_training_evaluation_manual_event_and_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("SISMO_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=90).run()
    assert not app.exception
    navigate(app, "Preprocesamiento")
    navigate(app, "Modelo")
    button(app, "Entrenar y evaluar").click().run(timeout=90)
    assert not app.exception
    assert "bundle" in app.session_state
    navigate(app, "Evaluación")
    assert len(app.metric) == 5
    app.slider[0].set_value(0.25).run()  # exploración, no modifica el umbral congelado
    assert not app.exception
    navigate(app, "Registro")
    button(app, "Abrir experimento congelado").click().run()
    assert not app.exception
    navigate(app, "Nuevos datos")
    import datetime

    date = next(x for x in app.date_input if x.label == "Fecha UTC")
    date.set_value(datetime.date(2026, 8, 20))
    button(app, "Preparar evento").click().run()
    assert not app.exception
    button(app, "Evaluar con modelo congelado").click().run()
    assert not app.exception
    assert app.session_state["external"][1] is None
    button(app, "Incorporar al catálogo y guardar").click().run()
    assert not app.exception
    assert len(app.session_state.raw) == 1051
    navigate(app, "Resumen")


def test_csv_upload_external_validation_with_real_heldout_events(tmp_path, monkeypatch, catalog):
    monkeypatch.setenv("SISMO_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=90).run()
    history, future = catalog.iloc[:850], catalog.iloc[850:]
    app.sidebar.file_uploader[0].upload("historico.csv", history.to_csv(index=False).encode(), "text/csv").run()
    button(app, "Utilizar archivo").click().run()
    assert not app.exception
    assert len(app.session_state.raw) == 850
    navigate(app, "Modelo")
    next(w for w in app.multiselect if w.label == "Familias a comparar").set_value(["Regresión logística"])
    next(w for w in app.multiselect if w.label == "Conjuntos de variables").set_value(["F25"])
    button(app, "Entrenar y evaluar").click().run(timeout=90)
    assert not app.exception
    navigate(app, "Nuevos datos")
    next(w for w in app.file_uploader if w.label == "Nuevos eventos").upload(
        "posteriores.csv", future.to_csv(index=False).encode(), "text/csv"
    ).run()
    button(app, "Evaluar con modelo congelado").click().run()
    assert not app.exception
    rows, metrics, _ = app.session_state["external"]
    assert len(rows) == 200
    assert metrics["N"] > 0
    assert rows.target.isna().sum() > 0
    navigate(app, "Registro")
    assert not app.exception
