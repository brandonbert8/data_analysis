"""Las visualizaciones deben reproducir los artefactos y conservar la procedencia."""

import numpy as np
import pandas as pd
import pytest

from umsa.data import load_benchmark, load_metadata
from umsa.figures import (
    benchmark_figure,
    benchmark_values,
    coverage_inventory,
    notebook_clusters,
    rigor_components,
    sentence_profile,
    term_contributions,
    tfidf_experiment,
    tutor_counts,
)
from umsa.notebook import notebook_cells


def test_all_five_original_figures_have_a_named_dashboard_destination():
    original = {str(cell["cell"]) for cell in notebook_cells() if cell["images"]}
    coverage = coverage_inventory()
    assert original == {"23", "24", "25", "28", "33"}
    assert original <= set(coverage["Celdas"])
    assert coverage.loc[coverage["Celdas"].isin(original), "Etapa en la app"].notna().all()


def test_original_tfidf_dimensions_and_duplicate_are_reproduced():
    result = tfidf_experiment("Inicial · 73 documentos")
    assert result["matrix"].shape == (73, 1500)
    assert len(result["pairs"]) == 73 * 72 // 2
    duplicate = result["pairs"].loc[(result["pairs"]["Documento A"] == "46514.pdf")
                                    & (result["pairs"]["Documento B"] == "36615.pdf")]
    assert duplicate["Similitud"].iloc[0] == pytest.approx(1)
    np.testing.assert_allclose(np.diag(result["similarity"]), 1)
    assert (result["pairs"]["Documento A"] != result["pairs"]["Documento B"]).all()
    contributions = term_contributions(result, 0, 1)
    assert contributions["Aporte al coseno"].sum() == pytest.approx(result["similarity"][0, 1])


def test_post_deduplication_data_is_not_labelled_as_historical_71_rows():
    result = tfidf_experiment("Final · 70 tesis")
    assert result["matrix"].shape[0] == 70
    assert "36615.pdf" not in result["frame"]["archivo_pdf"].tolist()
    assert len(result["pairs"]) == 70 * 69 // 2


def test_rigor_waterfall_reconstructs_every_stored_score(corpus):
    recalculated = rigor_components(corpus).sum(axis=1).round(2)
    np.testing.assert_allclose(recalculated, corpus["indice_rigor_academico"], atol=1e-9)


def test_stylometry_explanation_reproduces_saved_values(corpus):
    for row in corpus.iloc[[0, 12, 40]].itertuples():
        lengths, windows = sentence_profile(row.texto_completo)
        assert np.std(lengths) == pytest.approx(row.burstiness_oraciones, abs=5.1e-5)
        assert windows["Diversidad léxica"].mean() == pytest.approx(row.riqueza_lexica_mattr, abs=5.1e-5)


def test_teacher_modes_explain_notebook_and_application_discrepancy():
    metadata = load_metadata()
    notebook = tutor_counts(metadata, "Limpieza de sufijos · cuaderno")
    app = tutor_counts(metadata, "Por persona · app")
    assert notebook.iloc[0]["Proyectos"] == 16
    assert app.iloc[0]["Proyectos"] == 29


def test_benchmark_preserves_accuracy_dispersion_and_auc_reference():
    values = benchmark_values(load_benchmark())
    forest = values.loc[(values["Modelo"] == "Random Forest") & (values["Métrica"] == "Accuracy")].iloc[0]
    assert forest["Valor"] == pytest.approx(.5143)
    assert forest["Desviación"] == pytest.approx(.0833)
    accuracy = benchmark_figure(values, "Accuracy")
    auc = benchmark_figure(values, "ROC-AUC")
    assert accuracy.data[0].error_x.array is not None
    assert auc.data[0].error_x.array is None
    assert auc.layout.shapes[0].x0 == .5
    assert values.loc[values["Métrica"] != "Accuracy", "Desviación"].isna().all()


def test_cluster_curve_uses_saved_scores_and_groups(corpus):
    scores, groups = notebook_clusters()
    best = scores.loc[scores["Silhouette"].idxmax()]
    assert int(best["K"]) == 6
    assert best["Silhouette"] == pytest.approx(.1812)
    assert groups["Tesis"].sum() == len(corpus)
    pd.testing.assert_series_equal(groups.set_index("Clúster")["Tesis"].sort_index(),
                                   corpus["cluster_tematico"].value_counts().sort_index(), check_names=False,
                                   check_index_type=False, check_dtype=False)
