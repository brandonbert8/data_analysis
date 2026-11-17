from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from sismo.data import REQUIRED, clean_catalog, csv_bytes, fingerprint, read_csv
from sismo.features import CATS, F25, FeatureConfig, build_features, temporal_folds, temporal_partitions
from sismo.modeling import (
    TemporalStack,
    TrainConfig,
    block_intervals,
    make_pipeline,
    metrics,
    pick_threshold,
    score_new,
    train_experiment,
)
from sismo.storage import Store


def events(days, magnitudes=None):
    return pd.DataFrame(
        {
            "id": [f"test-{i}" for i in range(len(days))],
            "time": [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d) for d in days],
            "latitude": -21.2,
            "longitude": -68.2,
            "depth": 100,
            "mag": magnitudes if magnitudes is not None else 4.0,
        }
    )


def test_csv_schema_encoding_and_size():
    raw = read_csv(b"time;latitude;longitude;depth;mag\n2025-01-01;-21;-68;100;4.5\n")
    assert len(clean_catalog(raw).frame) == 1
    with pytest.raises(ValueError, match="obligatorias"):
        clean_catalog(pd.DataFrame({"magnitud": [4.5]}))
    with pytest.raises(ValueError, match="vacío"):
        read_csv(b"")


def test_cleaning_rejects_invalid_core_and_audits_optional():
    raw = events([0, 1, 2, 3])
    raw.loc[1, "latitude"] = 95
    raw.loc[2, "mag"] = np.inf
    raw["time"] = raw["time"].astype(str)
    raw.loc[3, "time"] = "not-a-date"
    raw["magError"] = "error"
    result = clean_catalog(raw)
    assert result.audit["rechazados"] == 3
    assert result.frame.magError.isna().all()
    assert result.rejected.motivo_rechazo.str.len().gt(0).all()
    assert str(result.frame.time.dtype) == "datetime64[ns, UTC]"


def test_revisions_keep_latest_update_and_missing_ids_are_stable():
    raw = events([0, 0])
    raw["id"] = "same"
    raw["updated"] = ["2025-01-02Z".replace("Z", "T00:00:00Z"), "2025-01-03T00:00:00Z"]
    raw.loc[1, "mag"] = 5.3
    result = clean_catalog(raw.iloc[::-1])
    assert result.audit["duplicados"] == 1
    assert result.frame.mag.iloc[0] == 5.3
    missing = events([0]).drop(columns="id")
    assert clean_catalog(missing).frame.id.iloc[0] == clean_catalog(missing).frame.id.iloc[0]


def test_future_boundaries_simultaneity_geography_and_censoring():
    raw = events([0, 0, 7, 8, 20], [4, 6, 5, 6, 4])
    raw.loc[3, "latitude"] = -10  # otra celda
    f = build_features(clean_catalog(raw).frame, FeatureConfig())
    assert f.iloc[0].target == 1  # exactamente siete días: incluido
    assert f.iloc[2].target == 0  # el fuerte posterior está en otra celda
    assert pd.isna(f.iloc[-1].target)
    assert f.iloc[0].cell_n7 == 0 and f.iloc[1].cell_n7 == 0
    isolated = build_features(clean_catalog(events([0, 0, 20], [4, 6, 4])).frame, FeatureConfig())
    assert isolated.iloc[0].target == 0  # simultáneo no es futuro


def test_appending_future_never_changes_historical_features(catalog):
    early = catalog.iloc[:800].copy()
    before = build_features(early, FeatureConfig()).set_index("id")
    after = build_features(catalog, FeatureConfig()).set_index("id").loc[before.index]
    pd.testing.assert_frame_equal(before[F25 + CATS], after[F25 + CATS])


def test_partition_and_cv_embargo_are_in_days(frame):
    parts = temporal_partitions(frame, 7)
    train, val, test = [parts[k] for k in ["Entrenamiento", "Validación", "Prueba"]]
    assert train.label_end.max() < val.time.min()
    assert val.label_end.max() < test.time.min()
    assert train.warmup_complete.all() and test.observable.all()
    assert len(set(train.id) & set(test.id)) == 0
    for a, b in temporal_folds(train, 7):
        assert train.iloc[a].label_end.max() < train.iloc[b].time.min()
        assert not set(train.iloc[a].time) & set(train.iloc[b].time)


def test_simultaneous_events_stay_in_same_partition(catalog):
    copied = catalog.copy()
    copied.loc[copied.index[1::2], "time"] = copied.time.iloc[::2].to_numpy()[: len(copied.iloc[1::2])]
    parts = temporal_partitions(build_features(copied, FeatureConfig()), 7)
    dates = [set(parts[k].time) for k in ["Entrenamiento", "Validación", "Prueba"]]
    assert not dates[0] & dates[1] and not dates[1] & dates[2]


def test_missing_optional_features_and_unknown_categories_are_supported(catalog):
    simple = clean_catalog(catalog[REQUIRED].copy()).frame
    f = build_features(simple, FeatureConfig())
    parts = temporal_partitions(f, 7)
    with threadpool_limits(1):
        p = make_pipeline("Boosting", F25).fit(
            parts["Entrenamiento"], parts["Entrenamiento"].target.astype(int)
        )
        test = parts["Prueba"].copy()
        test["country"] = "Nunca visto"
        assert np.isfinite(p.predict_proba(test)).all()


def test_metrics_single_class_are_honest():
    m = metrics([0, 0, 0], [0.1, 0.8, 0.2], 0.5)
    assert m["AP"] is None and m["ROC_AUC"] is None and m["Recall"] is None
    assert m["Tasa_señales"] == pytest.approx(1 / 3)
    assert m["Brier"] is not None
    assert pick_threshold([0, 0], [0.1, 0.2])[0] == 0.5
    assert metrics([0, 1], [0.1, 0.8], 0.5, probability=False)["Brier"] is None


def test_model_selection_and_threshold_do_not_see_test_labels(catalog, experiment, monkeypatch):
    import sismo.modeling as modeling

    original = modeling.temporal_partitions

    def reversed_test(*args, **kwargs):
        parts = original(*args, **kwargs)
        parts["Prueba"] = parts["Prueba"].copy()
        parts["Prueba"]["target"] = 1 - parts["Prueba"].target
        return parts

    monkeypatch.setattr(modeling, "temporal_partitions", reversed_test)
    changed = train_experiment(
        catalog, config=TrainConfig(feature_sets=("F25",), models=("Regresión logística",))
    )
    assert changed["metadata"]["selected"] == experiment["metadata"]["selected"]
    pd.testing.assert_series_equal(changed["results"].CV_AP, experiment["results"].CV_AP)
    pd.testing.assert_series_equal(changed["results"].umbral, experiment["results"].umbral)


def test_temporal_stack_can_fit_full_training_but_is_not_selected_from_incomplete_cv(catalog):
    config = TrainConfig(feature_sets=("F25",), models=("Stacking temporal",))
    b = train_experiment(catalog, config=config)
    row = b["results"].query("modelo == 'Stacking temporal'").iloc[0]
    assert not row.seleccionable
    assert b["metadata"]["selected"] != row.candidato
    assert np.isfinite(b["predictions"][row.candidato]["test"]).all()
    assert clone(TemporalStack()).horizon == 7


def test_external_uses_history_and_waits_for_future(experiment):
    end = experiment["catalog"].time.max()
    raw = events([0, 1, 10], [4, 6, 4])
    raw["time"] = [end + pd.Timedelta(days=d) for d in [1, 2, 11]]
    clean = clean_catalog(raw).frame
    rows, result, drift = score_new(experiment, clean)
    assert rows.iloc[0].target == 1
    assert pd.isna(rows.iloc[-1].target)
    assert result["N"] == 2
    assert len(drift) == 6
    assert rows.iloc[0].events_last_30d > 1
    single, result_single, _ = score_new(experiment, clean.iloc[:1])
    assert result_single is None
    assert single.estado.str.startswith("Pendiente").all()


def test_external_validation_is_persisted_without_retraining(experiment, tmp_path):
    new = events([0])
    new["time"] = experiment["catalog"].time.max() + pd.Timedelta(days=1)
    new = clean_catalog(new).frame
    external = score_new(experiment, new)
    store = Store(tmp_path)
    ident = store.save_validation(experiment, new, *external)
    assert store.validations()[0]["pendientes"] == 1
    assert store.validation_report(ident)["predictions"][0]["target"] is None
    assert store.validations()[0]["experiment_id"] == experiment["metadata"]["id"]


def test_external_rejects_historical_ids_or_dates(experiment):
    with pytest.raises(ValueError, match="IDs"):
        score_new(experiment, experiment["catalog"].iloc[-5:])
    raw = experiment["catalog"].iloc[-5:].copy()
    raw["id"] = "new-" + raw.id
    with pytest.raises(ValueError, match="posteriores"):
        score_new(experiment, raw)


def test_block_intervals_do_not_invent_ci_for_single_class():
    times = pd.Series(pd.date_range("2025-01-01", periods=30, tz="UTC"))
    ci = block_intervals(np.zeros(30), np.ones(30) * 0.3, times, repetitions=20)
    assert ci["AP"] is None and ci["ROC_AUC"] is None and ci["réplicas_válidas"] == 0


def test_catalog_round_trip_preserves_fingerprint(catalog, tmp_path):
    store = Store(tmp_path)
    store.save_catalog(catalog)
    restored = store.load_catalog()
    assert fingerprint(catalog) == fingerprint(restored)
    pd.testing.assert_frame_equal(catalog, restored)


def test_artifact_integrity_and_model_reload(experiment, tmp_path):
    store = Store(tmp_path)
    store.save_experiment(experiment)
    ident = experiment["metadata"]["id"]
    restored = store.load_experiment(ident)
    assert restored["metadata"] == experiment["metadata"]
    assert store.experiments()[0]["feature_config"] == asdict(FeatureConfig())
    key = restored["metadata"]["selected"]
    test = restored["partitions"]["Prueba"]
    with threadpool_limits(1):
        np.testing.assert_allclose(
            restored["models"][key].predict_proba(test), experiment["models"][key].predict_proba(test)
        )
    path = tmp_path / "models" / (ident + ".joblib")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        store.load_experiment(ident)


def test_export_neutralizes_spreadsheet_formulas():
    exported = csv_bytes(pd.DataFrame({"place": ["=HYPERLINK('x')", "+formula", "Bolivia"]})).decode(
        "utf-8-sig"
    )
    assert "'=HYPERLINK" in exported and "'+formula" in exported
