from pathlib import Path

import pytest

from sismo.data import clean_catalog, read_csv
from sismo.features import FeatureConfig, build_features
from sismo.modeling import TrainConfig, train_experiment


@pytest.fixture(scope="session")
def catalog():
    return clean_catalog(
        read_csv((Path(__file__).parents[1] / "datasets" / "query (2).csv").read_bytes())
    ).frame


@pytest.fixture(scope="session")
def frame(catalog):
    return build_features(catalog, FeatureConfig())


@pytest.fixture(scope="session")
def experiment(catalog):
    return train_experiment(
        catalog, config=TrainConfig(feature_sets=("F25",), models=("Regresión logística",))
    )
