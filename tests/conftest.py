from pathlib import Path

import pytest

from umsa.data import build_similarity_index, load_bundle, load_corpus, percentile_table


@pytest.fixture(scope="session")
def corpus():
    return load_corpus()


@pytest.fixture(scope="session")
def index(corpus):
    return build_similarity_index(corpus)


@pytest.fixture(scope="session")
def percentiles(corpus):
    return percentile_table(corpus)


@pytest.fixture(scope="session")
def bundle():
    return load_bundle()


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).parents[1]
