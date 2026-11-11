"""Integration test against the real model and index.

Skipped when the index has not been built (run `make index` first). In CI the
same behaviour is checked end to end against the Docker image instead.
"""

import pytest

import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    if not settings.INDEX_PATH.exists():
        pytest.skip("search index not built; run `make index`")
    from search_query import SearchEngine

    return SearchEngine()


def test_finds_relevant_articles(engine):
    results = engine.search("stock market and oil prices", k=5)
    assert len(results) == 5
    assert sum(r.category == "Business" for r in results) >= 3


def test_results_are_sorted_by_distance(engine):
    scores = [r.score for r in engine.search("olympic games", k=5)]
    assert scores == sorted(scores)
