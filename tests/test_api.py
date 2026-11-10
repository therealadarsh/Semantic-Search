"""Fast unit tests for the HTTP layer.

These use a fake search engine, so they run in a second and need no model or
index. The real model is exercised by tests/test_search.py and by the Docker
smoke test in CI.
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from search_query import SearchResult


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((query, k))
        return [
            SearchResult(text=f"article {i}", category="World", score=0.1 * i) for i in range(k)
        ]


@pytest.fixture
def engine():
    fake = FakeEngine()
    app.state.engine = fake
    yield fake
    app.state.engine = None


@pytest.fixture
def client(engine):
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_structured_results(client, engine):
    response = client.get("/search-query", params={"query": "stock market"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "stock market"
    assert len(body["results"]) == 5
    assert body["results"][0] == {"text": "article 0", "category": "World", "score": 0.0}
    assert engine.calls == [("stock market", 5)]


def test_search_respects_k(client, engine):
    response = client.get("/search-query", params={"query": "football", "k": 2})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


@pytest.mark.parametrize(
    "params",
    [{}, {"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": 1000}],
)
def test_search_rejects_bad_input(client, params):
    assert client.get("/search-query", params=params).status_code == 422
