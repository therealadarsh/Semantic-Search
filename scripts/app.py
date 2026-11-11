"""HTTP API for semantic search.

Run locally:   uvicorn app:app --app-dir scripts --reload
Docs:          http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from pydantic import BaseModel

import settings
from search_query import SearchEngine


class Article(BaseModel):
    text: str
    category: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Article]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and index once at startup. Tests can pre-set
    # app.state.engine to a fake so they never touch the real model.
    if getattr(app.state, "engine", None) is None:
        app.state.engine = SearchEngine()
    yield


app = FastAPI(title="Semantic Search", version="1.0.0", lifespan=lifespan)


def get_engine(request: Request) -> SearchEngine:
    return request.app.state.engine


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "semantic-search", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    """Used by the Docker HEALTHCHECK, CI smoke test and hosting platform."""
    return {"status": "ok"}


@app.get("/search-query", response_model=SearchResponse)
def search(
    engine: Annotated[SearchEngine, Depends(get_engine)],
    query: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=50)] = settings.DEFAULT_TOP_K,
) -> SearchResponse:
    results = engine.search(query, k=k)
    return SearchResponse(query=query, results=[Article(**vars(r)) for r in results])
