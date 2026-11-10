"""Semantic search over the AG News index.

The model, dataset and index are loaded once when SearchEngine is created,
not on every request. Loading takes a few seconds; a search takes milliseconds.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import settings


@dataclass(frozen=True)
class SearchResult:
    text: str
    category: str
    score: float


class SearchEngine:
    def __init__(
        self,
        model_path: str = settings.MODEL_PATH,
        dataset_path: Path = settings.DATASET_PATH,
        index_path: Path = settings.INDEX_PATH,
    ) -> None:
        # Heavy imports live here so that importing this module (for example in
        # API unit tests that use a fake engine) does not load PyTorch.
        import datasets
        from sentence_transformers import SentenceTransformer

        if not Path(index_path).exists():
            raise FileNotFoundError(f"No search index at {index_path}. Build it with `make index`.")
        self._model = SentenceTransformer(model_path)
        self._dataset = datasets.load_from_disk(str(dataset_path))
        self._dataset.load_faiss_index("embeddings", str(index_path))
        self._labels = self._dataset.features["label"].names

    def search(self, query: str, k: int = settings.DEFAULT_TOP_K) -> list[SearchResult]:
        query_vector = self._model.encode(query, convert_to_numpy=True)
        scores, examples = self._dataset.get_nearest_examples("embeddings", query_vector, k=k)
        return [
            SearchResult(text=text, category=self._labels[label], score=float(score))
            for text, label, score in zip(examples["text"], examples["label"], scores, strict=True)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search news articles from the command line.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=settings.DEFAULT_TOP_K)
    args = parser.parse_args()
    for result in SearchEngine().search(args.query, k=args.k):
        print(f"[{result.score:.3f}] ({result.category}) {result.text}\n")


if __name__ == "__main__":
    main()
