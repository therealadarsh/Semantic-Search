"""Embed every article and build the FAISS index used by the search API.

The index is a build artifact: it is generated in the Docker build (see the
`index` stage of the Dockerfile) or locally with `make index`, never committed.
"""

import argparse
from pathlib import Path

import datasets
from sentence_transformers import SentenceTransformer

import settings


def build_index(dataset_path: Path, index_path: Path, model_path: str) -> None:
    dataset = datasets.load_from_disk(str(dataset_path))
    model = SentenceTransformer(model_path)
    print(f"Embedding {len(dataset)} articles with {model_path}...")
    # Encoding in batches is much faster than one row at a time.
    embeddings = model.encode(
        dataset["text"], batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    dataset.add_faiss_index_from_external_arrays(embeddings, index_name="embeddings")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_faiss_index("embeddings", str(index_path))
    print(f"Saved index to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS search index.")
    parser.add_argument("--dataset", type=Path, default=settings.DATASET_PATH)
    parser.add_argument("--index", type=Path, default=settings.INDEX_PATH)
    parser.add_argument("--model", default=settings.MODEL_PATH)
    args = parser.parse_args()
    build_index(args.dataset, args.index, args.model)


if __name__ == "__main__":
    main()
