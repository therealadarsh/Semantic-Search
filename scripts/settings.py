"""Single source of truth for paths and model names.

Every script (fetching data, building the index, serving the API) reads its
configuration from here. That guarantees the model used to build the FAISS
index is the same model used to embed queries at search time.

Each value can be overridden with an environment variable, which is how the
Docker image and CI configure the app without code changes (12-factor style).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Base model from the Hugging Face Hub, used when no fine-tuned model exists.
BASE_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
_LOCAL_MODEL = ROOT / "models" / "semantic-model"

MODEL_PATH = os.getenv("MODEL_PATH", str(_LOCAL_MODEL) if _LOCAL_MODEL.is_dir() else BASE_MODEL)
DATASET_NAME = "ag_news"
DATASET_SPLIT = "train[:12000]"
DATASET_PATH = Path(os.getenv("DATASET_PATH", ROOT / "data" / "ag_news_train_12k"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "data" / "training_index.faiss"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
