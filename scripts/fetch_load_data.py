"""Download the AG News subset used by the search index.

The repo already ships this data in data/ag_news_train_12k, so this script is
only needed to re-create it (for example, after changing DATASET_SPLIT).
"""

from datasets import load_dataset

import settings


def main() -> None:
    if settings.DATASET_PATH.exists():
        print(f"Dataset already exists at {settings.DATASET_PATH}, skipping download.")
        return
    print(f"Downloading {settings.DATASET_NAME} ({settings.DATASET_SPLIT})...")
    dataset = load_dataset(settings.DATASET_NAME, split=settings.DATASET_SPLIT)
    dataset.save_to_disk(str(settings.DATASET_PATH))
    print(f"Saved to {settings.DATASET_PATH}")


if __name__ == "__main__":
    main()
