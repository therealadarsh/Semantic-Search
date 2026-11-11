# The same commands CI runs, so "it passes locally" means "it passes in CI".
IMAGE ?= semantic-search:local
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: install lint format test index run docker-build docker-run smoke

install: ## Create .venv and install dev dependencies (CPU-only PyTorch)
	python3 -m venv .venv
	.venv/bin/pip install --index-url $(TORCH_INDEX) "$$(grep -E '^torch==' requirements.txt)"
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -v

index: ## Build data/training_index.faiss
	python scripts/embed_data.py

run: ## Run the API with auto-reload on http://localhost:8000/docs
	uvicorn app:app --app-dir scripts --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

smoke: docker-build
	./tests/smoke_test.sh $(IMAGE)
