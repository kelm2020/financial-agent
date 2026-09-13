.PHONY: setup up down migrate ingest mock run cli test test-rag embeddings-cache calibrate-rag eval-rag rerank-cache calibrate-rerank eval-rag-rerank eval-guardrails coverage lint format check

setup:
	uv sync

up:
	docker compose up --build --detach --wait

down:
	docker compose down

migrate:
	uv run python -m scripts.initialize_database

ingest: migrate
	uv run python -m scripts.ingest_knowledge

mock:
	uv run uvicorn mock_api.main:app --host 0.0.0.0 --port 8001 --reload

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

cli:
	uv run python -m app.cli --token "$$TOKEN"

# Unit suite only; needs neither network nor Postgres. tests/conftest.py ignores .env, clears
# provider keys and blocks real HTTP transports, so a local key can never be used here.
test:
	uv run pytest

# Level-A guardrail metrics (§10.1.7): dev for tuning, test held out. Numerator/denominator/95% bound.
eval-guardrails:
	uv run python -m scripts.evaluate_guardrails --split dev
	uv run python -m scripts.evaluate_guardrails --split test

# Includes the pgvector integration suite (needs `make up`); it uses its own database.
test-rag:
	RUN_POSTGRES_TESTS=1 uv run pytest tests/test_retriever.py tests/test_rag_scripts.py tests/test_rag_integration.py

# The only retrieval command that calls the embedding provider (needs OPENAI_API_KEY).
embeddings-cache:
	uv run python -m scripts.build_embedding_cache

# Reads the dev split only and prints the RAG_MIN_DENSE_SCORE to configure.
calibrate-rag:
	uv run python -m scripts.calibrate_retrieval

# Held-out test split, offline store and the real Postgres index.
eval-rag: ingest
	uv run python -m scripts.evaluate_retrieval --split test --store memory
	uv run python -m scripts.evaluate_retrieval --split test --store postgres

# Cross-encoder reranker (Cohere). Only rerank-cache calls the provider (needs COHERE_API_KEY);
# calibration (dev only) and evaluation then run offline from data/rerank_cache.json.
rerank-cache: ingest
	uv run python -m scripts.build_rerank_cache --postgres

calibrate-rerank:
	uv run python -m scripts.calibrate_retrieval --reranker

eval-rag-rerank: ingest
	uv run python -m scripts.evaluate_retrieval --split test --store memory --reranker
	uv run python -m scripts.evaluate_retrieval --split test --store postgres --reranker

# Coverage is only meaningful with the Postgres store exercised (needs `make up`).
coverage:
	RUN_POSTGRES_TESTS=1 uv run pytest --cov

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .

check: lint coverage
