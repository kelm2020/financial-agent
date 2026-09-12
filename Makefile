.PHONY: setup up down migrate mock test coverage lint format check

setup:
	uv sync

up:
	docker compose up --build --detach --wait

down:
	docker compose down

migrate:
	uv run python -m scripts.initialize_database

mock:
	uv run uvicorn mock_api.main:app --host 0.0.0.0 --port 8001 --reload

test:
	uv run pytest

coverage:
	uv run pytest --cov

lint:
	uv run ruff check .
	uv run mypy

format:
	uv run ruff format .

check: lint coverage
