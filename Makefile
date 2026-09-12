.PHONY: setup up down migrate mock test test-f0 coverage lint format check

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

test-f0:
	uv run pytest tests/test_tools_contract.py

coverage:
	uv run pytest --cov

lint:
	uv run ruff check .
	uv run mypy

format:
	uv run ruff format .

check: lint coverage

