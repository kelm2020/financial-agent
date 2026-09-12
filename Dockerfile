FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1

WORKDIR /workspace
COPY pyproject.toml uv.lock* README.md ./
COPY app app
COPY config config
COPY migrations migrations
COPY mock_api mock_api
COPY scripts scripts
COPY alembic.ini ./

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "uvicorn", "mock_api.main:app", "--host", "0.0.0.0", "--port", "8001"]

