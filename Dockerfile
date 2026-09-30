FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Abhängigkeiten zuerst (Docker-Layer-Cache)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/reports.json ./data/reports.json
RUN uv sync --frozen --no-dev

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "annual_report_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
