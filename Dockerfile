# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /app

# Dependencies first (cached layer), production dependencies only.
COPY pyproject.toml uv.lock .python-version ./
# All provider SDK extras; narrow with e.g. --extra groq to slim the image.
RUN uv sync --frozen --no-dev --no-install-project --all-extras

COPY alembic.ini ./
COPY migrations ./migrations
COPY prompts ./prompts
COPY decisions ./decisions
COPY app ./app

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
