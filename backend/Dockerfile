FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install uv.
RUN pip install --no-cache-dir uv

# Copy dependency metadata first.
# Docker caches this layer until dependencies change.
COPY pyproject.toml uv.lock ./

# Install project dependencies.
RUN uv sync --frozen --no-dev

# Copy application source.
COPY app ./app
COPY tests ./tests

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]