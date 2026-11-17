FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy SISMO_DATA_DIR=/app/runtime
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY . .
RUN useradd --create-home --uid 10001 sismolab && mkdir -p /app/runtime && chown -R sismolab:sismolab /app/runtime
USER sismolab
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]
CMD ["/app/.venv/bin/streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
