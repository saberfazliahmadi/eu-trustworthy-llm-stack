FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=mock

WORKDIR /app

# CPU-only PyTorch keeps the image small; the offline stack needs no GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[live,dev]"

COPY data ./data
COPY tests ./tests

RUN useradd --create-home appuser && mkdir -p /app/reports/figures && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "trust_llm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
