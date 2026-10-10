FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

ARG INSTALL_ML=false
COPY api/requirements.txt /tmp/requirements.txt
COPY api/requirements-ml.txt /tmp/requirements-ml.txt
RUN if [ "$INSTALL_ML" = "true" ]; then \
        python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" && \
        python -m pip install --no-cache-dir -r /tmp/requirements-ml.txt; \
    else \
        python -m pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/jobs \
    && chown app:app /app/jobs
COPY --chown=app:app api/app ./app
COPY --chown=app:app ml ./ml
COPY --chown=app:app frontend ./frontend
COPY --chown=app:app data/data.csv data/data.json data/source_text.txt ./data/

ENV DATA_DIR=/app/data \
    CLASSIFIER_BACKEND=catalog \
    ML_ENABLED=false \
    QDRANT_URL=http://qdrant:6333

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
