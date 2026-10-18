# Stage 1: build the React/Vite demo UI into static files (base path /ui/).
# Only the compiled dist/ goes into the final image; sources and node_modules stay here.
FROM node:22-slim AS ui-build
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: API image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Embeddings for vector search. true (default): a multilingual transformer is installed
# and downloaded during the build and a reranker is trained for it. If anything in that
# step fails (no internet for Hugging Face, ...) the build still succeeds and the service
# falls back to the offline spaCy vectors; /health/ready reports the embedder in use.
ARG INSTALL_TRANSFORMER=true
ARG EMBEDDER_MODEL=intfloat/multilingual-e5-small
COPY api/requirements.txt /tmp/requirements.txt
COPY api/requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/jobs \
    && chown app:app /app/jobs
COPY --chown=app:app api/app ./app
COPY --chown=app:app ml ./ml
# Built UI (index.html + assets/); served by app.main at /ui/.
COPY --chown=app:app --from=ui-build /ui/dist ./frontend
COPY --chown=app:app data/data.csv data/data.json data/source_text.txt ./data/

# Transformer step (best effort, see INSTALL_TRANSFORMER above). Runs as root before
# USER app: installs torch (CPU) + sentence-transformers, bakes the model into
# /opt/models/embedder and trains the reranker that uses that model's vector features.
COPY scripts ./scripts
# Labeled set + scorer: the transformer step keeps vector features only if they help.
COPY eval/run_eval.py eval/labeled_examples.csv ./eval/
RUN if [ "$INSTALL_TRANSFORMER" = "true" ]; then \
      ( python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" \
        && python -m pip install --no-cache-dir -r /tmp/requirements-ml.txt \
        && python scripts/download_model.py "$EMBEDDER_MODEL" /opt/models/embedder \
        && DATA_DIR=/app/data EMBEDDER_BACKEND=sentence-transformers MODEL_NAME=/opt/models/embedder \
           MODEL_LOCAL_ONLY=true python scripts/build_model.py --dense --gate --models-dir /app/ml/artifacts \
      ) || echo "WARNING: transformer step failed - falling back to spaCy vectors"; \
    fi
RUN chown -R app:app /app/ml /opt/models 2>/dev/null || true

ENV DATA_DIR=/app/data \
    CLASSIFIER_BACKEND=catalog \
    VECTOR_SEARCH=true \
    EMBEDDER_BACKEND=auto \
    MODEL_NAME=/opt/models/embedder \
    MODEL_LOCAL_ONLY=true \
    QDRANT_URL=http://qdrant:6333

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
