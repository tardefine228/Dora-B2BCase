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
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Embeddings for vector search. true (default): a multilingual transformer is installed
# and downloaded during the build and a reranker is trained for it. If anything in that
# step fails (no internet for Hugging Face, ...) the build still succeeds and the service
# falls back to the offline spaCy vectors; /health/ready reports the embedder in use.
# README documents INSTALL_TRANSFORMER=false for a ~2 minute build without it.
ARG INSTALL_TRANSFORMER=true
ARG EMBEDDER_MODEL=intfloat/multilingual-e5-small
COPY api/requirements.txt /tmp/requirements.txt
COPY api/requirements-ml.txt /tmp/requirements-ml.txt
# pip's cache lives in a BuildKit cache mount: a re-run of any pip layer reinstalls
# from the local cache instead of re-downloading hundreds of megabytes.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.txt

RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/jobs \
    && chown app:app /app/jobs

# The heavy ML toolchain: torch (CPU) + sentence-transformers + the transformer
# weights (~1 GB of downloads). Deliberately ordered BEFORE any application code
# is copied and fed only by the requirement files and the download script, so
# editing app/ml/ui code cannot invalidate this layer — otherwise every rebuild
# pays ~10 minutes of downloads again. The Hugging Face cache is a cache mount
# and the weights are copied out of it, for the same reason.
COPY scripts/download_model.py ./scripts/download_model.py
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/huggingface \
    if [ "$INSTALL_TRANSFORMER" = "true" ]; then \
      ( python -c "import urllib.request as u; u.urlopen('https://huggingface.co/api/models/$EMBEDDER_MODEL', timeout=20)" \
        || { echo "WARNING: huggingface.co unreachable - check the antivirus/proxy web filter"; \
             echo "WARNING: whitelist huggingface.co and *.hf.co (model CDN), then rebuild"; \
             echo "WARNING: skipping the transformer step early, falling back to spaCy vectors"; exit 1; }; \
        python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" \
        || { echo "WARNING: torch install failed - download.pytorch.org blocked?"; exit 1; }; \
        python -m pip install -r /tmp/requirements-ml.txt \
        || { echo "WARNING: sentence-transformers install failed - pypi.org blocked?"; exit 1; }; \
        python scripts/download_model.py "$EMBEDDER_MODEL" /root/.cache/huggingface/embedder \
        || { echo "WARNING: model download failed - huggingface.co or the *.hf.co CDN blocked mid-download"; exit 1; }; \
        mkdir -p /opt/models \
        && cp -a /root/.cache/huggingface/embedder /opt/models/embedder \
      ) || echo "WARNING: transformer step failed - falling back to spaCy vectors"; \
    fi

# Application code, copied after the downloads: a code edit re-runs at most the
# offline training step below, never the toolchain/model layer above.
COPY --chown=app:app api/app ./app
COPY --chown=app:app ml ./ml
# Built UI (index.html + assets/); served by app.main at /ui/.
COPY --chown=app:app --from=ui-build /ui/dist ./frontend
COPY --chown=app:app data/data.csv data/data.json data/source_text.txt ./data/
COPY scripts ./scripts
# Labeled set + scorer: the transformer step keeps vector features only if they help.
# Everything `build_model.py --dense --gate` imports at run time: run_eval scores the
# gate, dense_gate pools the reliability table, quality_gate records the verdict.
# A missing module here fails the step instantly and is swallowed by `|| echo`, so
# the image would ship with the lexical reranker only (see the test that pins this).
COPY eval/run_eval.py eval/labeled_examples.csv eval/dense_gate.py eval/quality_gate.py ./eval/
RUN if [ "$INSTALL_TRANSFORMER" = "true" ]; then \
      ( DATA_DIR=/app/data EMBEDDER_BACKEND=sentence-transformers MODEL_NAME=/opt/models/embedder \
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

# --no-proxy-headers: uvicorn's own ProxyHeadersMiddleware trusts 127.0.0.1 by
# default and rewrites scope["client"] from X-Forwarded-For, which silently
# defeats the app's TRUSTED_PROXIES policy (and cannot be seen by tests). Trust
# for forwarded headers is decided in app/middleware/ratelimit.py only.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
