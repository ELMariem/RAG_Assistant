FROM python:3.11-slim-bookworm

# --- OS-level deps ---
# curl/gnupg: needed to add Microsoft's package repo
# unixodbc-dev + msodbcsql17: required by pyodbc for the mssql+pyodbc:// connection in memory.py/auth.py
# build-essential: some ML wheels (docling deps, etc.) still compile bits from source
#
# Note: pinned to bookworm (Debian 12) on purpose, and the key is imported as a keyring file
# rather than with the old 'apt-key' command, which newer Debian releases removed entirely.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg build-essential unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,armhf,arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download and cache the ML models at build time (baked into this image layer),
# so containers start instantly instead of re-downloading from HuggingFace every
# fresh start. These must match config.EMBEDDING_MODEL and the reranker in rag.py --
# if you change either there, update this line too and rebuild.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# App code
COPY app.py auth.py config.py eval_metrics.py ingest.py llm_providers.py \
     main.py memory.py rag.py run_evaluation.py ./

# Dirs main.py/config.py expect to exist at runtime
# (data/documents, data/figures, chroma_db are also created lazily by app.py/config.py,
#  but 'static' is mounted unconditionally at import time in main.py and will crash startup
#  if missing)
RUN mkdir -p static data/documents data/figures chroma_db

EXPOSE 8000

# Runtime config expected via env (do not bake secrets into the image):
#   JWT_SECRET_KEY, SQL_SERVER_CONNECTION_STRING, GROQ_API_KEY (optional),
#   OLLAMA_HOST (e.g. http://ollama:11434 if Ollama runs in its own container)

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]