FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# models downloaded at build time (names must match config.py EMBEDDING_MODEL and RERANKER_MODEL)
USER user
RUN python -c "from huggingface_hub import snapshot_download as d; d('OrdalieTech/Solon-embeddings-large-0.1'); d('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
