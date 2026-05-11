# --- Stage 1: build deps ---
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ---
COPY serving/ ./serving/

EXPOSE 8000
CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
