FROM python:3.11.9-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Torch arrives as a transitive dependency of sentence-transformers, and pip
# would otherwise resolve the CUDA build -- ~2.5GB of kernels for a card the
# reranker never touches. Installing the CPU wheel first satisfies the
# requirement so the CUDA one is never fetched.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the cross-encoder into the image rather than downloading it on first
# query. Keeps `docker compose up` self-contained and offline, and keeps the
# first query's latency honest instead of hiding a ~90MB fetch in it.
ENV HF_HOME=/opt/hf
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('$RERANK_MODEL', device='cpu')"

COPY . .

# --host 0.0.0.0 rather than src/api/app.py's 127.0.0.1 default: that default
# is right for a bare-metal single-user run, but inside a container the
# loopback is the container's own, unreachable through the "8000:8000" port
# mapping in docker-compose.yml.
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
