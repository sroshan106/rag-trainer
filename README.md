# rag-trainer

A retrieval-augmented generation pipeline that runs entirely on one machine. Documents are chunked and embedded into Postgres/pgvector; a query retrieves the relevant chunks and a local Ollama model answers from them. No data leaves the host.

Built and tuned against a 4GB GTX 1050, so every default here is chosen under real VRAM pressure rather than assumed abundance.

## Prerequisites

The compose file requests GPU access for Ollama (`driver: nvidia` device reservation). Docker Engine does not ship GPU support — install it on the host first, or containers fail with `could not select device driver "nvidia" with capabilities: [[gpu]]`:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Requires an NVIDIA driver already installed on the host (check with `nvidia-smi`). No GPU, or don't want to bother? Drop the `deploy:` GPU block from the ollama service in `docker-compose.yml` to run on CPU instead.

## Quick start

```bash
cp .env.example .env
docker compose up -d          # postgres+pgvector, ollama, api, ui
docker compose exec app python -m src.ingestion.pipeline data/uploads/your.csv
```

Then open the dashboard at http://localhost:5173. The API is at http://localhost:8000.

You must download a chat model before your first query — there is deliberately no default model. Use the **Settings** view in the dashboard, or `ollama pull llama3.2:3b`.

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload        # API on :8000
cd ui && npm install && npm run dev     # dashboard on :5173
```

Postgres with the pgvector extension and Ollama both still need to be reachable at the URLs in `.env`.

## Tests

```bash
pytest                    # unit tests only
RAG_INTEGRATION=1 pytest -m integration   # also hits live Postgres + Ollama
```

## Layout

```
src/
  api/          FastAPI routers and schemas — HTTP mapping only
  jobs/         background job runner (ingest, benchmark, model pull)
  rag/          the query path: graph, nodes, prompts, citations, history
  ingestion/    the write path: loaders, splitter, pipeline, file provenance
  vectorstore/  pgvector, lexical search, hybrid fusion, cross-encoder rerank
  observability/ tracing spans, JSON logging, host and GPU metrics
ui/             React + Vite dashboard
tests/          unit, integration, and the benchmark harness
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture, query/ingest flows, components, configuration lifetimes, and where new code goes.
- [ROADMAP.md](ROADMAP.md) - Future features, UI backlog, refactoring tasks, and historical benchmark baselines.
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions and RTK guidelines.
