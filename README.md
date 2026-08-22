# rag-trainer

A local, private retrieval-augmented generation pipeline that runs entirely on one machine. Documents (.csv, .json, .jsonl, .txt, .md, .pdf) are parsed, chunked, and embedded into Postgres/pgvector; user queries retrieve the most relevant passages via hybrid dense + lexical search and cross-encoder reranking, and a local Ollama model generates grounded answers with deterministic citations. No data leaves the host.

Built and calibrated against a personal workstation, ensuring every default operates under real hardware constraints rather than assumed abundance.

---

## System Requirements

**Recommended:** A workstation-class GPU with 8GB+ VRAM, 16GB system RAM, and an SSD -- runs the full bundled chat model lineup with headroom for concurrent Benchmark workers and a GPU-accelerated reranker.

**Minimum:** 6GB VRAM (or CPU-only, with materially slower generation), 8GB system RAM, and Docker support -- runs the smallest bundled models (`llama3.2:1b`, `gemma2:2b`) with the cross-encoder reranker on CPU.

---

## Features

- **100% Local & Air-Gapped:** Zero external API calls, zero per-token cost, zero data egress.
- **Multi-Format Ingestion:** Ingest `.csv`, `.json`, `.jsonl`, `.txt`, `.md`, and `.pdf` files with semantic recursive or token chunking.
- **Smart Ingest Deduplication:** Batch embedding with duplicate text reuse, divide-and-conquer fault tolerance, and SHA-256 upload deduplication.
- **Hybrid Retrieval & Reranking:** Combines dense pgvector cosine similarity with PostgreSQL `tsvector` full-text search via Reciprocal Rank Fusion (RRF), followed by MS MARCO Cross-Encoder reranking.
- **Deterministic Citations & Document Viewer:** Source citations are computed in code and link directly to specific addressable units (rows, lines, pages) inside the integrated document viewer.
- **Token-by-Token SSE Streaming:** Real-time token streaming with live pipeline stage indicators (Retrieve → Grade → Generate) and generation telemetry (tokens/sec).
- **Comprehensive Benchmark Bench:** Run multi-suite evaluations with live progression, upload custom test datasets, and perform side-by-side Retrieval Impact Checks (A/B testing RAG vs raw LLM).
- **System Telemetry Dashboard:** 1Hz real-time host CPU, RAM, Swap, Disk, and NVIDIA GPU (utilization, VRAM, temp, wattage) monitoring.
- **Model Management:** Download, monitor, and delete local chat and reranker models directly from the UI.

---

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
docker compose exec app python -m src.ingestion.pipeline data/uploads/your_file.csv
```

Then open the dashboard at **http://localhost:5173**. The API is available at **http://localhost:8000**.

> [!IMPORTANT]
> You must download a chat model before your first query — there is deliberately no default model fallback. Use the **Settings** view in the dashboard, or run `ollama pull llama3.2:3b`.

---

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload        # API on :8000
cd ui && npm install && npm run dev     # dashboard on :5173
```

Postgres with the `pgvector` extension and Ollama both still need to be reachable at the URLs configured in `.env`.

---

## Tests

```bash
pytest                                    # unit tests only
RAG_INTEGRATION=1 pytest -m integration   # also hits live Postgres + Ollama
```

---

## Project Layout

```
src/
  api/            FastAPI routers (ask, ingest, documents, benchmark, metrics, models, jobs) & schemas
  benchmark/      Benchmark test dataset management & custom file inspection
  db/             Pooled SQLAlchemy engine cache
  jobs/           Background job runner (ingest, benchmark, model pulls with cooperative cancel)
  rag/            Query execution: LangGraph workflow, nodes, prompts, citations, history, model catalog
  ingestion/      Write pipeline: units parser, multi-format loaders, splitters, pipeline, file provenance
  vectorstore/    pgvector store, lexical full-text search, hybrid RRF fusion, cross-encoder rerank
  observability/  Span tracing, structured JSON logging with ring buffer, host & GPU metrics
ui/               React 19 + Vite + Tailwind CSS dashboard
tests/            Unit tests, integration tests, and benchmark evaluation harness
```

---

## Documentation

- [PRODUCT_EXPLAINER.md](PRODUCT_EXPLAINER.md) - Product capabilities, user workflows, feature breakdown, and product design rationale.
- [TECH_EXPLAINER.md](TECH_EXPLAINER.md) - In-depth technical explainer on architecture, mathematical mechanisms, retrieval algorithms, and engineering decisions.
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture, query/ingest flows, components, configuration lifetimes, and codebase structure.
- [ROADMAP.md](ROADMAP.md) - Completed milestones, refactor backlog, future ideas, and baseline evaluation metrics.
