# RAG Engine UI

The frontend dashboard for the local retrieval-augmented generation engine, built with **React 19**, **Vite**, **Tailwind CSS**, **Lucide React**, and **React Router**.

---

## Views & Capabilities

1. **Ask (`/ask`):**
   - Interactive chat interface with real-time token-by-token SSE streaming.
   - Live pipeline stage indicators: **Retrieve** → **Grade** → **Generate**.
   - Model selector (explicit model requirement), confidence scoring (0–100%), and token generation throughput (`tokens/sec`).
   - Latency breakdown (`rerank_ms` and `generate_ms`).
   - Interactive source citation cards linking to the inline document viewer.
   - Query history drawer with individual and bulk clear capabilities.
2. **Ingest (`/ingest`):**
   - Multi-format file upload dropzone supporting `.csv`, `.json`, `.jsonl`, `.txt`, `.md`, and `.pdf`.
   - Chunking splitter selection (`recursive` vs `token`).
   - Live stage progress bar tracking parsing, splitting, batch embedding, and full-text GIN index construction.
   - Ingested files table with SHA-256 deduplication, unit/chunk counts, and document deletion.
3. **Document Viewer (`/documents/:fileId` & `DocumentModal`):**
   - Inspect addressable document units (spreadsheet rows, document lines, PDF pages) exactly as indexed by the pipeline.
   - Interactive table viewer for CSVs and formatted reader for prose documents.
   - Jump-to-unit locator to verify citation references in context.
4. **Benchmark (`/benchmark`):**
   - Evaluate model retrieval and question-answering accuracy across custom test suites.
   - Upload and manage custom benchmark CSV test files.
   - Interleaved round-robin execution with real-time metric progression.
   - **Retrieval Impact Check:** Side-by-side A/B comparison of grounded RAG generation vs direct raw LLM generation.
5. **System (`/system`):**
   - Real-time 1Hz SSE stream of host CPU (overall and per-core), RAM, Swap, and Disk I/O.
   - Dedicated NVIDIA GPU telemetry: GPU compute load, VRAM usage vs capacity, temperature (°C), and power draw (Watts).
6. **Settings (`/settings`):**
   - Chat model catalog with hardware requirements, one-click Ollama pull with progress/cancel, and disk deletion.
   - Cross-Encoder reranker catalog with Hugging Face Hub downloads and cache detection.
   - Pinned active embedding model status and dimension safety notes.

---

## Development & Build

### Running Locally

```bash
# Install dependencies
npm install

# Start Vite dev server (runs on :5173 with proxy to API on :8000)
npm run dev
```

### Production Build

```bash
# Build static assets to dist/
npm run build

# Preview production build locally
npm run preview
```

---

## Architecture & API Proxy

During development, Vite proxies all `/api` requests to the backend server configured via `VITE_API_PROXY_TARGET` (defaults to `http://127.0.0.1:8000` or `http://app:8000` in Docker).

Real-time telemetry and query token generation utilize standard **Server-Sent Events (SSE)** via `EventSource` and `fetch` streaming readers.
