# RAG System Implementation Plan (LangChain + LangGraph)

## Overview

Build retrieval-augmented generation pipeline. User query retrieves relevant document chunks from vector store, chunks + query feed LLM, LLM generates grounded answer. LangGraph orchestrates control flow (retrieve → grade → generate → maybe retry), LangChain handles loaders/splitters/embeddings/vector store integrations.

**Runs fully local, fully containerized — no external LLM API, no host-installed dependencies beyond Docker + NVIDIA driver.** LLM + embeddings served by Ollama, running as a Docker service with GPU passthrough (confirmed: NVIDIA GTX 1050, 4GB VRAM, driver 580.173.02, CUDA 13.0). `docker compose up` brings up Postgres+pgvector, Ollama (GPU), model pull, and app — one command, portable across any machine with Docker + nvidia-container-toolkit.

**Target architecture:**

```
Query → [Retrieve Node] → [Grade/Filter Node] → [Generate Node] → Answer
              ↑                    │
              └──── (retry if docs weak) ────┘
```

---

## Phase 1: Environment & Dependencies

- [x] Create virtual environment (`python -m venv .venv`)
- [x] Install core packages:
  ```
  langchain
  langgraph
  langchain-community
  langchain-ollama        # local LLM + embeddings
  langchain-postgres      # pgvector integration
  psycopg[binary]
  pgvector
  tiktoken                # token counting
  python-dotenv
  pypdf
  ```
- [x] Set env vars in `.env`: `DATABASE_URL` (Postgres connection string), `OLLAMA_BASE_URL` (default `http://localhost:11434` for bare venv use; `http://ollama:11434` when run via Docker)
- [x] Install `nvidia-container-toolkit` on host (one-time, enables GPU passthrough into Docker containers):
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- [x] Verify GPU detected on host: `nvidia-smi` shows driver + GPU (confirmed working — GTX 1050, 4GB, CUDA 13.0)
- [x] Verify GPU passthrough into container: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` shows GPU inside container
- [x] Project structure:
  ```
  rag/
  ├── src/
  │   ├── config.py             # shared env-flag parsing
  │   ├── ingestion/
  │   │   ├── loaders.py
  │   │   ├── splitter.py
  │   │   └── pipeline.py       # loaders -> splitter -> vectorstore
  │   ├── vectorstore/
  │   │   └── store.py
  │   ├── rag/
  │   │   ├── graph.py          # state schema, wiring, CLI entrypoint
  │   │   ├── nodes.py
  │   │   ├── prompts.py
  │   │   └── citations.py      # code-side source attribution
  │   └── observability/
  │       └── tracing.py        # node spans, latency, structured logs
  ├── data/
  │   └── documents.csv
  ├── tests/
  │   ├── ingestion/  vectorstore/  rag/  observability/
  │   ├── integration/           # live db + ollama, opt-in
  │   └── benchmark/
  │       ├── data/*.csv         # labeled question sets
  │       ├── run_benchmark.py
  │       └── run_latency.py
  ├── .env
  ├── .env.example
  ├── Dockerfile
  ├── docker-compose.yml
  ├── .dockerignore
  ├── pytest.ini
  ├── requirements.txt
  ├── plan.md
  ├── ui_plan.md                # dashboard/UI plan (separate document)
  └── explainer.md
  ```

  Modules are grouped by domain rather than the flat `src/*.py` sketched above —
  ingestion, vectorstore, rag, and observability each own their files.
- [x] Verify installs: `python -c "import langgraph, langchain; print('ok')"`

**Done marker:** environment activates, imports succeed, `.env` loads (only used for bare-venv/non-Docker runs).

### Docker (fully containerized — primary run path)

- [x] `Dockerfile` — Python 3.11-slim app image, installs `requirements.txt`, runs `src/graph.py`
- [x] `docker-compose.yml` — services:
  - `ollama` — `ollama/ollama:latest` image, GPU reservation (`driver: nvidia, capabilities: [gpu]`), persists models via named volume, healthcheck via `ollama list`
  - `ollama-pull` — one-shot init container, runs after `ollama` healthy, pulls `llama3.2:3b` + `nomic-embed-text` into the shared volume, exits
  - `db` — `pgvector/pgvector:pg16` image, exposes 5432, persists via named volume, healthcheck via `pg_isready`
  - `app` — builds from `Dockerfile`, waits on `db` healthy + `ollama` healthy + `ollama-pull` completed, `DATABASE_URL` points at `db` service, `OLLAMA_BASE_URL` points at `ollama` service (`http://ollama:11434`) — all container-network hostnames, zero host dependency beyond Docker + GPU driver
- [x] Requires `nvidia-container-toolkit` on host (one-time setup, see above) for the `ollama` service to see the GPU
- [x] `.dockerignore` — excludes `.venv`, `.git`, `.env`, `__pycache__`
- [x] Run: `docker compose up --build` — single command, no host Ollama install, no host Postgres install
- [x] First run: `ollama-pull` downloads ~2-4GB of model weights into the `ollama_data` volume (one-time; cached on subsequent runs)
- [x] Verify: `docker compose exec db psql -U rag -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"` (image ships extension, just needs enabling per DB)
- [x] Verify GPU used inside `ollama` container: `docker compose exec ollama nvidia-smi`
- [ ] Confirm app container connects to db + ollama services, ingest/query works end-to-end

**Done marker:** `docker compose up` on a fresh machine (Docker + NVIDIA driver + nvidia-container-toolkit installed) brings up Postgres+pgvector, Ollama with GPU, models pulled, and app — zero manual installs beyond those three prerequisites, zero external API calls. This is the portability target.

---

## Phase 2: Document Loading

- [x] Pick loader(s) matching source data type:
  - `PyPDFLoader` — PDFs
  - `WebBaseLoader` — web pages
  - `TextLoader` — plain text
  - `DirectoryLoader` — batch load folder
- [x] Implement `load_documents(path) -> list[Document]` in `src/loaders.py`
- [x] Handle load failures gracefully (skip corrupt files, log which failed)
- [x] Sanity check: print doc count, sample first doc's `page_content` and `metadata`

**Gotcha:** preserve `metadata` (source filename, page number) — needed later for citations.

---

## Phase 3: Text Splitting / Chunking

- [x] Use `RecursiveCharacterTextSplitter` (default, splits on paragraph/sentence boundaries)
- [x] Config:
  - `chunk_size=1000` (tokens or chars — decide, tiktoken-based counter recommended)
  - `chunk_overlap=100–200` (10–20% of chunk size)
- [x] Implement `split_documents(docs) -> list[Document]` in `src/splitter.py`
- [x] Validate: no chunk exceeds embedding model's max input tokens
- [x] Spot-check 5–10 chunks manually — do they read as coherent units, or cut mid-sentence?

**Gotcha:** too small → loses context across chunk boundary. Too large → dilutes relevance signal, wastes LLM context window. Iterate empirically against real queries.

---

## Phase 4: Embeddings & Vector Store

- [x] Embedding model: **`nomic-embed-text` via Ollama** (local, no API cost, 768 dim)
- [x] Vector store: **PostgreSQL + pgvector** (self-hosted or managed e.g. Supabase/RDS)
- [x] Ensure Postgres has `pgvector` extension available (`CREATE EXTENSION IF NOT EXISTS vector;`)
- [x] Implement `src/vectorstore.py`:
  ```python
  import os
  from langchain_ollama import OllamaEmbeddings
  from langchain_postgres import PGVector

  COLLECTION_NAME = "rag_chunks"
  OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

  def build_vectorstore(chunks, connection=None):
      connection = connection or os.environ["DATABASE_URL"]
      embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
      return PGVector.from_documents(
          documents=chunks,
          embedding=embeddings,
          collection_name=COLLECTION_NAME,
          connection=connection,
      )

  def load_vectorstore(connection=None):
      connection = connection or os.environ["DATABASE_URL"]
      embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
      return PGVector(
          embeddings=embeddings,
          collection_name=COLLECTION_NAME,
          connection=connection,
      )
  ```
- [x] `DATABASE_URL` in `.env`: `postgresql+psycopg://user:password@localhost:5432/rag_db`
- [x] Ingest all chunks (writes rows to Postgres, index built via pgvector)
- [x] Test retrieval manually: `vectorstore.similarity_search("test query", k=5)` — inspect results for relevance
- [ ] (production) tune pgvector index — `ivfflat` or `hnsw` — once row count grows past a few thousand

**Gotcha:** embedding model used at ingest time MUST match model used at query time. Mismatched dims/model = broken retrieval, often silent (no error, just bad results).

---

## Phase 5: LangGraph Workflow

- [x] Define state schema (`src/graph.py`):
  ```python
  from typing_extensions import TypedDict
  from langchain_core.documents import Document

  class RAGState(TypedDict):
      query: str
      retrieved_docs: list[Document]
      graded_docs: list[Document]
      answer: str
      sources: list[str]
  ```
- [x] Implement nodes (`src/nodes.py`):
  - `retrieve_node(state)` — calls retriever, populates `retrieved_docs`
  - `grade_node(state)` — drops blank chunks and anything under the relevance cutoffs
  - `generate_node(state)` — builds prompt from graded docs, calls LLM, returns `answer`
- [x] Wire graph:
  ```python
  from langgraph.graph import StateGraph, END

  workflow = StateGraph(RAGState)
  workflow.add_node("retrieve", retrieve_node)
  workflow.add_node("grade", grade_node)
  workflow.add_node("generate", generate_node)

  workflow.set_entry_point("retrieve")
  workflow.add_edge("retrieve", "grade")
  workflow.add_edge("grade", "generate")
  workflow.add_edge("generate", END)

  graph = workflow.compile()
  ```
- [x] Test invocation: `graph.invoke({"query": "..."})`
- [x] Add tracing/logging (`src/observability/tracing.py` — local only, no LangSmith)

**Tracing:** one structured JSON span per node, carrying its duration plus the fields that
node exposes. Off by default; enable with `RAG_TRACE=true`. The same spans are what
`run_latency.py` aggregates, so there is no second timing path to keep in sync.

```
$ RAG_TRACE=true python -m src.rag.graph "What are Bullet Kin?"
{"span": "retrieve", "duration_ms": 196.7, "k": 5, "scores": [0.8008, 0.6986, 0.6716, 0.4844, 0.4477], "sources": [...]}
{"span": "grade", "duration_ms": 0.0, "cutoff": 0.6406, "bound": "ratio", "kept": 3, "dropped": 2}
{"span": "generate", "duration_ms": 3752.1, "refused": false, "model": "llama3.2:3b", "prompt_chars": 11839, "docs": 3, "prompt_eval_count": 2664, "eval_count": 14, "eval_duration": 654864000, "tokens_per_sec": 21.4}
{"span": "ask", "duration_ms": 3954.7}
```

The `grade` span records which bound set the cutoff — the absolute floor or the ratio against
the best hit — which is what explains a refusal. Token counts come from Ollama's own
`eval_count`/`eval_duration`, so tokens/sec is measured rather than estimated.

**Design note:** the conditional retry loop was implemented, then removed. Retrieval is
deterministic and pgvector returns hits sorted by descending similarity, so re-running the
same query — at any `k` — can never surface a chunk that clears the grader's cutoff when the
top results did not. The loop was unreachable-in-effect code. Reinstate it only alongside
query rewriting (Phase 7), which changes the query and therefore the result set.

**Grading:** `grade_node` filters on relevance score, not just blank chunks. Two cutoffs,
both env-tunable:
- `RAG_RELEVANCE_FLOOR` (default `0.6`) — absolute floor, lets an off-topic query refuse
  instead of citing the five least-bad chunks in the collection.
- `RAG_RELEVANCE_RATIO` (default `0.8`) — relative to the best hit, drops filler that clears
  the floor but is weak next to the top result.

Measured on the current collection: on-topic hits score 0.67–0.80, off-topic noise 0.44–0.48,
and a pure-gibberish query tops out at 0.56. Re-tune both if the corpus or embedding model
changes.

---

## Phase 6: Prompt & Generation

- [x] Design prompt template (`src/prompts.py`):
  ```python
  RAG_PROMPT = """Answer the question using ONLY the context below.
  If the context doesn't contain the answer, say so — don't guess.

  Context:
  {context}

  Question: {question}

  Answer:"""
  ```
- [x] Format retrieved docs into context string, include source metadata for citation
- [x] LLM: **`llama3.2:3b` via Ollama** (fits fully in 4GB VRAM, fast GPU inference on GTX 1050). `qwen2.5:7b` as fallback if quality insufficient — tight on 4GB, may partial CPU-offload
  ```python
  from langchain_ollama import ChatOllama
  llm = ChatOllama(model="llama3.2:3b", base_url=OLLAMA_BASE_URL)
  ```
- [x] Test generation on 10+ representative queries, check for:
  - Hallucination (answer not grounded in context)
  - Missed context (answer ignores relevant retrieved doc)
  - Citation accuracy — **known gap:** `llama3.2:3b` does not reliably emit `(source: ...)`
    tags even with explicit prompt instruction + few-shot example + `temperature=0`.
    Grounding and refusal work correctly; inline citation does not at this model size.
    `qwen2.5:7b` fallback not yet tried (requires pulling a new ~5GB image — deferred,
    needs explicit go-ahead before pulling).

**Done marker:** answers are grounded, refuse gracefully when context insufficient. Citation
is not reliable with the current model — treat as open item, not blocking MVP.

---

## Phase 7: Optional Enhancements (post-MVP)

- [ ] **Re-ranking** — Cohere rerank or cross-encoder model on top-k results before generation (improves precision when k is large)
- [ ] **Hybrid search** — combine BM25 keyword search + semantic search (catches exact-match terms embeddings miss)
- [ ] **Multi-turn memory** — add conversation history to state, rewrite follow-up queries with context
- [ ] **Query rewriting/expansion** — LLM rephrases vague queries before retrieval
- [ ] **Streaming responses** — stream `generate_node` output token-by-token
- [ ] **Web UI** — Streamlit (fast prototype) or FastAPI + frontend (production)

---

## Phase 8: Testing & Evaluation

- [x] Unit tests: loader, splitter, retriever (mock vector store) — 50 tests, `pytest`
- [x] Integration test: full graph invoke on known query → assert answer contains expected fact
- [x] Retrieval eval: labeled question sets, recall@k measured
- [x] Answer quality eval: keyword-overlap scoring + correct-refusal rate
- [x] Load/latency test: end-to-end and per-node percentiles

### Test layout

| Suite | Command | Needs services |
|---|---|---|
| Unit | `pytest` | no — vectorstore and LLM are faked |
| Integration | `RAG_INTEGRATION=1 pytest -m integration` | yes |
| Benchmark | `python -m tests.benchmark.run_benchmark` | yes |
| Latency | `python -m tests.benchmark.run_latency [repeats]` | yes |

Integration tests are excluded from the default run via `pytest.ini`, so `pytest` stays fast
and hermetic. Coverage includes the two behaviours that regressed before: that a known query
cites *only* relevant sources, and that an off-topic query refuses instead of guessing.

### Measured baseline

Benchmark, 120 labeled questions, `llama3.2:3b` + `nomic-embed-text`, `k=5`, floor 0.6 /
ratio 0.8:

| Question set | n | recall@5 | mean answer overlap | pass rate (overlap ≥ 0.3) |
|---|---|---|---|---|
| single-passage | 40 | 0.80 | 0.16 | 0.20 |
| multi-passage | 40 | 0.80 | 0.20 | 0.25 |
| no-answer | 40 | — | — | correct refusal 0.62 |

Latency, 12 queries (mixed hits and refusals), single-threaded:

| node | mean | p50 | p95 | max |
|---|---|---|---|---|
| retrieve | 88.9ms | 80.1ms | 183.0ms | 183.0ms |
| grade | 0.0ms | 0.0ms | 0.0ms | 0.0ms |
| generate | 3422.7ms | 0.0ms | 24667.8ms | 24667.8ms |
| end-to-end | 3514.7ms | 97.5ms | 24740.1ms | 24740.1ms |

**Reading these numbers:**

- **Retrieval is the healthy part.** recall@5 of 0.80 means the labeled document is in the
  top 5 four times out of five. Retrieval is not the bottleneck.
- **Answer overlap is a weak metric, not necessarily a weak answer.** It counts shared
  keywords against the reference answer after stopword removal, so a correct answer phrased
  differently scores low. Treat 0.16–0.20 as a regression baseline to compare future runs
  against, not as an accuracy percentage. Judging real answer quality needs LLM-as-judge or
  manual review — the honest open item in this phase.
- **Correct refusal at 0.62 is the real quality gap.** Roughly 38% of unanswerable questions
  still produced an answer. Since relevance grading now blocks weak retrievals, the remaining
  failures are cases where chunks clear the cutoff but do not contain the answer — which is a
  generation-side problem (`llama3.2:3b` not honouring "say so — don't guess"), not a
  retrieval one. A stronger model is the most direct lever.
- **Latency is entirely generation.** Retrieval is ~90ms; grading is free. The p50 of 97ms
  reflects that half the sample refuses and never calls the LLM at all. The 24.7s max is a
  long-context generation — worth capping with `num_predict` if it matters.
- **Baseline caveat:** these were run before tracing was added. Tracing only observes, so
  results should be unchanged, but the first re-run establishes the post-instrumentation
  baseline properly.

---

## Key Decisions to Lock Early

| Decision | Default choice | Revisit if... |
|---|---|---|
| Vector store | PostgreSQL + pgvector | need managed/serverless scale → Pinecone |
| Chunk size | 1000 tokens, 100 overlap | retrieval missing context → increase; noisy results → decrease |
| Embedding model | nomic-embed-text (Ollama, local) | quality insufficient → larger local model or hosted API |
| LLM | llama3.2:3b (Ollama, local GPU) | quality insufficient → qwen2.5:7b, or hosted API if local ceiling hit |
| Reranking | off | top-k results noisy/irrelevant → add local cross-encoder rerank |

## Overall Done Markers

- [x] Pipeline runs end-to-end from raw doc to answer
- [x] Retrieval returns relevant chunks for test query set — recall@5 0.80
- [x] Generated answers grounded in retrieved context, cite sources
- [x] Tests pass — 50 unit + 7 integration
- [ ] Eval metrics meet threshold — **no threshold was ever set.** Baseline is recorded above;
      correct-refusal at 0.62 is the number worth setting a target against first.
- [ ] (if applicable) UI functional for manual querying — see `ui_plan.md`

---

## Remaining open items

Everything else in this plan is complete. What is deliberately not done:

| Item | Why | Where it goes |
|---|---|---|
| App container does not ingest | `docker compose up` starts the app against an empty vectorstore; ingestion is a manual `python -m src.ingestion.pipeline`. Deferred by decision — the UI will own ingestion. | `ui_plan.md`, Ingest view |
| pgvector index tuning (ivfflat/hnsw) | 223 chunks. Sequential scan is faster than an index at this size; retrieval is ~90ms. | Revisit past a few thousand rows |
| Inline `(source: ...)` citations | `llama3.2:3b` does not honour the instruction at this model size. Worked around in code via `src/rag/citations.py`, which is deterministic and arguably better. | Needs a larger model |
| `qwen2.5:7b` evaluation | ~5GB pull, tight on 4GB VRAM. Not attempted — needs explicit go-ahead. | Would likely lift the 0.62 refusal rate |
| LLM-as-judge answer scoring | Keyword overlap is a regression signal, not an accuracy measure. | Phase 8 follow-up |
| Phase 7 enhancements | Post-MVP by design. Query rewriting is the one that would justify reinstating a retry loop. | `ui_plan.md` covers the UI item |
