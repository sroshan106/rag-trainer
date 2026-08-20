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

- [x] **Re-ranking** — local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reorders the fused candidate list before the grader sees it. `src/vectorstore/rerank.py`, on by default, `RAG_RERANK=false` to compare. The original deferral assumed `k=5` with nothing to reorder; hybrid retrieval fetches `RAG_FETCH_K=20` first, which is enough to make reordering meaningful. Runs on CPU (`RAG_RERANK_DEVICE=cpu`), so it costs latency rather than the VRAM the 4GB card does not have spare.
- [x] **Hybrid search** — Postgres full-text (`ts_rank_cd`) fused with dense retrieval by Reciprocal Rank Fusion. `src/vectorstore/lexical.py` + `src/vectorstore/hybrid.py`, on by default, `RAG_HYBRID=false` to compare. No re-ingest: the tsvector is a generated column over the text pgvector already stored.
- [ ] **Multi-turn memory** — add conversation history to state, rewrite follow-up queries with context
- [ ] **Query rewriting/expansion** — LLM rephrases vague queries before retrieval
- [x] **Streaming responses** — `ask_stream` in `src/rag/graph.py` steps the graph node by node and yields `stage`/`token`/`done`/`error` events; served over SSE by `src/api/routes/query.py` and rendered by the Ask view.
- [x] **Web UI** — FastAPI (`src/api/`) + React/Vite (`ui/`), with Ask, Ingest, System, and Benchmark views. Phases B and core C of `ui_plan.md`; Phase E (hosted providers, credentials, spend caps) deliberately not built.

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

Benchmark, 120 labeled questions, `llama3.2:3b` + `nomic-embed-text`, `k=5`. The current
configuration is hybrid retrieval at floor 0.56 / ratio 0.9; the two earlier columns are
kept because the comparison is the point.

| Question set | metric | dense, 0.6/0.8 | dense, 0.48/0.9 | **hybrid, 0.56/0.9** |
|---|---|---|---|---|
| single-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.88** |
| | mean overlap | 0.16 | 0.39 | **0.43** |
| | pass rate | 0.20 | 0.47 | **0.57** |
| multi-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.82** |
| | mean overlap | 0.20 | 0.31 | **0.30** |
| | pass rate | 0.25 | 0.42 | **0.35** |
| no-answer (n=40) | correct refusal | 0.62 | 0.20 | **0.35** |
| **all 120** | **questions right** | **0.357** | **0.363** | **0.423** |

The last row weights each suite's rate by its question count, so a configuration cannot win
by refusing everything or by answering everything.

Latency, 12 queries (mixed hits and refusals), single-threaded:

| node | mean | p50 | p95 | max |
|---|---|---|---|---|
| retrieve | 88.9ms | 80.1ms | 183.0ms | 183.0ms |
| grade | 0.0ms | 0.0ms | 0.0ms | 0.0ms |
| generate | 3422.7ms | 0.0ms | 24667.8ms | 24667.8ms |
| end-to-end | 3514.7ms | 97.5ms | 24740.1ms | 24740.1ms |

**Reading these numbers:**

- **Hybrid retrieval moved the ceiling; the threshold never could.** Sliding the cutoff from
  0.6 to 0.48 traded refusal for answers and landed at 0.363 against 0.357 — statistically
  nothing. Adding full-text search raised recall@5 from 0.80 to 0.88, which no cutoff value
  can do, and the combined score followed it to 0.423.
- **Correct refusal is still the weak metric**, and it is now the honest cost of the change:
  0.35 against the original 0.62. The floor was doing double duty as the refusal mechanism,
  and a lower floor gives that up. Full-text misses supply part of it back — an off-topic
  query matches no tsvector at all — but generation still answers when handed marginal
  context.
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
- **Overlap partly rewards verbosity.** A configuration that refuses less emits more words
  and collects more accidental keyword matches. Some of the gain from 0.16 to 0.43 is real
  (the right chunks now reach the model) and some is that artifact; the combined-score row
  exists to keep that honest.
- **The operating point was swept, not guessed.** `tests/benchmark/run_sweep.py` walked
  floors 0.44–0.60 at 12 questions/suite: combined 0.361, 0.361, 0.417, 0.528, 0.500. The
  peak at 0.56 is what is configured. At 12 questions a suite the resolution is 0.083, so
  treat the shape as real and the exact peak as approximate.
- **Latency numbers predate hybrid retrieval** and were measured before tracing existed.
  Retrieval now issues a second query per ask; the measured cost is ~250ms added to a ~90ms
  dense-only retrieve, which is still negligible against generation.

---

## Key Decisions to Lock Early

| Decision | Default choice | Revisit if... |
|---|---|---|
| Vector store | PostgreSQL + pgvector | need managed/serverless scale → Pinecone |
| Chunk size | 1000 tokens, 100 overlap | retrieval missing context → increase; noisy results → decrease |
| Embedding model | nomic-embed-text (Ollama, local) | quality insufficient → larger local model or hosted API |
| LLM | llama3.2:3b (Ollama, local GPU) | quality insufficient → qwen2.5:7b, or hosted API if local ceiling hit |
| Reranking | on — local cross-encoder on CPU, over the top `RAG_FETCH_K=20` | rerank latency dominates end-to-end time → `RAG_RERANK=false`, or shrink `RAG_RERANK_MAX_LENGTH` |

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
| Inline `(source: ...)` citations | `llama3.2:3b` ignores the instruction when handed five chunks, though it does honour it once hybrid grading narrows the context to one or two. Still worked around deterministically in `src/rag/citations.py`. | Unreliable below a larger model |
| `qwen2.5:7b` evaluation | ~5GB pull, tight on 4GB VRAM. Not attempted — needs explicit go-ahead. | Would likely lift the 0.62 refusal rate |
| LLM-as-judge answer scoring | Keyword overlap is a regression signal, not an accuracy measure. | Phase 8 follow-up |
| Multi-turn memory | The Ask view is a single-shot surface; follow-ups need conversation state plus query rewriting to resolve pronouns against history. Re-ranking and streaming, once grouped here, are both shipped. | Phase 7 |
| Query rewriting | The remaining Phase 7 item with real upside, and the prerequisite for reinstating a retry loop. Overlaps hybrid search, so it should be measured against the new 0.423 baseline rather than the old one. | Phase 7 |
| `ui_plan.md` Phases D and E | Traces, Latency, Index views; then hosted providers, credential storage, and spend caps. Phase E is the only part that can spend money. | `ui_plan.md` |
