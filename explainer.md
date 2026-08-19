# RAG System — Explainer

Companion to `plan.md`. `plan.md` is the checklist; this file explains *why* each piece
exists and how the parts fit together.

## What this system is

A retrieval-augmented generation pipeline. A user asks a question. Instead of sending
that question straight to an LLM (which would answer from whatever it memorised during
training, confidently and sometimes wrongly), the system first searches a private
document collection for passages relevant to the question, then hands those passages to
the LLM as context and asks it to answer using only that material.

Two benefits: the model can answer questions about documents it never saw during
training, and its answers can be traced back to a source.

## Why it runs entirely locally

Every component — the LLM, the embedding model, the vector database — runs in a
container on this machine. No document text and no query ever leaves the host, there is
no per-token cost, and the whole stack comes up with one `docker compose up`.

The tradeoff is quality and speed: `llama3.2:3b` is a small model chosen to fit inside
the 4GB of VRAM on a GTX 1050. It is adequate for grounded question answering over
retrieved context, which is an easier task than open-ended reasoning, but it will not
match a frontier hosted model. Swapping to a hosted API later means changing one
`ChatOllama(...)` construction — the rest of the pipeline is unaffected.

## The pieces

**Ollama** serves two models over HTTP: `llama3.2:3b` generates answers, and
`nomic-embed-text` turns text into 768-dimensional vectors. It runs with GPU passthrough
so inference uses the NVIDIA card rather than the CPU.

**Postgres + pgvector** stores the document chunks alongside their embedding vectors and
answers nearest-neighbour queries ("which 5 chunks are closest to this query vector?").
Using Postgres rather than a dedicated vector database means chunk text, metadata, and
vectors live in one system that already handles backups, transactions, and SQL.

**LangChain** supplies the adapters — document loaders, text splitters, and the
`OllamaEmbeddings` / `PGVector` integrations — so the pipeline is not hand-rolled HTTP
calls and SQL.

**LangGraph** orchestrates control flow as an explicit state graph rather than a linear
script. This matters because good RAG is not one straight pass: retrieval sometimes
returns nothing useful, and the graph can grade what came back and loop to retry before
generating. Expressing that as nodes and conditional edges keeps the retry logic
inspectable instead of buried in nested conditionals.

## How a query flows through it

```
Query → [Retrieve] → [Grade/Filter] → [Generate] → Answer
             ↑              │
             └──── retry if docs weak ────┘
```

1. **Retrieve** — the query is embedded with `nomic-embed-text`, and pgvector returns the
   `k` nearest chunks by cosine distance.
2. **Grade** — each retrieved chunk is checked for actual relevance. Vector similarity is
   a blunt instrument; the closest chunks are not always useful ones. Chunks that fail
   are dropped.
3. **Retry** — if grading leaves nothing, the graph loops back to retrieve (bounded at 2
   attempts, so a bad query fails fast instead of spinning).
4. **Generate** — surviving chunks are formatted into a prompt that instructs the model to
   answer from the context alone and to say so when the context is insufficient. That
   instruction is what keeps answers grounded.

## Ingestion, which happens first

Before any query works, documents must be loaded, split, embedded, and stored.

Splitting is the step that most affects quality. Documents are cut into ~1000-character
chunks with 100-200 characters of overlap. Chunks that are too small lose the context
that makes them meaningful; chunks that are too large dilute the relevance signal and
waste the LLM's context window. The overlap exists so a sentence spanning a chunk
boundary is not lost to both sides. These numbers are a starting point to tune against
real queries, not a settled answer.

The one hard constraint: **the embedding model used at ingest must be the same one used
at query time.** Different models produce vectors in incompatible spaces. A mismatch does
not raise an error — retrieval simply returns nonsense.

## Why Docker specifically

The stack has awkward host dependencies: a Postgres instance with a C extension
compiled in, an LLM server with CUDA libraries, and specific model weights. Compose
declares all of it, pins image versions, and wires the services together on a private
network with health checks, so `app` does not start until `db` is accepting connections
and the models have finished downloading.

The one thing Docker cannot provide is the GPU. That requires the NVIDIA driver and
`nvidia-container-toolkit` on the host, which is why they are the only manual
prerequisites.

### The Docker Desktop caveat

Docker Desktop on Linux runs its daemon inside a VM and cannot pass through the GPU. The
`ollama` service will fail with `could not select device driver "nvidia"`. This project
must run against the native engine:

```bash
docker context use default
```

Note that each daemon keeps its own image store, so images pulled under one context are
invisible to the other.

## Layout

```
src/loaders.py      read source files into Documents
src/splitter.py     cut Documents into chunks
src/vectorstore.py  embed chunks, read/write pgvector
src/nodes.py        retrieve / grade / generate node functions
src/graph.py        state schema, graph wiring, entrypoint
src/prompts.py      prompt templates
data/raw            source documents
data/processed      derived artifacts
```

`src/graph.py` currently holds a Phase 1 smoke check that verifies the app container can
reach Postgres and Ollama and that the pgvector extension is enabled. The real graph
replaces it in Phase 5.
