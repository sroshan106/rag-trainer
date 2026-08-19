# UI & Dashboard Implementation Plan

Companion to `plan.md`. That document covers the RAG pipeline itself; this one covers the
web interface layered on top of it — ingestion, index management, querying, benchmarking,
tracing, logging, latency, and system metrics.

Constraints carry over from `plan.md` with one deliberate exception. The system stays
**local-first and fully containerized** — Ollama on the GPU remains the default and the
fallback, and observability stays on-machine, which still rules out LangSmith and any hosted
observability vendor in favour of the Postgres instance already running.

The exception is generation: the UI adds **optional hosted LLM providers**, selected per
profile with a user-supplied API key. This is opt-in, off by default, and never silently
engaged. See *Hosted model providers* below — it carries real consequences for data egress,
cost, and credential handling that the local-only design did not have.

---

## Scope note: "training"

A retrieval-augmented system does not train a model. Nothing in this project updates model
weights, and nothing should — `llama3.2:3b` and `nomic-embed-text` are used as-is via Ollama.
Fine-tuning either one is out of reach on a 4GB GTX 1050 regardless.

What is genuinely meant by "train the data" in a RAG context, and what this plan builds, is
**index management**: chunking, embedding, and re-embedding the corpus, plus the ability to
run several configurations side by side and measure which retrieves better. That is the knob
that actually changes answer quality here. The dashboard view is named **Index** rather than
Training to keep the distinction honest, and it covers:

- ingest new documents into a collection
- re-chunk and re-embed an existing collection with different parameters
- maintain multiple collections simultaneously and compare them on the benchmark

If real fine-tuning becomes a goal later it belongs in a separate plan with different
hardware assumptions — a LoRA on a 3B model needs roughly 12–16GB VRAM.

---

## Architecture

```
Browser (React + Vite)
   │  REST for actions, SSE for streams
   ▼
FastAPI  (src/api/)
   ├── /api/query      → src.rag.graph          (existing)
   ├── /api/ingest     → src.ingestion.pipeline (existing)
   ├── /api/benchmark  → tests.benchmark        (existing, needs to return not print)
   ├── /api/traces     → observability tables   (new)
   ├── /api/logs       → log ring buffer        (new)
   └── /api/metrics    → psutil + NVML + docker (new)
   ▼
Postgres (existing container)  ─ pgvector collections + observability tables
Ollama   (existing container)  ─ LLM + embeddings
```

The API layer is deliberately thin. Every route delegates to code that already exists in
`src/`; the UI must not become a second implementation of the pipeline. Where a module
currently prints (`src/ingestion/pipeline.py`, `tests/benchmark/run_benchmark.py`), it gets
refactored to return structured results and the CLI entrypoint prints those — the API
consumes the same return value.

### Stack

**Backend:** FastAPI + Uvicorn. Async matters here because ingestion and benchmark runs are
long, and metrics/logs/answer-tokens all stream concurrently.

**Frontend:** React via Vite, TailwindCSS, Recharts for charts.

Streamlit was considered and rejected. It would be faster to a first screen — perhaps a day —
but its execution model reruns the whole script on every interaction, which fights every
requirement here: live 1Hz metric streams, token-by-token answer streaming, and a multi-panel
dashboard layout. It is a reasonable fallback if the UI needs to exist this week and can be
thrown away later.

**Job handling:** FastAPI `BackgroundTasks` plus a `jobs` table for status and progress.
No Celery, no Redis. This is a single-user local tool; a task queue would be infrastructure
for its own sake. The one requirement is that ingestion and benchmark runs survive a browser
refresh, which the `jobs` table provides.

### New layout

Follows the existing domain-segregated `src/` convention:

```
src/
├── api/
│   ├── app.py            # FastAPI instance, CORS, lifespan
│   ├── routes/
│   │   ├── query.py
│   │   ├── ingest.py
│   │   ├── collections.py
│   │   ├── benchmark.py
│   │   ├── traces.py
│   │   ├── logs.py
│   │   └── metrics.py
│   └── schemas.py        # pydantic request/response models
├── observability/
│   ├── tracing.py        # span recording
│   ├── logging.py        # JSON formatter + ring buffer
│   └── sysmetrics.py     # psutil / NVML / docker collectors
└── jobs/
    └── runner.py         # background job registry + progress
ui/                       # Vite app, separate from src/
```

---

## Can the browser read CPU, GPU, and disk?

**No.** This was checked specifically, and the answer determines the whole metrics design.

Browsers expose no API for host CPU utilization, GPU utilization, VRAM, disk free space, or
disk I/O. Every such API was either never specified or was removed for fingerprinting and
side-channel reasons. What actually exists:

| API | What it returns | Why it is insufficient |
|---|---|---|
| `navigator.hardwareConcurrency` | Logical core count | Static integer. No utilization. |
| `navigator.deviceMemory` | System RAM in GB | Chromium only. Rounded to 0.25/0.5/1/2/4/8 and **capped at 8** for fingerprint resistance. Static. |
| `performance.memory` | JS heap used/total/limit | Non-standard, Chromium only. Measures this tab's JavaScript heap — nothing about the host. |
| `performance.measureUserAgentSpecificMemory()` | Per-tab memory breakdown | Requires `crossOriginIsolated`. Still tab-scoped, not system. |
| `navigator.storage.estimate()` | Origin quota and usage | Browser storage budget for this origin. **Not** disk free space; the quota is a fraction of free space and is deliberately fuzzed. |
| WebGL `WEBGL_debug_renderer_info` | GPU model string | Firefox and Safari mask or spoof it by default. A name only — no utilization, no VRAM. |
| WebGPU `adapter.info` | vendor / architecture / device | Identification only. The spec exposes no utilization or memory counters. |
| Compute Pressure API | Coarse state: `nominal`/`fair`/`serious`/`critical` | Chromium 125+ only, needs secure context and permissions policy. Buckets, not numbers, and intentionally so. The closest thing that exists, and still not a percentage. |
| Battery Status API | Charge level, charging state | Removed from Firefox and Safari over fingerprinting. Irrelevant on a desktop. |

The deliberate design intent across all of these is that a web page must not be able to
profile the machine it runs on. That intent is not going to be worked around, and attempting
to infer CPU load by timing JavaScript loops produces numbers too noisy to display.

### Therefore: metrics are collected server-side and streamed to the browser

Verified working on this machine:

```
$ nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
NVIDIA GeForce GTX 1050, 4096 MiB, 2 MiB, 0 %

$ docker exec rag-ollama-1 nvidia-smi --query-gpu=name,utilization.gpu --format=csv
NVIDIA GeForce GTX 1050, 0 %
```

GPU telemetry is reachable from the host and from inside the `ollama` container.

**Collectors** (`src/observability/sysmetrics.py`):

- **CPU / RAM / disk / network** — `psutil`. Per-core percentages, memory and swap,
  `disk_usage` for free space, `disk_io_counters` for throughput.
- **GPU** — `pynvml` (the `nvidia-ml-py` package). Utilization, VRAM used/total, temperature,
  power draw, clocks. Preferred over shelling out to `nvidia-smi` — no subprocess per sample,
  and structured values instead of parsed CSV.
- **Per-container** — Docker SDK over `/var/run/docker.sock`, giving CPU and memory for
  `ollama`, `db`, and `app` separately. This is what distinguishes "the box is busy" from
  "Ollama is busy".

**Transport** — Server-Sent Events at `GET /api/metrics/stream`, one frame per second.
SSE rather than WebSocket because the flow is strictly server-to-client; it reconnects
automatically and needs no extra protocol handling.

### Two container problems to solve

**1. The `app` container cannot see the GPU as configured.** Only the `ollama` service has a
device reservation. To let the API read NVML directly, `app` needs its own:

```yaml
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, utility]
```

The `utility` capability is the load-bearing part — it is what places NVML and `nvidia-smi`
inside the container. `[gpu]` alone yields compute access without the management library.

Alternative if that proves awkward: read GPU stats by exec'ing into the `ollama` container
through the mounted docker socket. Slower and uglier, but it needs no change to `app`'s
device configuration.

**2. Mounting the docker socket grants root-equivalent host access.** Any process that can
talk to `/var/run/docker.sock` can start a privileged container and own the host. For a
single-user local dev dashboard this is a normal tradeoff, but it should be a deliberate one:

- mount it read-only (`/var/run/docker.sock:/var/run/docker.sock:ro` — note this limits the
  API surface but is not a real security boundary, since the socket is an API not a file)
- never expose the app's port beyond `127.0.0.1`
- if per-container stats turn out not to be worth it, drop the socket mount entirely and keep
  only psutil + NVML, which cover the metrics that actually matter

A third option that avoids the socket altogether is running the collector as a host-side
process outside Docker and having the API read from it. Cleanest security posture, worst
portability — it breaks the "one `docker compose up`" property that `plan.md` treats as the
portability target.

### The metric that actually matters here

On a 4GB card running a 3B model alongside an embedding model, **VRAM headroom is the number
to watch.** When `llama3.2:3b` and `nomic-embed-text` cannot both stay resident, Ollama
evicts one or offloads layers to CPU, and generation latency degrades sharply rather than
gradually. The dashboard should give VRAM used/total top billing, and the latency view should
make the resulting cliff visible — correlating a generation-latency spike with a VRAM event
is the single most useful diagnostic this dashboard can offer.

---

## Instrumentation the pipeline needs first

The dashboard cannot show tracing, latency, or logs that are not being recorded. None of this
exists today. This work comes before any frontend code.

### Tracing

Two tables in the existing Postgres database:

```sql
CREATE TABLE traces (
    id            uuid PRIMARY KEY,
    query         text NOT NULL,
    collection    text NOT NULL,
    started_at    timestamptz NOT NULL,
    duration_ms   integer,
    answer        text,
    sources       jsonb,
    refused       boolean NOT NULL DEFAULT false,
    error         text
);

CREATE TABLE spans (
    id            bigserial PRIMARY KEY,
    trace_id      uuid NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    node          text NOT NULL,          -- retrieve | grade | generate
    started_at    timestamptz NOT NULL,
    duration_ms   integer NOT NULL,
    payload       jsonb NOT NULL
);
CREATE INDEX ON spans (trace_id);
CREATE INDEX ON traces (started_at DESC);
```

Recorded per node:

- **retrieve** — `k`, every returned score, each chunk's `row_index` and source URL
- **grade** — the computed cutoff, which of `RELEVANCE_FLOOR` / `RELEVANCE_RATIO` bound it,
  and the kept/dropped verdict per chunk. This makes the refusal path auditable, which is
  the whole point of having added the cutoffs.
- **generate** — prompt length, model, temperature, and Ollama's own timing metadata.

That last item is worth calling out: Ollama returns `eval_count`, `eval_duration`,
`prompt_eval_count`, and `prompt_eval_duration` on every response, surfaced through
`response.response_metadata` on the LangChain `ChatOllama` result. That yields true
tokens/second and a clean split between prompt processing and generation without any
estimation. Capture it verbatim.

Implementation should be a decorator or context manager in `src/observability/tracing.py`
wrapping each node, so `src/rag/nodes.py` stays readable. Tracing must be failure-isolated —
a broken span write must never break a query — and disableable via `RAG_TRACING=false`,
matching the existing `RAG_CITATIONS` convention.

### Logging

Structured JSON to stdout via a custom `logging.Formatter`, so `docker compose logs` stays
useful, plus an in-process ring buffer (a bounded `collections.deque`) that the UI tails over
SSE. The ring buffer avoids a second write path to Postgres for data that is mostly noise;
anything worth keeping is already in `traces`.

### Latency

Derived entirely from `spans` — no separate timing store. p50/p95/p99 per node over a
selected window are a single SQL query using `percentile_cont`.

---

## Dashboard views

### 1. Ask

The primary view. Query box, streaming answer, and beneath it the retrieval detail that makes
the answer auditable: every retrieved chunk with its relevance score, the computed cutoff
drawn as a line, and dropped chunks shown greyed rather than hidden. When the system refuses,
this view shows *why* — which is exactly the information the current CLI throws away.

Thumbs up/down on each answer, written to a `feedback` table. Over time that accumulates into
the labeled evaluation set that `plan.md` Phase 8 calls for, without a separate labeling
effort.

Requires `generate_node` to gain a streaming path (`llm.stream` rather than `llm.invoke`) and
the graph to be invoked with `astream_events` so tokens reach the browser as they are produced.

### 2. Ingest

Drag-and-drop upload for CSV and PDF. Parse preview showing detected columns and row count
before committing. Chunk configuration — size and overlap — with a **dry-run preview** that
shows how the current settings would actually split a sample document. That preview is the
feature that makes `plan.md`'s "spot-check 5–10 chunks manually" a two-second operation
instead of a chore that never gets done.

Then run ingestion as a background job with live progress, streamed over SSE.

### 3. Index

Collection management, per the scope note above.

- list collections with chunk counts, embedding model, chunk parameters, created date
- create a new collection from the same source documents with different chunk settings
- re-embed an existing collection
- delete a collection
- **compare two collections on the benchmark** — this is the payoff. Chunk size 500 vs 1000
  stops being a guess and becomes a measured recall@5 difference.

This requires `COLLECTION_NAME` in `src/vectorstore/store.py` to become a parameter rather
than the module constant it is today, threaded through the retrieve path so a query can be
issued against a chosen collection.

### 4. Benchmark

Trigger a run, watch per-question progress, and read results against history.

`tests/benchmark/run_benchmark.py` already computes recall@k, answer overlap, and refusal
rate, but prints them and keeps nothing. It needs a `benchmark_runs` table storing each run's
metrics alongside the configuration that produced them — collection, model, `RELEVANCE_FLOOR`,
`RELEVANCE_RATIO`, chunk settings — so results become comparable across runs.

Views: metric trend lines across runs, per-question drill-down linking to the full trace, and
a run-vs-run diff highlighting which questions changed verdict. The diff is what catches a
tuning change that improves the mean while breaking specific queries.

Note the benchmark has never been executed. Its first run establishes the baseline, and it
should be run *before* further tuning so the relevance-cutoff change already made has a
measured effect rather than an assumed one.

### 5. Traces

Searchable list of recent queries with total latency, refusal flag, and source count. Opening
one shows a waterfall of its three spans with the full payload captured above — the retrieval
scores, the grading verdicts, the generation token counts.

Filters worth having on day one: refusals only, slowest first, and zero-sources.

### 6. Latency

- p50/p95/p99 end-to-end over a selected window
- stacked breakdown by node, which will show generation dominating retrieval by a wide margin
- tokens/second over time from Ollama's own metadata
- prompt-processing versus generation split
- latency plotted against concurrent VRAM usage, to expose the eviction cliff described above

### 7. System

Live host and container metrics from the SSE stream: CPU per core, RAM, swap, disk free and
I/O throughput, and GPU utilization, VRAM, temperature, and power. Per-container CPU and
memory for `ollama`, `db`, and `app`.

Prominent VRAM gauge with a threshold warning, for the reason given above.

Because these values arrive from the server rather than the browser, the same view works
identically when the dashboard is opened from another machine on the LAN — it reports the
machine actually running the models, which is the correct behavior and one more reason the
browser APIs would have been wrong even if they existed.

### 8. Logs

Live tail over SSE with level filtering and text search, plus a pause control — an
auto-scrolling log that cannot be stopped is unreadable during an incident.

### 9. Settings

Exposes the configuration surface catalogued in the next section.

---

## Configuration surface

Every knob the UI could expose, and — more importantly — which ones it *may* expose. Three
facts determine where each setting can live, and getting them wrong produces a dashboard that
either lies about what it changed or silently corrupts the index.

### Settings have three different lifetimes

**1. Baked into the index.** Chunk size, chunk overlap, and the embedding model are frozen
into a collection when it is built. They cannot be live toggles: changing chunk size does
nothing to already-embedded vectors, and changing the embedding model silently breaks
retrieval, because query-time and ingest-time embeddings must come from the same model —
`plan.md` Phase 4 already flags this as the classic failure that produces no error, just bad
results. These are **immutable properties of a collection**, chosen at creation. The UI offers
them when creating or re-embedding a collection, and displays them read-only thereafter.

**2. Live per query.** `k`, relevance cutoffs, generation parameters, prompt template, and
citations can change between one request and the next with no re-indexing. These are the
settings that belong in the Ask view as adjustable controls.

**3. Server-owned, never browser-editable.** `DATABASE_URL` and `OLLAMA_BASE_URL` stay
read-only in the UI. A browser-editable base URL is a server-side request forgery primitive —
the backend would dutifully connect wherever the field points, including internal addresses —
and `DATABASE_URL` carries credentials. Display them; never accept them.

### "Per model" means two different models

The request was for per-model settings, and the distinction matters because this project uses
two models with disjoint tuning:

- `RAG_RELEVANCE_FLOOR` and `RAG_RELEVANCE_RATIO` are properties of the **embedding model**,
  not the chat model. They are thresholds on cosine similarity scores, and every embedding
  model produces its own score distribution. The 0.6 floor was derived empirically against
  `nomic-embed-text` on this corpus — on-topic hits 0.67–0.80, noise 0.44–0.48, gibberish
  topping out at 0.56. Swap the embedding model and those numbers are meaningless. So the
  cutoffs belong to the **collection**, which is what pins the embedding model.
- Temperature, context length, and sampling parameters are properties of the **chat model**.
  They travel with the LLM, not the index.

A single flat "per model" settings page would conflate these. The structure that fits is
**named profiles**: a profile bundles a chat model with its generation parameters and its
retrieval settings, and references a collection. Switching from `llama3.2:3b` to `qwen2.5:7b`
then carries its own context length and temperature, while switching collections carries its
own cutoffs. Profiles are also what the Benchmark view compares — a benchmark run records the
profile that produced it, which is what makes runs comparable at all.

### Precedence

```
request parameter  >  active profile  >  environment variable  >  code default
```

Explicit per-request overrides win, so the Ask view can experiment without mutating saved
state. Environment variables remain the deployment-level default, preserving how the CLI and
`docker compose` behave today.

**Implementation note:** the current constants are read at *module import* —
`src/rag/nodes.py:17` and `:20` evaluate `os.environ.get` exactly once, and `RETRIEVE_K`,
the model name, and `temperature` are hardcoded literals at `:12` and `:38`. None of them can
be changed at runtime as written, and a settings page that writes environment variables would
appear to work while changing nothing until restart. Making these live requires threading a
config object through the graph state into the nodes, replacing the module-level reads. This
is Phase A work and a prerequisite for the Settings view, not something to bolt on later.

### Catalogue

Existing environment variables, live today:

| Setting | Env var | Default | Scope | UI control |
|---|---|---|---|---|
| Citations | `RAG_CITATIONS` | `true` | Per query | Toggle |
| Relevance floor | `RAG_RELEVANCE_FLOOR` | `0.6` | Per collection | Slider 0.0–1.0, step 0.01 |
| Relevance ratio | `RAG_RELEVANCE_RATIO` | `0.8` | Per collection | Slider 0.0–1.0, step 0.01 |
| Ollama endpoint | `OLLAMA_BASE_URL` | `http://localhost:11434` | Server | Read-only |
| Database URL | `DATABASE_URL` | — | Server | Read-only, credentials masked |

Currently hardcoded, should become configurable:

| Setting | Location now | Default | Scope | UI control |
|---|---|---|---|---|
| Retrieval `k` | `nodes.py:12` | `5` | Per query | Number, 1–50 |
| Chat model | `nodes.py:38` | `llama3.2:3b` | Per profile | Dropdown from installed Ollama models |
| Temperature | `nodes.py:38` | `0` | Per profile | Slider 0.0–2.0 |
| Collection | `store.py:9` | `rag_chunks` | Per query | Dropdown |
| Embedding model | `store.py:14` | `nomic-embed-text` | **Per collection, immutable** | Dropdown at creation only |
| Chunk size | `splitter.py:6` | `1000` | **Per collection, immutable** | Number at creation only |
| Chunk overlap | `splitter.py:7` | `150` | **Per collection, immutable** | Number at creation only |
| Prompt template | `prompts.py:5` | — | Per profile | Text editor, versioned |
| Source path | `pipeline.py:7` | `data/documents.csv` | Per ingest | File picker |

Ollama generation parameters, not currently plumbed at all, all per profile. `ChatOllama`
accepts these directly:

| Setting | Default | Why it matters here |
|---|---|---|
| `num_ctx` | 2048 | Context window. **The main VRAM lever on a 4GB card** — raising it to fit more retrieved chunks is often what triggers the CPU-offload cliff described earlier. Pair the control with a VRAM readout. |
| `num_predict` | -1 | Max output tokens. The simplest cap on worst-case latency. |
| `keep_alive` | 5m | How long a model stays resident. Directly controls whether the embedding model and chat model fight over VRAM. Worth surfacing given they are co-resident. |
| `top_p` / `top_k` | 0.9 / 40 | Sampling. Largely inert at `temperature=0`, but exposed for when temperature is raised. |
| `repeat_penalty` | 1.1 | Sampling. |
| `seed` | — | **Set this for reproducible benchmark runs.** Without it, run-vs-run diffs partly measure sampling noise. |
| `stop` | — | Stop sequences. |

Benchmark parameters, from `tests/benchmark/run_benchmark.py`:

| Setting | Location | Default | Notes |
|---|---|---|---|
| Overlap threshold | `run_benchmark.py:59` | `0.3` | Pass/fail line for answer keyword overlap. |
| Question sets | `DATA_DIR` CSVs | all three | Checkbox per set. |
| Refusal patterns | `run_benchmark.py:19` | 8 patterns | Editable list — refusal wording is model-specific, and a model that refuses in unlisted words is scored as hallucinating. |
| Profile under test | — | active | What makes runs comparable. |

Tracing and observability, to be introduced in Phase A:

| Setting | Default | Notes |
|---|---|---|
| `RAG_TRACING` | `true` | Follows the `RAG_CITATIONS` convention. |
| Trace retention | — | See open decision 3. |
| Log level | `INFO` | Live-changeable without restart. |
| Metrics sample interval | `1s` | Slower sampling for long unattended runs. |

### Guardrails

- **Validate ranges server-side.** A `k` of 10,000 or a negative floor should be rejected by
  the API, not merely constrained by a slider — the API is reachable independently of the UI.
- **Warn before destructive re-embedding.** Re-embedding a collection discards existing
  vectors and, on this hardware, takes real time. Require confirmation and state the cost.
- **Never expose arbitrary model pulls.** A free-text model field lets the browser fill the
  disk with multi-gigabyte weights. Offer a dropdown of already-installed models, and treat
  pulling a new one as a separate, explicit, confirmed action.
- **Record the config in every trace.** A trace whose parameters are unknown cannot explain
  an answer. The `traces` table should store the resolved configuration used, not merely a
  profile reference that may later be edited.

---

## Hosted model providers

The UI can point generation at a hosted API instead of local Ollama, configured per profile
with a user-supplied key. Worth stating plainly before the design: this reverses the
property that made the original architecture defensible. Retrieved chunks — actual content
from your corpus — are sent to a third party on every query, the pipeline gains a network
dependency and a rate limit it never had, and queries start costing money. None of that makes
it a bad idea; hosted models are markedly stronger than `llama3.2:3b`, and the citation
weakness documented in `plan.md` Phase 6 is precisely a small-model failure that a frontier
model does not have. It just needs to be a visible, deliberate switch rather than a
configuration detail — which is what the rest of this section builds.

### Provider abstraction

`src/rag/nodes.py:38` hardcodes `ChatOllama`. It becomes a factory keyed on the profile's
provider, returning a LangChain chat model — every provider below exposes the same
`.invoke()` / `.stream()` surface, so `generate_node` does not change:

| Provider | Package | Key env var | Notes |
|---|---|---|---|
| Ollama (local) | `langchain-ollama` | — | Default. No key, no cost, no rate limit. |
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` | See model table below. |
| OpenAI | `langchain-openai` | `OPENAI_API_KEY` | |
| Google Gemini | `langchain-google-genai` | `GOOGLE_API_KEY` | |
| OpenAI-compatible | `langchain-openai` + `base_url` | varies | Covers OpenRouter, Groq, Together, vLLM. One adapter, many backends. |

The last row is worth having early — a configurable `base_url` on the OpenAI adapter covers a
long tail of providers for no extra code.

**Anthropic models and pricing** (per million tokens; verified against current Anthropic
documentation at time of writing):

| Model | Model ID | Context | Input | Output |
|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3.00 | $15.00 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |

Use the exact ID strings — they are complete as written and take no date suffix. Pricing for
the other providers is deliberately not tabulated here; look it up at implementation time
rather than trusting a number copied into a planning document.

If Anthropic support is implemented directly rather than through LangChain, use the official
`anthropic` SDK — not an OpenAI-compatible shim pointed at Anthropic's endpoint.

### Credential handling

This is the part that most deserves care, because the dashboard designed above is unusually
good at leaking secrets if built naively: it has a settings form, a structured logger, a
trace store that captures node payloads, and an SSE stream — four separate paths a key can
escape through.

**Rules:**

1. **Keys never reach the browser.** The settings form is write-only: `POST` accepts a key,
   `GET` returns a masked fingerprint (`sk-ant-…4f2a`), a validation timestamp, and nothing
   else. There is no endpoint that returns a key in full, for any caller.
2. **Keys never enter the frontend bundle.** No `VITE_`-prefixed key, ever — Vite inlines
   those into client-side JavaScript at build time, which publishes them to anyone who opens
   the page.
3. **Redact before logging, not after.** The JSON formatter gets a redaction filter keyed on
   known secret field names *and* on provider key patterns (`sk-ant-`, `sk-`, `AIza`).
   Provider SDK exceptions sometimes embed request headers in their string representation, so
   the filter must run on exception text too, not only on structured fields.
4. **Never persist a key into `traces` or `spans`.** The generate span records the resolved
   configuration; the key is replaced by a profile reference and a fingerprint before write.
   This is a concrete risk, not a hypothetical — the config-in-every-trace guardrail above
   would otherwise write the key to Postgres on every single query.
5. **Storage: env or a file, not the database.** Simplest correct option is a
   `.env`-supplied key per provider, mounted at container start and never written back to.
   If keys must be settable from the UI at runtime, store them in a file with `0600`
   permissions outside the image, or encrypt at rest in Postgres with a key from the
   environment — but note that a database-stored key is inside the same backup and log
   surface as everything else, which is why the file option is preferred.
6. **Localhost binding now protects credentials, not just the docker socket.** The earlier
   security note argued for `127.0.0.1` because of the socket mount. With provider keys in
   play, LAN exposure means credential theft and someone else's spend. If remote access is
   ever wanted, that is the point at which real authentication stops being optional.
7. **Validate on save.** A cheap round-trip to the provider — a token count or a one-token
   completion — confirms the key works and surfaces the error immediately, rather than at the
   next query.

### Cost control

Local inference is free, so nothing in the design so far has any concept of spend. Hosted
providers make several existing features expensive by default, and one of them is a genuine
foot-gun.

**The benchmark is the foot-gun.** It runs roughly 100 questions in one click. At `k=5` with
1000-token chunks, each query sends on the order of 6K input tokens and returns a few hundred
output — call it 600K input and 30K output per full run:

| Model | Approximate cost per benchmark run |
|---|---|
| Claude Opus 5 | ~$3.75 |
| Claude Sonnet 5 | ~$2.25 |
| Claude Haiku 4.5 | ~$0.75 |
| Local Ollama | $0.00 |

Not ruinous, but it is real money attached to a button that currently reads as free, and a
tuning session that runs the benchmark twenty times is a different conversation. Required:

- **Pre-flight estimate.** Before a benchmark or bulk operation runs against a hosted
  provider, show projected token count and cost and require confirmation.
- **Per-request cost recorded on every trace**, computed from the provider's reported usage
  and the profile's model rates. Cost then becomes a first-class dashboard metric alongside
  latency.
- **A spend cap.** A configurable daily or per-run ceiling that hard-stops further hosted
  calls, with the local model as the fallback. Without this, one runaway loop is unbounded.
- **Cost panel in the Latency view** — cost per query, per day, per profile. The comparison
  that matters is cost against measured benchmark quality, which is exactly the number the
  Benchmark view already collects.

### Keep embeddings local

Chat provider and embedding provider are separate decisions, and the recommendation is to
switch only the chat model. Hosted embeddings are the worse trade here for three reasons:

1. **Full-corpus egress.** Hosted chat sends only retrieved chunks. Hosted embeddings send
   *every chunk of every document* at ingest — categorically more data leaving the machine.
2. **It invalidates the collection.** Embedding model is an immutable collection property;
   changing it requires a full re-embed, and querying an old collection with a new embedding
   model silently breaks retrieval (`plan.md` Phase 4).
3. **It invalidates the relevance cutoffs.** `RELEVANCE_FLOOR = 0.6` was derived from
   `nomic-embed-text` score distributions on this corpus. A different embedding model
   produces a different distribution, so the floor must be re-derived empirically or the
   grader will either refuse everything or refuse nothing.

`nomic-embed-text` also runs comfortably in VRAM and costs nothing. Hosted embeddings should
be possible but should not be the recommended path, and the UI should warn about the re-embed
and re-tune when selected.

A useful side effect of hosted chat: freeing `llama3.2:3b` from VRAM leaves the whole 4GB to
the embedding model, which removes the co-residency pressure and model-eviction cliff
described in the metrics section. Worth surfacing in the System view — VRAM behaviour changes
character depending on which provider is active.

### Failure modes the local path never had

Ollama on localhost does not rate-limit, does not require the network, and does not bill.
Adding hosted providers means `generate_node` — which today has no error handling at all —
needs to deal with:

- **429 rate limits.** Provider SDKs retry with backoff by default (the Anthropic SDK retries
  429s and 5xx twice); surface the retry in the trace rather than letting it look like slow
  generation.
- **Timeouts and network failure.** Distinguish these from a slow local model in the UI —
  they look identical in a latency chart and have nothing in common.
- **Authentication and quota errors.** Surface distinctly and immediately: an expired key and
  an exhausted quota need different responses from the user.
- **Automatic fallback to local**, configurable per profile. When the hosted provider is
  unreachable, degrade to Ollama rather than failing the query — and **record the substitution
  in the trace**, because an answer silently produced by a different model than the profile
  names would corrupt every benchmark comparison built on top of it.

### Token accounting becomes provider-specific

The tracing design assumed Ollama's `eval_count` / `eval_duration` metadata. Hosted providers
report usage in their own shapes — Anthropic returns `usage.input_tokens`,
`usage.output_tokens`, plus `cache_read_input_tokens` and `cache_creation_input_tokens` when
prompt caching is in play. Normalize to a common `{input_tokens, output_tokens, cost_usd}`
record at the provider-adapter boundary so the Latency and Benchmark views stay
provider-agnostic. Tokens/second remains derivable for streaming responses across all
providers; Ollama's precise prompt-eval-versus-generation split does not, so that panel should
degrade rather than display zeros.

One provider-specific optimization worth noting: with prompt caching, the RAG system prompt
and instruction block are a stable prefix across queries and cache well. The retrieved context
changes per query and must sit *after* the cache breakpoint. This can cut hosted input cost
substantially and is worth building in from the start rather than retrofitting.

### Additional configuration surface

All per profile:

| Setting | Default | Notes |
|---|---|---|
| Provider | `ollama` | Dropdown. Selecting a hosted provider must show an egress warning. |
| Model | per provider | Dropdown, populated per selected provider. |
| API key | — | Write-only field. Masked on read. Validated on save. |
| Base URL override | — | For OpenAI-compatible providers. **Server-side allowlist required** — a free-text URL that the backend will dutifully connect to is the same SSRF primitive flagged for `OLLAMA_BASE_URL`. |
| Max tokens | provider default | Output ceiling. Direct cost control. |
| Request timeout | 60s | Hosted-only. |
| Max retries | 2 | Hosted-only. |
| Fallback to local | `true` | On hosted failure, degrade to Ollama and mark the trace. |
| Daily spend cap | — | Hard stop, per provider. |
| Prompt caching | `true` where supported | Provider-specific. |

---

## Dependencies to add

```
fastapi
uvicorn[standard]
psutil                  # CPU, RAM, disk
nvidia-ml-py            # GPU via NVML (imports as pynvml)
docker                  # per-container stats over the socket
python-multipart        # file upload
sse-starlette           # SSE responses
```

Hosted providers, optional — install only what is actually used:

```
langchain-anthropic     # Claude
langchain-openai        # OpenAI, and any OpenAI-compatible endpoint via base_url
langchain-google-genai  # Gemini
```

Frontend: `react`, `react-dom`, `vite`, `tailwindcss`, `recharts`.

---

## Phasing

**Phase A — Instrumentation.** Tracing tables and span recording, JSON logging with ring
buffer, `sysmetrics.py` collectors. Refactor `pipeline.py` and `run_benchmark.py` to return
structured results instead of printing. Parameterize `COLLECTION_NAME`. No UI yet, and
everything here is independently testable. *This phase is a prerequisite for every view.*

**Phase B — API.** FastAPI app, all routes, SSE streams for metrics/logs/tokens, background
job runner with the `jobs` table, GPU device reservation for the `app` service in compose.
Verifiable end to end with `curl` before any frontend exists.

**Phase C — Core UI.** Vite scaffold, layout and navigation, then Ask, Ingest, and System.
These three make the tool usable; the rest are analysis surfaces on top.

**Phase D — Analysis UI.** Traces, Latency, Benchmark, Index comparison. Charts and history
views, which are the most frontend-heavy and least blocking work in the plan.

**Phase E — Hosted providers.** Provider factory replacing the hardcoded `ChatOllama`,
credential storage with redaction, cost accounting and spend caps, fallback-to-local, and the
provider settings UI. Deliberately last: it is the only phase that adds an external
dependency and a way to spend money, and it is far easier to reason about once tracing and
cost surfaces already exist to observe it. The redaction filter (Phase A logging) should
nonetheless be written to handle key patterns from the start — retrofitting redaction after
keys have already been logged does not un-log them.

---

## Open decisions

1. **Auth.** Assumed none — bound to `127.0.0.1`, single user. If this is ever exposed on the
   LAN, the docker socket mount makes it a genuine remote-root risk, not a theoretical one —
   and once provider keys are stored, it is a credential-theft and spend risk as well.
2. **Docker socket.** Worth it for per-container stats, or drop it and keep psutil + NVML only?
   Recommend starting without it and adding it if the container-level breakdown proves needed.
3. **Trace retention.** Unbounded growth is fine for a while at single-user query volume, but
   the span payloads carry full retrieval score arrays. A retention window or a periodic prune
   job should be decided before this runs for months.
4. **Streaming and citations.** The `sources` list is currently computed after generation
   completes. With token streaming the answer arrives first and sources land at the end —
   acceptable, but the UI needs to render a pending state rather than an empty source list.
5. **Multi-collection querying.** Query one collection at a time, or fan out across several
   and merge? One at a time is simpler and sufficient for the A/B comparison use case.
6. **Which hosted providers to actually implement.** Anthropic plus the OpenAI-compatible
   adapter covers the most ground for the least code — the adapter alone reaches OpenRouter,
   Groq, Together, and vLLM. Adding Gemini is a third integration for one more vendor.
7. **Whether hosted keys are settable from the UI at all.** Env-only keys are meaningfully
   safer: no storage, no write endpoint, no encryption question, and one fewer path for a key
   to reach a log. The cost is that changing providers means editing `.env` and restarting.
   Recommend env-only until that friction actually bites.
8. **Whether benchmark runs may use hosted providers.** Given the per-run cost and the
   one-click trigger, a defensible default is local-only benchmarking with hosted runs behind
   an explicit confirmation — or an outright separate permission.
