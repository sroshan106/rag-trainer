# Future Ideas

Loose backlog of directions worth exploring later. Not committed, not scoped — just captured so they don't get lost.

## 1. Auto-suggest model pair based on uploaded document + local hardware

When a user uploads a document (or corpus) to ingest, auto-recommend:

- **One local model** — chosen based on document characteristics (size, language, domain) *and* the current machine's actual capability (VRAM, RAM, CPU core count, whether a CUDA/ROCm GPU is present at all). The suggestion should degrade gracefully: e.g. suggest a 3B model on an 8GB-VRAM laptop, a 7-8B model on a 24GB GPU, and fall back to CPU-only quantized options if no GPU detected.
- **One remote/hosted LLM** — chosen based on document complexity (e.g. long-context needs, multi-hop reasoning implied by the questions) rather than hardware, since it isn't hardware-bound.

Needs:
- A hardware probe step (detect GPU vendor/VRAM, RAM, CPU) — could reuse ideas from the earlier GPU-monitoring discussion (`nvidia-smi`, `nvitop`, or a vendor-neutral probe).
- A lightweight document profiler (token count, vocabulary/domain heuristics, language detection) to drive the "why this model" reasoning.
- A recommendation table mapping (doc profile × hardware tier) → suggested local model, kept as a simple config/lookup rather than another ML model.

## 2. Auto-benchmark by uploading a test case file

Let a user upload their own question/answer CSV (same shape as `tests/benchmark/data/*.csv`) through a UI, and have the system:

1. Run it through the current live pipeline (retrieve → grade → generate).
2. Compute the same metrics `run_benchmark.py` already produces (recall@k, answer overlap, refusal rate).
3. Store the result (see idea 3) so it can be compared across runs/models over time.

Could reuse `tests/benchmark/run_benchmark.py` almost as-is as the core eval engine — just needs an entry point that takes an uploaded file path instead of the fixed `data/` directory, and returns structured results instead of printing.

Worth revisiting the eval metric itself too — the keyword-overlap `pass_rate` metric was seen to be harsh on correct-but-differently-worded answers (see benchmark diagnostic session where a graded answer got 0.67 overlap despite being essentially correct). A semantic-similarity-based scorer (embedding cosine sim between expected and actual answer) might be a truer signal than raw keyword overlap.

## 3. Persist benchmark history: hashes + model + results

Track benchmark runs over time so scores are comparable:

- Hash the uploaded test file (content hash) so re-runs of the same test set are recognized as the same "suite."
- Hash or fingerprint the corpus/vectorstore state (e.g. hash of ingested document set) so results are attributable to a specific data snapshot.
- Store per-run: test-set hash, corpus hash, model used (local + remote, if both were tried), metrics (recall@k, overlap, pass_rate, refusal_rate), timestamp.
- This becomes the audit trail for "did changing embedding model / grading threshold / prompt actually help."

Simplest storage: a small relational table (SQLite is already in the stack per `plan.md`/`ui_plan.md` references) — no need for anything fancier until scale demands it.

## 4. SaaS packaging: multi-user, login, hosted benchmarking

Turn the above into a hosted product:

- **Auth**: user accounts, login (email/password or OAuth). Each user's uploads, corpora, and benchmark history scoped to their account.
- **Per-user isolation**: separate vectorstore namespace/collection per user (or per project within a user), separate benchmark history table rows keyed by user_id.
- **Compute model**: since the "local model" recommendation is inherently tied to *someone's* hardware, in a SaaS context this either means:
  - running the "local" model on shared backend infra sized for the median user (loses the hardware-personalization angle), or
  - keeping true local-model support only for a self-hosted/desktop tier, and using only remote/hosted LLMs in the SaaS tier.
  - Worth deciding explicitly which tier does what before building further — the hardware-aware recommendation (idea 1) only makes sense if the app truly has access to the end machine's specs, which is true for a local/desktop app but not for a multi-tenant server.
- **Billing hook**: benchmark runs and remote-LLM calls are the natural metering unit (cost-bearing), while local-model runs on the user's own machine are free to the operator — could shape a pricing tier around that split.
- **Storage**: multi-tenant Postgres likely replaces SQLite at this point (matches "Postgres for both text collection and vector DB" option already noted in `plan.md`/document data).

## 5. On-premise hosting for multi-user orgs (internal or external tenants)

A third deployment shape, distinct from idea 4's public SaaS: the app runs inside an organization's own infrastructure (their datacenter, VPC, or a dedicated instance we manage on their behalf), but still serves multiple users — either purely internal staff, or a mix of internal + external users the org grants access to (e.g. an org's customers, contractors, partners).

This sits between "single local desktop app" and "public multi-tenant SaaS":

- **Auth stays multi-user, but scoped to the org**: SSO/SAML/OIDC against the org's own identity provider (Okta, Azure AD, Google Workspace) is more likely required than plain email/password, since it's an org deployment. Role distinction matters — internal admins vs internal users vs external guest users likely need different permission levels (e.g. who can upload new corpora/ingest documents vs who can only query).
- **Hardware-aware model suggestion (idea 1) becomes viable again here** — unlike public SaaS, the org's own server hardware is known and fixed, so the "local model" recommendation logic can target that one machine/cluster's actual specs rather than a hypothetical median user.
- **Data stays on the org's infrastructure** — this is likely the entire pitch versus public SaaS: no document or query ever leaves the org's network boundary. Matters for orgs with compliance constraints (legal, healthcare, finance) that currently rule out public SaaS entirely.
- **External users add a wrinkle**: if the org grants access to outside users (e.g. their own customers), need per-tenant data isolation *within* the on-prem deployment too — not just user-level scoping, but making sure org A's guest users can't see org B's guest users' data if the deployment ever serves more than one org's guest population. Simplest: one deployment = one org, full stop, and any external users are still just "users of that org's instance," fully isolated from any other org's separate deployment.
- **Deployment mechanics**: this is the natural fit for the existing `Dockerfile`/`docker-compose.yml` in the repo — package as a self-contained stack (app + vectorstore + optional local LLM runtime) the org's IT team runs on their own box. Licensing/updates become the operational question (how do orgs get patches — pull an image, or an update mechanism?).
- **Pricing shape**: likely a license/seat model per org rather than the metered per-call billing that made sense for idea 4's public SaaS, since local-model compute cost is now the org's own hardware, not ours.

This is arguably the deployment shape most consistent with the project's current "local-only, privacy-first" architecture decision (see `rag_architecture_decisions` memory) — it keeps that property while still adding multi-user support, which idea 4 (public SaaS) trades away.

## 6. Hybrid retrieval: BM25 + dense with Reciprocal Rank Fusion — **implemented**

> Built as `src/vectorstore/lexical.py` + `src/vectorstore/hybrid.py`. recall@5 rose from
> 0.80 to 0.88 (single-passage) and the combined benchmark score from 0.357 to 0.423. See
> `plan.md` "Measured baseline". The rest of this section is kept as the reasoning that led
> there.

Dense retrieval alone is weak on literal-phrase lookups, and the grading thresholds that compensate for it are fragile. Measured on the current 223-chunk corpus with `nomic-embed-text`:

| Query | Top-5 relevance scores |
| --- | --- |
| who said I love you and ran off | 0.5685 … 0.5218 |
| Ruby phoned Carla and said I love you | 0.6691 … 0.5455 |
| who is the Doctor | 0.6126 … 0.5963 |
| capital of Mongolia (off-topic) | 0.4454 … 0.4215 |
| quantum chromodynamics (off-topic) | 0.4531 … 0.4432 |

On-topic and off-topic scores are separated by roughly 0.07, and for the "I love you / ran off" query the correct chunk beats the best irrelevant chunk by only **+0.0156** cosine. `RELEVANCE_FLOOR` currently sits at 0.48 — about 0.027 above the observed noise ceiling. That works for this corpus, but any new document set shifts the distribution and the floor has to be re-tuned by hand.

BM25 over the same chunks, same query, ranks the correct chunk first with a wide margin (16.86 vs 13.01 for the runner-up) — precisely because the query is a literal-phrase lookup (`"I love you"`, `"ran off"`), which is dense retrieval's weak spot and lexical search's strength.

### Shape as built

Postgres already hosts the vectors via `PGVector`, so the lexical half needed no new infrastructure:

1. Add a `tsvector` column over chunk text plus a GIN index on the existing collection table.
2. Run `ts_rank_cd` full-text search alongside the vector search, retrieving top-k from each independently.
3. Fuse the two ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), rather than trying to normalize a cosine score against a BM25 score — the scales aren't comparable, ranks are.

The payoff landed differently than expected. `grade_node` did not become purely rank-based:
it keeps every full-text hit unconditionally and applies the cosine cutoff only to
dense-only documents. That works because Postgres returns *nothing* when no query term
matches, so an off-topic query yields an empty lexical list rather than a weak one — the
refusal signal comes from the miss, not from a second threshold.

`RELEVANCE_FLOOR` therefore still exists and still needs a corpus-specific value (swept to
0.56, see `tests/benchmark/run_sweep.py`). It governs a narrower decision than before, but
the hope of deleting it outright was not realised.

### Tested and rejected

- **nomic task prefixes** (`search_query:` / `search_document:`). Measured on this corpus, the gold-vs-best-noise gap *shrank* from 0.0156 to 0.0013. Not worth a re-ingest here, despite being the documented usage for the model.
- **Retrieval retry loop.** The comment in `src/rag/graph.py` is correct: re-running a deterministic, similarity-sorted retrieval can't surface a chunk the first pass missed. Only worth revisiting alongside query rewriting.

## 7. UI/UX backlog — deferred dashboard work

Backlog for the local-only RAG dashboard (FastAPI + LangGraph + pgvector + Ollama in `src/`, React 19 + Vite + Tailwind 4 + recharts in `ui/`). Ordered by priority.

**Already landed / in progress** (tracked elsewhere, not repeated below): streaming answers, query cancellation, a per-stage indicator, the transcript-shaped Ask page, per-answer actions, keyboard shortcuts, and the empty-collection state.

The theme running through everything here is the same: the pipeline already computes far more than it shows. Scores, rerank ordering, grader cutoffs, and token counts all exist at runtime and are thrown away at the API boundary or written only to a log line. Most of this roadmap is plumbing that information outward rather than new machinery.

---

### A. Rich sources

**Highest priority.** Everything else in the "explain the answer" family depends on this shape landing first.

Today `QueryResponse.sources` in `src/api/schemas.py` is `list[str]`, and `collect_sources` in `src/rag/citations.py` does exactly one thing — dedupe `doc.metadata["source"]` in rank order — discarding the rest of the document. Meanwhile `retrieve_node` in `src/rag/nodes.py` has already attached, per chunk:

- dense similarity under `hybrid.DENSE_SCORE_KEY` (`"relevance_score"`)
- lexical rank under `hybrid.LEXICAL_SCORE_KEY` (`"lexical_score"`)
- RRF fusion score under `hybrid.FUSION_SCORE_KEY` (`"fusion_score"`)
- cross-encoder score under `rerank.RERANK_SCORE_KEY` (`"rerank_score"`)

plus the full `page_content`. All of it is dropped one function later.

**Work**

- Add a `Source` model to `src/api/schemas.py` (url, title, snippet, `dense_score` / `lexical_score` / `fusion_score` / `rerank_score`, chunk id) and widen `QueryResponse.sources` and `HistoryEntry.sources` to `list[Source]`.
- Rework `collect_sources` to build those objects instead of strings. It currently dedupes by source URL; with chunk-level objects the dedupe key becomes the chunk, and multiple chunks from one document should group under one card.
- Thread the objects through `generate_node`'s return and `RAGState.sources` in `src/rag/graph.py`.
- Rewrite `ui/src/components/SourceList.jsx` from a link list into expandable cards: collapsed shows title + URL + score bars, expanded shows the chunk text with query terms highlighted.
- Add a "dropped by grader" section listing chunks that were retrieved but cut, with the score they got and the cutoff they missed. This is what makes a refusal explain itself — `grade_node` computes `cutoff = max(RELEVANCE_FLOOR, max(dense_scores) * RELEVANCE_RATIO)` (`RAG_RELEVANCE_FLOOR` default 0.56, `RAG_RELEVANCE_RATIO` default 0.9) and already records which bound decided it.

**DB implication.** `query_history.sources` is `JSONB` (`src/rag/history.py`), so richer objects need no migration. But existing rows hold plain strings — `_row_to_dict` and every reader, including `SourceList.jsx`, must tolerate both shapes indefinitely. Normalize old strings to `{url: s}` on read rather than branching in the UI.

*Effort: medium (touches schema, citations, nodes, graph, history reader, one component). Impact: high — unblocks B and F, and is the single biggest jump in perceived trustworthiness.*

---

### B. Retrieval inspector

A panel showing the whole retrieval funnel for one query: all ~20 fetched candidates (`FETCH_K`, from `RAG_FETCH_K`, default 20), how the reranker reordered them (before/after position, with movement arrows), grader pass/fail per chunk, and which chunks actually reached the prompt.

The data already exists. `retrieve_node` calls `tracing.detail(k=..., fetch_k=..., reranked=..., hybrid=..., scores=..., lexical_hits=..., sources=...)` and the rerank span records `fetched`/`kept`/`scores`; `grade_node` records `cutoff`, `bound` (`"floor"` vs `"ratio"`), `kept_lexical`, `kept_dense`, `kept`, `dropped`. All of it goes to the `rag.trace` logger and nowhere else.

**Work**

- `graph.ask` already wraps the invocation in `tracing.collect()` to extract span durations. Extend that: keep the collected span details, not just `duration_ms`, and return them on the response behind a flag (`debug: true` on `QueryRequest`) so the normal path stays lean.
- New view or a drawer on Ask, fed from the same response — no extra round trip.

Doubles as the debugging tool for tuning `RAG_FETCH_K` and the relevance thresholds, and as the most demo-able feature in the project.

*Effort: medium. Impact: high, especially for anyone tuning retrieval.*

---

### C. A/B playground

Run one query under two configurations side by side and diff the answers: llama3.2:3b vs qwen3:4b, hybrid on/off, rerank on/off, different `FETCH_K`.

The obstacle is not the UI. Those settings are process-level environment flags read at import time in `src/rag/nodes.py` and `src/vectorstore/` — `RAG_HYBRID` (via `hybrid_enabled()`), `RAG_FETCH_K`, `RAG_RELEVANCE_FLOOR`, `RAG_RELEVANCE_RATIO`, `RAG_RERANK` (via `rerank.rerank_enabled()`). Model choice is the one knob already per-query, validated against `AVAILABLE_MODELS`.

**Work**

- Add an optional config dict to `RAGState` in `src/rag/graph.py` and have `retrieve_node` / `grade_node` read `state.get(...)` with the module constant as fallback. That is the real work; the module-level defaults stay as the fallback so nothing else changes.
- Extend `QueryRequest` with an optional overrides object, validated and clamped in `src/api/routes/query.py`.
- Two-column Ask variant with a config picker per column.

Note the memory constraint: two models cannot be resident at once on a 4GB card, so an A/B across models runs sequentially and will pay a model-swap cost. Say so in the UI rather than letting it look hung.

*Effort: high (state threading is invasive). Impact: high for evaluation work, lower for daily use.*

---

### D. Closed eval loop

Ask and Benchmark are disconnected today: you can run a benchmark against the fixed question sets in `tests/benchmark/data/` (`single_passage_answer_questions.csv`, `multi_passage_answer_questions.csv`, `no_answer_questions.csv`), but nothing you observe while actually using the system feeds back into them.

**Work**

- Thumbs up/down on an answer, written against the history row's `id` — a new nullable `rating` column in `query_history` (`src/rag/history.py` already has a `_migrate` hook for additive columns).
- On thumbs-down, let the user type the answer they expected.
- An endpoint that appends `{question, expected_answer}` to the appropriate CSV under `tests/benchmark/data/`, routed by whether an answer was expected at all (a refusal that should not have refused belongs in a passage set; a hallucination on an unanswerable question belongs in `no_answer_questions.csv`).
- Surface the count of user-contributed rows in the Benchmark view so the next run visibly measures them.

*Effort: medium. Impact: high — turns ad-hoc observation into regression coverage, which is the whole point of having a benchmark.*

---

### E. Prompt budget warning

`src/rag/nodes.py` sets `NUM_CTX = 8192`, but `QWEN3_NUM_CTX = 3072` because qwen3:4b spills to CPU on this 4GB card above that (the comment records the measurement: 4096 already spills 6%, and CPU prefill on a RAG prompt measured 400s+). `_num_ctx_for` picks between them by model prefix.

The failure mode is silent. Ollama truncates from the *front* of the prompt — which is exactly where the highest-ranked context sits, since `format_context` lays documents out in rank order. The answer degrades without any error, and the code comments already flag this as a known risk.

**Work**

- Estimate prompt tokens in `generate_node`. `tracing.detail` already logs `prompt_chars`; a chars/4 estimate is adequate, and `_token_usage` pulls Ollama's real `prompt_eval_count` off the response for an exact figure after the fact.
- Return both the estimate and `_num_ctx_for(model)` on the response.
- Render a budget bar under the answer, and a visible warning when the estimate exceeds the window — naming which sources were probably cut, since after A those are individually identifiable.

*Effort: low. Impact: medium-high — this is a real silent correctness bug, not a nicety.*

---

### F. Corpus browser

Browse what has actually been ingested, at chunk level rather than file level. `ui/src/views/Ingest.jsx` shows an upload history (filename, sha256, size, document count) but nothing about the chunks themselves.

**Work**

- Paginated chunk listing per ingested file, reading from the pgvector collection table.
- Direct lexical search over chunks — `search()` in `src/vectorstore/lexical.py` already does Postgres full-text search over `doc_tsv`, so this is an endpoint over an existing function, not new retrieval code.
- Visualize chunk boundaries within a document, per splitter. The splitter list already comes from `GET /api/ingest/splitters`; showing where each one cuts makes the choice empirical instead of a guess.

*Effort: medium-high (new view plus pagination). Impact: medium — mostly valuable while tuning ingestion.*

---

### G. Shareable permalink

History entries already have an `id`, and `GET /api/history/{entry_id}` already exists (`src/api/routes/history.py:get_entry`). There is simply no `/ask/:id` route in `ui/src/App.jsx`, so a specific past answer cannot be linked to.

**Work**

- Add `<Route path="/ask/:id" element={<Ask />} />`, load the entry when the param is present, and render it read-only.
- Add `getHistoryEntry(id)` to `ui/src/api.js`.
- Link history rows to their permalink.

*Effort: low (an hour). Impact: low but disproportionate to cost — do it opportunistically.*

---

### H. Multi-turn conversations

`RAGState` in `src/rag/graph.py` carries `query`, `model`, `retrieved_docs`, `graded_docs`, `answer`, `sources` — no conversation history. Every question is independent, so "what about the second one?" retrieves against the literal string.

This needs query rewriting: condense (history + follow-up) into a standalone query before `retrieve_node` runs.

Note what `build_graph` says about itself — the retrieve → grade → generate path deliberately has no retry edge, because retrieval is deterministic and sorted by descending similarity, so re-running the same query at any k cannot surface a chunk the grader would accept. The comment's own conclusion is that a loop only makes sense alongside query rewriting. So both land together, or neither does: add a `rewrite` node, and only then does a grade → rewrite → retrieve retry edge have a reason to exist.

**Work**

- Add `messages` to `RAGState`; add a `rewrite` node in `src/rag/nodes.py` using the small model with a condense prompt in `src/rag/prompts.py`.
- Add a conversation id to `query_history` so a thread can be reconstructed.
- Ask view becomes a real thread rather than a list of independent exchanges.

*Effort: high. Impact: medium — genuinely changes how the thing feels to use, but it is the largest single item here and touches the graph's core invariant.*

---

### I. Smaller fixes

Ordered roughly by pain-to-effort ratio within each view.

**Ingest** (`ui/src/views/Ingest.jsx`)

- Drag-and-drop zone; the file picker is the only way in today.
- CSV preview before ingest: first few rows, detected text and `source_url` columns, total row count — needs a dry-run endpoint in `src/api/routes/ingest.py` that parses without writing.
- The progress bar renders `job.progress` as a width and nothing else; add counts (n of m) and an ETA from elapsed time.
- Replace the `window.confirm` delete with an inline two-step confirm.

**System** (`ui/src/views/System.jsx`)

- Chart history lives in `useState`, so navigating away and back resets it. Hoist to a module-level ring buffer outside the component.
- Add a sparkline per gauge, so the numbers have context without reading the big chart.
- The card's stated purpose — correlate a latency spike with a VRAM event — is not actually achievable, because the chart has no query markers. Overlay markers from query history (`created_at` + `latency_ms`) as recharts `ReferenceLine`s.
- `gpu.memory_pct > 85` only turns the gauge text red. Add a persistent banner, and detect the actual failure this predicts: a model spilled to CPU (see the `QWEN3_NUM_CTX` comment in `src/rag/nodes.py`).

**Benchmark** (`ui/src/views/Benchmark.jsx`)

- `ResultsTable` joins all metrics into one `·`-separated string per row. Give each metric its own column.
- Add a delta column against the previous run, colored by direction.
- Add a trend chart across runs — recharts is already a dependency for System.
- Record the run config (workers, sample, cache, model) on history rows, since a delta between runs with different configs is meaningless. `BenchmarkRequest` already carries all four; they just are not persisted with the result.

**Global**

- Errors render as a full-width red `Card` that shoves the layout down on appearance. Replace with a toast or inline placement.
- Manual dark-mode toggle — only the OS preference is honored today.
- Loading skeletons instead of blank panels.
- Mobile: header nav in `ui/src/App.jsx` overflows, and no table collapses.
- Add an `<ErrorBoundary>` around the routes. One throw in any view currently blanks the entire app.

*Effort: each is small to medium; the System ring buffer and the ErrorBoundary are the two highest-value quick wins.*

## Open questions to resolve before committing to any of the above

- Is the product positioned as self-hosted (privacy-first, current direction per architecture decisions) or hosted SaaS (idea 4)? These pull in different directions — worth an explicit decision before idea 4 gets built, since it changes the hardware-detection story (idea 1) fundamentally.
- What's the actual unit of "test case file" — same CSV schema as internal benchmark data, or does it need a friendlier upload format for external users?
- Does the semantic-similarity scorer (idea 2) replace or supplement the keyword-overlap metric — need both for backward comparability with existing benchmark data?
