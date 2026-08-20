// Thin fetch wrappers over the FastAPI routes in src/api/routes/. Kept
// framework-free (no axios/react-query) since the frontend dependency list
// is meant to stay minimal -- four views don't need a data-fetching library.

const BASE = "/api";

// FastAPI puts the message in `detail`; anything else (a proxy error page,
// a dropped connection) only has a status line to report.
async function errorFrom(res) {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail ?? detail;
  } catch {
    // response wasn't JSON -- fall back to statusText
  }
  return new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
}

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw await errorFrom(res);
  if (res.status === 204) return null;
  return res.json();
}

export function runQuery(query, model = null) {
  return request("/query", { method: "POST", body: JSON.stringify({ query, model }) });
}

export function queryModels() {
  return request("/query/models");
}

export function collectionStatus() {
  return request("/query/collection");
}

// Streaming counterpart to runQuery. Not an EventSource: that is GET-only, and
// a question doesn't belong in a URL -- so the stream is a POST whose body is
// read and parsed here. `signal` aborts the request, which the server sees as
// a disconnect and records as a cancelled query.
//
// onEvent receives the decoded event objects in order: stage, token..., then
// done or error. See src/rag/graph.py:ask_stream for their shape.
export async function streamQuery(query, { model = null, signal, onEvent } = {}) {
  const res = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, model }),
    signal,
  });
  if (!res.ok) throw await errorFrom(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line. Line endings are CRLF from
    // sse-starlette but LF from anything else in front of it, so both count.
    for (;;) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) break;
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (data) onEvent?.(JSON.parse(data));
    }
  }
}

export function ingestSplitters() {
  return request("/ingest/splitters");
}

// Multipart, so no Content-Type header -- the browser must set the boundary.
export async function uploadAndIngest(file, splitter = null) {
  const body = new FormData();
  body.append("file", file);
  if (splitter) body.append("splitter", splitter);
  const res = await fetch(`${BASE}/ingest/upload`, { method: "POST", body });
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

export function activeIngest() {
  return request("/ingest/active");
}

export function ingestHistory() {
  return request("/ingest/history");
}

export function deleteIngestedFile(id) {
  return request(`/ingest/files/${id}`, { method: "DELETE" });
}

export function startBenchmark({
  workers = 4,
  sample = null,
  use_cache = true,
  chunk_size = 10,
  model = null,
} = {}) {
  return request("/benchmark", {
    method: "POST",
    body: JSON.stringify({ workers, sample, use_cache, chunk_size, model }),
  });
}

export function benchmarkModels() {
  return request("/benchmark/models");
}

export function benchmarkHistory() {
  return request("/benchmark/history");
}

export function queryHistory(limit = 50) {
  return request(`/history?limit=${limit}`);
}

export function deleteHistoryEntry(id) {
  return request(`/history/${id}`, { method: "DELETE" });
}

export function clearHistory() {
  return request("/history", { method: "DELETE" });
}

export function getJob(id) {
  return request(`/jobs/${id}`);
}

// Cooperative -- the job keeps reporting "running" until it unwinds, so the
// caller should keep polling rather than assume this took effect immediately.
export function cancelJob(id) {
  return request(`/jobs/${id}/cancel`, { method: "POST" });
}

export function listModelCatalog() {
  return request("/models");
}

export function pullModel(model) {
  return request("/models/pull", { method: "POST", body: JSON.stringify({ model }) });
}

export function pullHistory() {
  return request("/models/pull/history");
}

export function getMetrics() {
  return request("/metrics");
}

const TERMINAL_STATUSES = ["done", "failed", "cancelled"];

// Polls a job until it reaches a terminal state, calling onUpdate after
// every poll. Returns the final job. Caller passes an AbortSignal-like
// `cancelled` ref to stop early (e.g. component unmount).
export async function pollJob(id, { onUpdate, intervalMs = 1000, isCancelled } = {}) {
  for (;;) {
    if (isCancelled?.()) return null;
    const job = await getJob(id);
    onUpdate?.(job);
    if (TERMINAL_STATUSES.includes(job.status)) return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export function metricsStreamUrl() {
  return `${BASE}/metrics/stream`;
}
