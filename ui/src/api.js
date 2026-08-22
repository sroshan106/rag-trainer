const BASE = "/api";

async function errorFrom(res) {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail ?? detail;
  } catch {}
  return new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
}

async function request(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const headers = isForm ? {} : { "Content-Type": "application/json" };
  const res = await fetch(`${BASE}${path}`, { headers, ...options });
  if (!res.ok) throw await errorFrom(res);
  return res.status === 204 ? null : res.json();
}

const postJson = (path, data) => request(path, { method: "POST", body: JSON.stringify(data) });
const postForm = (path, form) => request(path, { method: "POST", body: form });
const del = (path) => request(path, { method: "DELETE" });

export const runQuery = (query, model = null) => postJson("/query", { query, model });
export const queryModels = () => request("/query/models");
export const collectionStatus = () => request("/query/collection");

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
    for (;;) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) break;
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const data = frame
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("\n");
      if (data) onEvent?.(JSON.parse(data));
    }
  }
}

export const ingestSplitters = () => request("/ingest/splitters");
export const activeIngest = () => request("/ingest/active");
export const ingestHistory = () => request("/ingest/history");
export const deleteIngestedFile = (id) => del(`/ingest/files/${id}`);

export function uploadAndIngest(file, splitter = null, indexColumns = null, citationColumns = null) {
  const form = new FormData();
  form.append("file", file);
  if (splitter) form.append("splitter", splitter);
  if (indexColumns && indexColumns.length > 0) {
    form.append("index_columns", JSON.stringify(indexColumns));
  }
  if (citationColumns && citationColumns.length > 0) {
    form.append("citation_columns", JSON.stringify(citationColumns));
  }
  return postForm("/ingest/upload", form);
}

export const documentMeta = (fileId) => request(`/documents/${fileId}`);
export const documentUnits = (fileId, { offset = 0, limit = 50 } = {}) =>
  request(`/documents/${fileId}/units?offset=${offset}&limit=${limit}`);
export const documentUnit = (fileId, index) =>
  request(`/documents/${fileId}/units/${index}`);

export const startBenchmark = ({
  workers = 4,
  sample = null,
  use_cache = true,
  chunk_size = 10,
  model = null,
  test_files = null,
} = {}) => postJson("/benchmark", { workers, sample, use_cache, chunk_size, model, test_files });

export const benchmarkModels = () => request("/benchmark/models");
export const compareQuery = (query, model = null) => postJson("/benchmark/compare", { query, model });
export const getBenchmarkTestFiles = () => request("/benchmark/test-files");

export function uploadBenchmarkTestFile(file, mappings = {}) {
  const form = new FormData();
  form.append("file", file);
  if (mappings.question_col) form.append("question_col", mappings.question_col);
  if (mappings.answer_col) form.append("answer_col", mappings.answer_col);
  if (mappings.doc_index_col) form.append("doc_index_col", mappings.doc_index_col);
  return postForm("/benchmark/test-files/upload", form);
}

export const deleteBenchmarkTestFile = (id) => del(`/benchmark/test-files/${id}`);
export const benchmarkHistory = () => request("/benchmark/history");
export const activeBenchmark = () => request("/benchmark/active");

export const queryHistory = (limit = 50) => request(`/history?limit=${limit}`);
export const deleteHistoryEntry = (id) => del(`/history/${id}`);
export const clearHistory = () => del("/history");

export const getJob = (id) => request(`/jobs/${id}`);
export const cancelJob = (id) => postJson(`/jobs/${id}/cancel`, {});

export const listModelCatalog = () => request("/models");
export const pullModel = (model) => postJson("/models/pull", { model });
export const pullHistory = () => request("/models/pull/history");
export const deleteModel = (model) => del(`/models/${encodeURIComponent(model)}`);
export const getMetrics = () => request("/metrics");
export const getLogs = (limit = 100) => request(`/metrics/logs?limit=${limit}`);

const TERMINAL_STATUSES = ["done", "failed", "cancelled"];

export async function pollJob(id, { onUpdate, intervalMs = 1000, isCancelled } = {}) {
  for (;;) {
    if (isCancelled?.()) return null;
    const job = await getJob(id);
    onUpdate?.(job);
    if (TERMINAL_STATUSES.includes(job.status)) return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export const metricsStreamUrl = () => `${BASE}/metrics/stream`;
