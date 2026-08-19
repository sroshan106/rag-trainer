// Thin fetch wrappers over the FastAPI routes in src/api/routes/. Kept
// framework-free (no axios/react-query) since the frontend dependency list
// is meant to stay minimal -- four views don't need a data-fetching library.

const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (res.status === 204) return null;
  return res.json();
}

export function runQuery(query) {
  return request("/query", { method: "POST", body: JSON.stringify({ query }) });
}

export function startIngest() {
  return request("/ingest", { method: "POST" });
}

export function startBenchmark({ workers = 8, sample = null, use_cache = true } = {}) {
  return request("/benchmark", {
    method: "POST",
    body: JSON.stringify({ workers, sample, use_cache }),
  });
}

export function benchmarkHistory() {
  return request("/benchmark/history");
}

export function getJob(id) {
  return request(`/jobs/${id}`);
}

export function getMetrics() {
  return request("/metrics");
}

// Polls a job until it reaches a terminal state, calling onUpdate after
// every poll. Returns the final job. Caller passes an AbortSignal-like
// `cancelled` ref to stop early (e.g. component unmount).
export async function pollJob(id, { onUpdate, intervalMs = 1000, isCancelled } = {}) {
  for (;;) {
    if (isCancelled?.()) return null;
    const job = await getJob(id);
    onUpdate?.(job);
    if (job.status === "done" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export function metricsStreamUrl() {
  return `${BASE}/metrics/stream`;
}
