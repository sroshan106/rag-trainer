import { useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { startBenchmark, benchmarkHistory, benchmarkModels, cancelJob, pollJob } from "../api.js";

export default function Benchmark() {
  const [workers, setWorkers] = useState(4);
  const [sample, setSample] = useState("");
  const [chunkSize, setChunkSize] = useState(10);
  const [useCache, setUseCache] = useState(true);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [stopping, setStopping] = useState(false);
  const cancelledRef = useRef(false);

  const running = job && (job.status === "running" || job.status === "pending");

  async function loadHistory() {
    try {
      setHistory(await benchmarkHistory());
    } catch {
      // history is a convenience panel -- a failed load shouldn't block the run form
    }
  }

  useEffect(() => {
    loadHistory();
    benchmarkModels()
      .then(({ models: available, default: fallback }) => {
        setModels(available);
        setModel(fallback);
      })
      .catch(() => {
        // an empty picker falls back to the server-side default model
      });
  }, []);

  // Stop polling if the view unmounts -- the run itself keeps going server-side.
  useEffect(() => () => {
    cancelledRef.current = true;
  }, []);

  async function onStart() {
    setError(null);
    setStopping(false);
    cancelledRef.current = false;
    try {
      const started = await startBenchmark({
        workers: Number(workers) || 4,
        sample: sample ? Number(sample) : null,
        use_cache: useCache,
        chunk_size: Number(chunkSize) || 10,
        model: model || null,
      });
      setJob(started);
      await pollJob(started.id, {
        onUpdate: setJob,
        isCancelled: () => cancelledRef.current,
      });
      loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setStopping(false);
    }
  }

  async function onStop() {
    if (!job) return;
    setStopping(true);
    try {
      // Cancellation lands between chunks, so the job stays "running" for a
      // few seconds after this returns -- polling continues either way.
      await cancelJob(job.id);
    } catch (err) {
      setError(err.message);
      setStopping(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Benchmark"
        subtitle="Runs the labeled question sets in interleaved chunks, so every suite reports numbers while the run is still going."
      >
        <div className="flex flex-wrap items-end gap-4 mb-3">
          <Field label="Model">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-2 py-1 text-sm"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Workers">
            <input
              type="number"
              min={1}
              max={32}
              value={workers}
              onChange={(e) => setWorkers(e.target.value)}
              className="w-20 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-2 py-1 text-sm"
            />
          </Field>
          <Field label="Sample per suite">
            <input
              type="number"
              min={1}
              placeholder="all"
              value={sample}
              onChange={(e) => setSample(e.target.value)}
              className="w-24 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-2 py-1 text-sm"
            />
          </Field>
          <Field label="Chunk size">
            <input
              type="number"
              min={1}
              max={200}
              value={chunkSize}
              onChange={(e) => setChunkSize(e.target.value)}
              className="w-20 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-2 py-1 text-sm"
            />
          </Field>
          <label className="flex items-center gap-2 text-sm pb-1.5">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
            />
            Use cache
          </label>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onStart}
            disabled={running}
            className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {running ? "Running..." : "Run benchmark"}
          </button>
          {running && (
            <button
              onClick={onStop}
              disabled={stopping}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              {stopping ? "Stopping..." : "Stop"}
            </button>
          )}
        </div>
      </Card>

      {error && (
        <Card title="Error" className="border-red-300 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </Card>
      )}

      {job && (
        <Card title="Current run">
          <div className="flex items-center gap-3 mb-3">
            <StatusBadge status={job.status} />
            <span className="text-xs text-neutral-500">{job.message}</span>
          </div>
          {running && <ProgressBar value={job.progress} />}
          {/* Rendered at every status, not just "done": the job publishes
              running totals after each chunk, which is the whole point. */}
          {job.result && <ResultsTable results={job.result} partial={job.status !== "done"} />}
          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-red-600 dark:text-red-400 whitespace-pre-wrap">
              {job.error}
            </pre>
          )}
        </Card>
      )}

      <Card title="History" subtitle="Past runs from this session, most recent first.">
        {history.length === 0 ? (
          <p className="text-sm text-neutral-500">No runs yet.</p>
        ) : (
          <div className="flex flex-col gap-4">
            {history.map((h) => (
              <div key={h.id} className="border-t border-neutral-200 dark:border-neutral-800 pt-3 first:border-0 first:pt-0">
                <div className="flex items-center gap-3 mb-2">
                  <StatusBadge status={h.status} />
                  <span className="text-xs text-neutral-500">
                    {new Date(h.created_at * 1000).toLocaleString()}
                  </span>
                </div>
                {h.result && <ResultsTable results={h.result} partial={h.status !== "done"} />}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-neutral-500">{label}</span>
      {children}
    </label>
  );
}

function ProgressBar({ value }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-neutral-200 dark:bg-neutral-800 mb-3">
      <div
        className="h-full rounded-full bg-blue-500 transition-[width] duration-500"
        style={{ width: `${Math.round((value ?? 0) * 100)}%` }}
      />
    </div>
  );
}

function ResultsTable({ results, partial = false }) {
  return (
    <div className="overflow-x-auto">
      {partial && (
        <p className="text-xs text-amber-600 dark:text-amber-400 mb-2">
          Partial — scored over the questions answered so far.
        </p>
      )}
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-neutral-500 border-b border-neutral-200 dark:border-neutral-800">
            <th className="py-1 pr-4">Suite</th>
            <th className="py-1 pr-4">n</th>
            <th className="py-1 pr-4">Metrics</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.name} className="border-b border-neutral-100 dark:border-neutral-900">
              <td className="py-1.5 pr-4 font-mono text-xs">{r.name}</td>
              <td className="py-1.5 pr-4">{r.n}</td>
              <td className="py-1.5 pr-4 text-xs">
                {Object.entries(r)
                  .filter(([k]) => k !== "name" && k !== "n")
                  .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(2) : v}`)
                  .join("  ·  ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
