import { useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { startBenchmark, benchmarkHistory, pollJob } from "../api.js";

export default function Benchmark() {
  const [workers, setWorkers] = useState(8);
  const [sample, setSample] = useState("");
  const [useCache, setUseCache] = useState(true);
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const cancelledRef = useRef(false);

  const running = job && job.status === "running";

  async function loadHistory() {
    try {
      setHistory(await benchmarkHistory());
    } catch {
      // history is a convenience panel -- a failed load shouldn't block the run form
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function onStart() {
    setError(null);
    cancelledRef.current = false;
    try {
      const started = await startBenchmark({
        workers: Number(workers) || 8,
        sample: sample ? Number(sample) : null,
        use_cache: useCache,
      });
      setJob(started);
      await pollJob(started.id, {
        onUpdate: setJob,
        isCancelled: () => cancelledRef.current,
      });
      loadHistory();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Benchmark"
        subtitle="Runs tests.benchmark.run_benchmark.run_all as a background job against the labeled question sets."
      >
        <div className="flex flex-wrap items-end gap-4 mb-3">
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
          <label className="flex items-center gap-2 text-sm pb-1.5">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
            />
            Use cache
          </label>
        </div>
        <button
          onClick={onStart}
          disabled={running}
          className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {running ? "Running..." : "Run benchmark"}
        </button>
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
          {job.status === "done" && job.result && <ResultsTable results={job.result} />}
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
                {h.status === "done" && h.result && <ResultsTable results={h.result} />}
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

function ResultsTable({ results }) {
  return (
    <div className="overflow-x-auto">
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
