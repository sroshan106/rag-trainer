import { Fragment, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import {
  startBenchmark,
  benchmarkHistory,
  benchmarkModels,
  activeBenchmark,
  cancelJob,
  pollJob,
} from "../api.js";

export default function Benchmark() {
  const [workers, setWorkers] = useState(4);
  const [sample, setSample] = useState("");
  const [chunkSize, setChunkSize] = useState(10);
  const [useCache, setUseCache] = useState(true);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [stopping, setStopping] = useState(false);
  const [expandedIds, setExpandedIds] = useState(new Set());
  const cancelledRef = useRef(false);

  function toggleExpand(id) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const running = job && (job.status === "running" || job.status === "pending");

  async function loadHistory() {
    try {
      setHistory(await benchmarkHistory());
    } catch {
      // history is a convenience panel -- a failed load shouldn't block the run form
    }
  }

  useEffect(() => {
    cancelledRef.current = false;
    let ignore = false;

    loadHistory();
    benchmarkModels()
      .then(({ models: available }) => {
        if (ignore) return;
        setModels(available);
        setModelsLoaded(true);
        // No server default -- auto-pick the first installed model so a run
        // still needs one explicit click, not a manual selection every time.
        setModel((prev) => prev || available[0] || "");
      })
      .catch(() => {
        // No API yet -- the picker stays empty.
      });

    activeBenchmark()
      .then((runningJob) => {
        if (ignore || !runningJob) return;
        setJob(runningJob);
        return pollJob(runningJob.id, {
          onUpdate: (updated) => {
            if (!cancelledRef.current) setJob(updated);
          },
          isCancelled: () => cancelledRef.current,
        }).then((finalJob) => {
          if (!cancelledRef.current) loadHistory();
        });
      })
      .catch(() => {
        // No active benchmark
      });

    return () => {
      ignore = true;
      cancelledRef.current = true;
    };
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
      {modelsLoaded && models.length === 0 && (
        <Card className="border-amber-300 dark:border-amber-800">
          <p className="text-sm">
            No chat model is downloaded yet, so there is nothing to benchmark against.{" "}
            <Link to="/settings" className="font-medium underline">
              Download one in Settings
            </Link>{" "}
            first.
          </p>
        </Card>
      )}

      <Card
        title="Benchmark"
        subtitle="Runs the labeled question sets in interleaved chunks, so every suite reports numbers while the run is still going."
      >
        <div className="flex flex-wrap items-end gap-4 mb-3">
          <Field label="Model">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-1 text-sm dark:[color-scheme:dark]"
            >
              {models.map((m) => (
                <option key={m} value={m} className="bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100">
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
              className="w-20 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-1 text-sm"
            />
          </Field>
          <Field label="Sample per suite">
            <input
              type="number"
              min={1}
              placeholder="all"
              value={sample}
              onChange={(e) => setSample(e.target.value)}
              className="w-24 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-1 text-sm"
            />
          </Field>
          <Field label="Chunk size">
            <input
              type="number"
              min={1}
              max={200}
              value={chunkSize}
              onChange={(e) => setChunkSize(e.target.value)}
              className="w-20 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-1 text-sm"
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
            disabled={running || !model}
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
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-3">
              <StatusBadge status={job.status} />
              <span className="text-xs text-neutral-500">{job.message}</span>
            </div>
            <RunParams params={job.params} />
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
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-neutral-500 border-b border-neutral-200 dark:border-neutral-800">
                  <th className="pb-2 pr-4 font-medium">Date & Time</th>
                  <th className="pb-2 pr-4 font-medium">Model</th>
                  <th className="pb-2 pr-4 font-medium">Status</th>
                  <th className="pb-2 pr-4 font-medium">Configuration</th>
                  <th className="pb-2 pr-4 font-medium">Results Summary</th>
                  <th className="pb-2 text-right font-medium"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
                {history.map((h) => {
                  const isExpanded = expandedIds.has(h.id);
                  return (
                    <Fragment key={h.id}>
                      <tr
                        onClick={() => toggleExpand(h.id)}
                        className="hover:bg-neutral-50 dark:hover:bg-neutral-900/60 cursor-pointer transition-colors"
                      >
                        <td className="py-2.5 pr-4 text-xs whitespace-nowrap text-neutral-500">
                          {new Date(h.created_at * 1000).toLocaleString()}
                        </td>
                        <td className="py-2.5 pr-4 whitespace-nowrap">
                          <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800">
                            {h.params?.model || "default"}
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 whitespace-nowrap">
                          <StatusBadge status={h.status} />
                        </td>
                        <td className="py-2.5 pr-4">
                          <RunParams params={h.params} />
                        </td>
                        <td className="py-2.5 pr-4">
                          {formatMetricsSummary(h.result) || (
                            <span className="text-xs text-neutral-400 italic">
                              {h.message || "—"}
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 text-right whitespace-nowrap">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpand(h.id);
                            }}
                            className="text-xs text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200 underline"
                          >
                            {isExpanded ? "Hide details" : "View details"}
                          </button>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr className="bg-neutral-50/50 dark:bg-neutral-900/30">
                          <td colSpan={6} className="px-4 py-3">
                            <div className="flex flex-col gap-2">
                              {h.message && h.status !== "done" && (
                                <div className="text-xs text-neutral-500 italic mb-1">
                                  Status message: {h.message}
                                </div>
                              )}
                              {h.result ? (
                                <ResultsTable results={h.result} partial={h.status !== "done"} />
                              ) : (
                                <p className="text-xs text-neutral-400">No score data available.</p>
                              )}
                              {h.status === "failed" && h.error && (
                                <pre className="mt-2 text-xs text-red-600 dark:text-red-400 whitespace-pre-wrap">
                                  {h.error}
                                </pre>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function formatMetricsSummary(result) {
  if (!result || !Array.isArray(result) || result.length === 0) return null;
  const suitesWithData = result.filter((r) => r.n > 0);
  if (suitesWithData.length === 0) return <span className="text-xs text-neutral-400">0 questions answered</span>;

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {suitesWithData.map((r) => {
        const shortName = r.name
          .replace("_passage_answer_questions.csv", "")
          .replace("_answer_questions.csv", "")
          .replace(".csv", "")
          .replace(/_/g, " ");

        const metricEntries = Object.entries(r).filter(([k]) => k !== "name" && k !== "n");

        return (
          <span key={r.name} className="inline-flex items-center gap-1">
            <span className="font-medium text-neutral-700 dark:text-neutral-300 capitalize">{shortName}:</span>
            <span className="text-neutral-500 dark:text-neutral-400">
              {metricEntries
                .map(([k, v]) => {
                  const label = k
                    .replace("@5", "")
                    .replace("mean_answer_overlap", "overlap")
                    .replace("pass_rate(overlap>=0.3)", "pass")
                    .replace("correct_refusal_rate", "refusal");
                  const val = typeof v === "number" ? (v <= 1 ? `${(v * 100).toFixed(0)}%` : v.toFixed(2)) : v;
                  return `${label}: ${val}`;
                })
                .join(", ")}
            </span>
          </span>
        );
      })}
    </div>
  );
}

function RunParams({ params }) {
  if (!params) return null;
  const { model, workers, sample, chunk_size, use_cache } = params;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      {model && (
        <span className="font-mono px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800 font-medium">
          {model}
        </span>
      )}
      {workers != null && (
        <span className="px-2 py-0.5 rounded bg-neutral-100 text-neutral-600 border border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700">
          {workers} {workers === 1 ? "worker" : "workers"}
        </span>
      )}
      <span className="px-2 py-0.5 rounded bg-neutral-100 text-neutral-600 border border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700">
        sample: {sample != null && sample !== "" ? sample : "all"}
      </span>
      {chunk_size != null && (
        <span className="px-2 py-0.5 rounded bg-neutral-100 text-neutral-600 border border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700">
          chunk: {chunk_size}
        </span>
      )}
      {use_cache != null && (
        <span className="px-2 py-0.5 rounded bg-neutral-100 text-neutral-600 border border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700">
          {use_cache ? "cache" : "no-cache"}
        </span>
      )}
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
