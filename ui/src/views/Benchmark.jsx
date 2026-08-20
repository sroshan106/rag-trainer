import { Fragment, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Play,
  Square,
  ChevronDown,
  ChevronUp,
  Sliders,
  AlertCircle,
  AlertTriangle,
} from "lucide-react";

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
      // Ignore
    }
  }

  useEffect(() => {
    cancelledRef.current = false;
    let ignore = false;

    benchmarkHistory().then((data) => {
      if (!ignore) setHistory(data);
    }).catch(() => {});
    benchmarkModels()
      .then(({ models: available }) => {
        if (ignore) return;
        setModels(available);
        setModelsLoaded(true);
        setModel((prev) => prev || available[0] || "");
      })
      .catch(() => {});

    activeBenchmark()
      .then((runningJob) => {
        if (ignore || !runningJob) return;
        setJob(runningJob);
        return pollJob(runningJob.id, {
          onUpdate: (updated) => {
            if (!cancelledRef.current) setJob(updated);
          },
          isCancelled: () => cancelledRef.current,
        }).then(() => {
          if (!cancelledRef.current) loadHistory();
        });
      })
      .catch(() => {});

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
      await cancelJob(job.id);
    } catch (err) {
      setError(err.message);
      setStopping(false);
    }
  }

  function applyPreset(presetType) {
    if (presetType === "quick") {
      setWorkers(2);
      setSample("5");
      setChunkSize(5);
      setUseCache(true);
    } else if (presetType === "full") {
      setWorkers(8);
      setSample("");
      setChunkSize(20);
      setUseCache(true);
    }
  }

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-100">Benchmark Suite</h1>
        <p className="text-sm text-slate-400 mt-1">
          Evaluate pipeline answer overlap, refusal precision, and latencies across test suites
        </p>
      </div>

      {modelsLoaded && models.length === 0 && (
        <div className="rounded-xl border border-amber-800/80 bg-amber-950/30 p-4 text-sm text-amber-300 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p>
              No chat model is downloaded yet.{" "}
              <Link to="/settings" className="font-semibold underline hover:text-amber-200">
                Download one in Settings
              </Link>{" "}
              first to run benchmarks.
            </p>
          </div>
        </div>
      )}

      {/* Config Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Sliders className="h-4 w-4 text-blue-400" />
            <span>Benchmark Configuration</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Presets:</span>
            <button
              type="button"
              onClick={() => applyPreset("quick")}
              disabled={running}
              className="px-2.5 py-1 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors"
            >
              ⚡ Quick Test (5)
            </button>
            <button
              type="button"
              onClick={() => applyPreset("full")}
              disabled={running}
              className="px-2.5 py-1 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors"
            >
              🔬 Full Eval
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-5">
          <Field label="Model">
            <div className="relative">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={running}
                className="w-full appearance-none rounded-lg border border-slate-700/80 bg-slate-900/90 text-slate-200 px-3 py-2 pr-8 text-xs font-mono font-medium focus:outline-none focus:border-blue-500 transition-colors cursor-pointer"
              >
                {models.map((m) => (
                  <option key={m} value={m} className="bg-[#111726] text-slate-200">
                    {m}
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                <span className="text-[10px]">▼</span>
              </div>
            </div>
          </Field>

          <Field label="Parallel Workers">
            <input
              type="number"
              min={1}
              max={32}
              value={workers}
              disabled={running}
              onChange={(e) => setWorkers(e.target.value)}
              className="w-full rounded-lg border border-slate-700/80 bg-slate-900/90 text-slate-200 px-3 py-2 text-xs font-medium focus:outline-none focus:border-blue-500"
            />
          </Field>

          <Field label="Sample Size">
            <input
              type="number"
              min={1}
              placeholder="all questions"
              value={sample}
              disabled={running}
              onChange={(e) => setSample(e.target.value)}
              className="w-full rounded-lg border border-slate-700/80 bg-slate-900/90 text-slate-200 px-3 py-2 text-xs font-medium focus:outline-none focus:border-blue-500"
            />
          </Field>

          <Field label="Interleaved Chunk Size">
            <input
              type="number"
              min={1}
              max={200}
              value={chunkSize}
              disabled={running}
              onChange={(e) => setChunkSize(e.target.value)}
              className="w-full rounded-lg border border-slate-700/80 bg-slate-900/90 text-slate-200 px-3 py-2 text-xs font-medium focus:outline-none focus:border-blue-500"
            />
          </Field>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-800/80">
          <label className="flex items-center gap-2 text-xs font-medium text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useCache}
              disabled={running}
              onChange={(e) => setUseCache(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-blue-600 focus:ring-0"
            />
            <span>Use caching (skip repeating queries if cached)</span>
          </label>

          <div className="flex items-center gap-2">
            <button
              onClick={onStart}
              disabled={running || !model}
              className="flex items-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              <Play className="h-4 w-4" />
              <span>{running ? "Running..." : "Run Benchmark"}</span>
            </button>
            {running && (
              <button
                onClick={onStop}
                disabled={stopping}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 text-sm font-medium transition-colors"
              >
                <Square className="h-3.5 w-3.5" />
                <span>{stopping ? "Stopping..." : "Stop"}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-800/80 bg-rose-950/40 p-4 text-sm text-rose-300 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Error: </span>
            {error}
          </div>
        </div>
      )}

      {/* Current Run Card */}
      {job && (
        <Card title="Current Run" className="border-blue-900/60 bg-blue-950/20">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-3">
              <StatusBadge status={job.status} />
              <span className="text-xs font-medium text-slate-300">{job.message}</span>
            </div>
            <RunParams params={job.params} />
          </div>

          {running && <ProgressBar value={job.progress} />}
          {job.result && <ResultsTable results={job.result} partial={job.status !== "done"} />}
          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-rose-400 bg-rose-950/40 p-3 rounded-lg border border-rose-900/60 whitespace-pre-wrap font-mono">
              {job.error}
            </pre>
          )}
        </Card>
      )}

      {/* History Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
        <h2 className="text-base font-bold text-slate-100 mb-1">Benchmark History</h2>
        <p className="text-xs text-slate-400 mb-4">Past evaluations from this session, newest first</p>

        {history.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-xs">
            No benchmark runs yet. Configure parameters above and click "Run Benchmark".
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400 border-b border-slate-800 pb-2">
                  <th className="pb-2.5 pr-4 font-semibold">Date & Time</th>
                  <th className="pb-2.5 pr-4 font-semibold">Model</th>
                  <th className="pb-2.5 pr-4 font-semibold">Status</th>
                  <th className="pb-2.5 pr-4 font-semibold">Config</th>
                  <th className="pb-2.5 pr-4 font-semibold">Results Summary</th>
                  <th className="pb-2.5 text-right font-semibold"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {history.map((h) => {
                  const isExpanded = expandedIds.has(h.id);
                  return (
                    <Fragment key={h.id}>
                      <tr
                        onClick={() => toggleExpand(h.id)}
                        className="hover:bg-slate-900/60 cursor-pointer transition-colors"
                      >
                        <td className="py-3 pr-4 whitespace-nowrap text-slate-400">
                          {new Date(h.created_at * 1000).toLocaleString()}
                        </td>
                        <td className="py-3 pr-4 whitespace-nowrap">
                          <span className="font-mono font-medium px-2 py-0.5 rounded-md bg-blue-950/60 text-blue-300 border border-blue-800/60">
                            {h.params?.model || "default"}
                          </span>
                        </td>
                        <td className="py-3 pr-4 whitespace-nowrap">
                          <StatusBadge status={h.status} />
                        </td>
                        <td className="py-3 pr-4">
                          <RunParams params={h.params} />
                        </td>
                        <td className="py-3 pr-4">
                          {formatMetricsSummary(h.result) || (
                            <span className="text-slate-500 italic">{h.message || "—"}</span>
                          )}
                        </td>
                        <td className="py-3 text-right whitespace-nowrap">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpand(h.id);
                            }}
                            className="text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1"
                          >
                            <span>{isExpanded ? "Hide" : "Details"}</span>
                            {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                          </button>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr className="bg-slate-900/30">
                          <td colSpan={6} className="px-4 py-3.5">
                            <div className="flex flex-col gap-2">
                              {h.message && h.status !== "done" && (
                                <div className="text-xs text-slate-400 italic mb-1">
                                  Status message: {h.message}
                                </div>
                              )}
                              {h.result ? (
                                <ResultsTable results={h.result} partial={h.status !== "done"} />
                              ) : (
                                <p className="text-xs text-slate-500">No score data recorded.</p>
                              )}
                              {h.status === "failed" && h.error && (
                                <pre className="mt-2 text-xs text-rose-400 bg-rose-950/40 p-3 rounded-lg border border-rose-900/60 whitespace-pre-wrap font-mono">
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
      </div>
    </div>
  );
}

function formatMetricsSummary(result) {
  if (!result || !Array.isArray(result) || result.length === 0) return null;
  const suitesWithData = result.filter((r) => r.n > 0);
  if (suitesWithData.length === 0) return <span className="text-xs text-slate-500">0 answered</span>;

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
          <span key={r.name} className="inline-flex items-center gap-1.5 bg-slate-900/80 px-2 py-0.5 rounded-md border border-slate-800">
            <span className="font-semibold text-slate-300 capitalize">{shortName}:</span>
            <span className="text-slate-400">
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
  const { workers, sample, chunk_size, use_cache } = params;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
      {workers != null && (
        <span className="px-2 py-0.5 rounded-md bg-slate-900/80 text-slate-300 border border-slate-800">
          {workers}w
        </span>
      )}
      <span className="px-2 py-0.5 rounded-md bg-slate-900/80 text-slate-300 border border-slate-800">
        sample: {sample != null && sample !== "" ? sample : "all"}
      </span>
      {chunk_size != null && (
        <span className="px-2 py-0.5 rounded-md bg-slate-900/80 text-slate-300 border border-slate-800">
          chunk: {chunk_size}
        </span>
      )}
      {use_cache != null && (
        <span className="px-2 py-0.5 rounded-md bg-slate-900/80 text-slate-300 border border-slate-800">
          {use_cache ? "cache" : "no-cache"}
        </span>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-slate-400">{label}</span>
      {children}
    </label>
  );
}

function ProgressBar({ value }) {
  return (
    <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden mb-3">
      <div
        className="h-full rounded-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-300"
        style={{ width: `${Math.round((value ?? 0) * 100)}%` }}
      />
    </div>
  );
}

function ResultsTable({ results, partial = false }) {
  return (
    <div className="overflow-x-auto">
      {partial && (
        <p className="text-xs text-amber-400 font-medium mb-2">
          Partial — scored over the questions answered so far.
        </p>
      )}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-400 border-b border-slate-800 pb-2">
            <th className="py-2 pr-4 font-semibold">Test Suite</th>
            <th className="py-2 pr-4 font-semibold">n</th>
            <th className="py-2 pr-4 font-semibold">Metrics</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 font-mono">
          {results.map((r) => (
            <tr key={r.name} className="hover:bg-slate-900/40">
              <td className="py-2 pr-4 font-semibold text-slate-200">{r.name}</td>
              <td className="py-2 pr-4 text-slate-400">{r.n}</td>
              <td className="py-2 pr-4 text-slate-300">
                {Object.entries(r)
                  .filter(([k]) => k !== "name" && k !== "n")
                  .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(2) : v}`)
                  .join("  •  ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

