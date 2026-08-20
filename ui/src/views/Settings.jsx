import { useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { listModelCatalog, pullModel, pollJob } from "../api.js";

// All three model kinds are optional at any given moment -- nothing is
// force-downloaded at container startup anymore (see docker-compose.yml).
// This view is the only place a download actually happens.
export default function Settings() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  // Keyed by model name so several rows can download independently and each
  // shows its own progress bar / status.
  const [jobs, setJobs] = useState({});
  const cancelledRef = useRef(false);

  function refresh() {
    listModelCatalog().then(setData).catch((err) => setError(err.message));
  }

  useEffect(() => {
    cancelledRef.current = false;
    refresh();
    return () => {
      cancelledRef.current = true;
    };
  }, []);

  function download(model) {
    setError(null);
    pullModel(model)
      .then((job) => {
        setJobs((prev) => ({ ...prev, [model]: job }));
        return pollJob(job.id, {
          onUpdate: (j) => setJobs((prev) => ({ ...prev, [model]: j })),
          isCancelled: () => cancelledRef.current,
        });
      })
      .then((finalJob) => {
        if (finalJob?.status === "done") refresh();
      })
      .catch((err) => setError(err.message));
  }

  return (
    <div className="flex flex-col gap-4">
      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300 px-3 py-2 text-sm">
          {error}
        </div>
      )}

      <Card
        title="Chat models"
        subtitle="Selectable in Ask and Benchmark once downloaded. Interchangeable -- pick any that's installed."
      >
        <ModelTable
          rows={data?.catalog.map((m) => ({
            name: m,
            installed: data.installed.includes(m),
          }))}
          job={(m) => jobs[m]}
          onDownload={download}
        />
      </Card>

      <Card
        title="Embedding model"
        subtitle="Not a choice -- every ingested chunk was vectorized with this one model. Ingestion and Ask both need it, but it isn't fetched automatically; download it here once."
      >
        <ModelTable
          rows={data?.embed_models.map((m) => ({
            name: m,
            installed: data.embed_installed.includes(m),
          }))}
          job={(m) => jobs[m]}
          onDownload={download}
        />
      </Card>

      <Card
        title="Reranker"
        subtitle={
          data
            ? `${data.rerank_enabled ? "Enabled" : "Disabled"} via RAG_RERANK. Downloaded here so the first real query doesn't pay for it.`
            : ""
        }
      >
        <ModelTable
          rows={
            data
              ? [{ name: data.rerank_model, installed: data.rerank_installed }]
              : undefined
          }
          job={(m) => jobs[m]}
          onDownload={download}
        />
      </Card>
    </div>
  );
}

function ModelTable({ rows, job, onDownload }) {
  if (!rows) return <div className="text-sm text-neutral-500">Loading...</div>;
  return (
    <div className="flex flex-col divide-y divide-neutral-200 dark:divide-neutral-800">
      {rows.map((row) => (
        <ModelRow key={row.name} row={row} job={job(row.name)} onDownload={onDownload} />
      ))}
    </div>
  );
}

function ModelRow({ row, job, onDownload }) {
  const busy = job && ["pending", "running"].includes(job.status);
  return (
    <div className="py-3 flex items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="font-mono text-sm truncate">{row.name}</div>
        {busy && (
          <div className="mt-1">
            <div className="h-1.5 w-full max-w-xs rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
              />
            </div>
            <div className="text-xs text-neutral-500 mt-0.5 truncate">{job.message}</div>
          </div>
        )}
        {job && !busy && job.status !== "done" && (
          <div className="text-xs mt-0.5">
            <StatusBadge status={job.status} />
            {job.error && <span className="ml-2 text-red-600 dark:text-red-400">{job.error}</span>}
          </div>
        )}
      </div>
      {row.installed ? (
        <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
          installed
        </span>
      ) : (
        <button
          type="button"
          disabled={busy}
          onClick={() => onDownload(row.name)}
          className="text-sm font-medium px-3 py-1.5 rounded-md bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 disabled:opacity-50"
        >
          {busy ? "Downloading..." : "Download"}
        </button>
      )}
    </div>
  );
}
