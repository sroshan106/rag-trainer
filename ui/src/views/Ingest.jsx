import { useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { startIngest, pollJob } from "../api.js";

export default function Ingest() {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const cancelledRef = useRef(false);

  const running = job && job.status === "running";

  async function onStart() {
    setError(null);
    cancelledRef.current = false;
    try {
      const started = await startIngest();
      setJob(started);
      await pollJob(started.id, {
        onUpdate: setJob,
        isCancelled: () => cancelledRef.current,
      });
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Ingest"
        subtitle="Runs src.ingestion.pipeline against data/documents.csv as a background job."
      >
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-3">
          The pipeline's entry point is fixed to that one CSV and the default chunk
          settings -- this triggers the same run as{" "}
          <code className="bg-neutral-100 dark:bg-neutral-800 px-1 rounded">
            python -m src.ingestion.pipeline
          </code>
          . Re-running adds to the existing collection rather than replacing it.
        </p>
        <button
          onClick={onStart}
          disabled={running}
          className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {running ? "Ingesting..." : "Start ingest"}
        </button>
      </Card>

      {error && (
        <Card title="Error" className="border-red-300 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </Card>
      )}

      {job && (
        <Card title="Job status">
          <div className="flex items-center gap-3 mb-3">
            <StatusBadge status={job.status} />
            <span className="text-xs text-neutral-500">{job.message}</span>
          </div>
          <div className="h-2 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
            <div
              className="h-full bg-neutral-900 dark:bg-neutral-100 transition-all"
              style={{ width: `${Math.round(job.progress * 100)}%` }}
            />
          </div>

          {job.status === "done" && job.result && (
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-xs text-neutral-500">Documents loaded</dt>
                <dd className="font-medium">{job.result.documents ?? "unknown"}</dd>
              </div>
              <div>
                <dt className="text-xs text-neutral-500">Chunks written</dt>
                <dd className="font-medium">{job.result.chunks ?? "unknown"}</dd>
              </div>
            </dl>
          )}

          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-red-600 dark:text-red-400 whitespace-pre-wrap">
              {job.error}
            </pre>
          )}
        </Card>
      )}
    </div>
  );
}
