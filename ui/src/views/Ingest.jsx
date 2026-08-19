import { useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import {
  uploadAndIngest,
  activeIngest,
  ingestHistory,
  ingestSplitters,
  deleteIngestedFile,
  pollJob,
} from "../api.js";

const BUSY_STATUSES = ["pending", "running"];

export default function Ingest() {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [history, setHistory] = useState([]);
  const [deletingId, setDeletingId] = useState(null);
  const [splitters, setSplitters] = useState([]);
  const [splitter, setSplitter] = useState("");
  const cancelledRef = useRef(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    ingestSplitters()
      .then(({ splitters: available, default: def }) => {
        setSplitters(available);
        setSplitter(def);
      })
      .catch(() => {
        // No API yet -- upload just uses the server's default splitter.
      });
  }, []);

  function refreshHistory() {
    ingestHistory()
      .then(setHistory)
      .catch(() => {
        // No API yet, or the table doesn't exist -- an empty panel is fine.
      });
  }

  useEffect(refreshHistory, []);

  // "pending" counts as busy too: a freshly submitted job has not been picked
  // up by its thread yet, and leaving the button live through that window is
  // what allowed a double-click to queue two ingests.
  const busy = submitting || (job !== null && BUSY_STATUSES.includes(job.status));

  // Re-attach to a run already in progress, so reloading the page (or opening
  // a second tab) shows the running job instead of an idle-looking form.
  useEffect(() => {
    cancelledRef.current = false;
    let ignore = false;
    activeIngest()
      .then((running) => {
        if (ignore || !running) return;
        setJob(running);
        return pollJob(running.id, {
          onUpdate: setJob,
          isCancelled: () => cancelledRef.current,
        });
      })
      .catch(() => {
        // No API yet, or nothing running -- the idle form is the right default.
      });
    return () => {
      ignore = true;
      cancelledRef.current = true;
    };
  }, []);

  async function track(startFn) {
    setError(null);
    setSubmitting(true);
    cancelledRef.current = false;
    try {
      const started = await startFn();
      setJob(started);
      await pollJob(started.id, {
        onUpdate: setJob,
        isCancelled: () => cancelledRef.current,
      });
      refreshHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function onDelete(entry) {
    if (!window.confirm(`Delete ${entry.filename} and its vectors? This can't be undone.`)) {
      return;
    }
    setError(null);
    setDeletingId(entry.id);
    try {
      await deleteIngestedFile(entry.id);
      refreshHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  }

  function onUpload() {
    if (!file) return;
    const chosen = file;
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    track(() => uploadAndIngest(chosen, splitter || null));
  }

  const buttonClass =
    "rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 " +
    "px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Upload a dataset"
        subtitle="A CSV with a text column, plus optional source_url and index columns."
      >
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-3">
          Re-uploading the same bytes is refused rather than stored twice. Only one
          ingest runs at a time.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            disabled={busy}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="text-sm file:mr-3 file:rounded-md file:border-0 file:bg-neutral-200
              dark:file:bg-neutral-800 file:px-3 file:py-2 file:text-sm
              disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {splitters.length > 0 && (
            <select
              value={splitter}
              onChange={(e) => setSplitter(e.target.value)}
              disabled={busy}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm disabled:opacity-50 dark:[color-scheme:dark]"
            >
              {splitters.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          )}
          <button onClick={onUpload} disabled={busy || !file} className={buttonClass}>
            {busy ? "Ingesting..." : "Upload and ingest"}
          </button>
        </div>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-3">
          The file is parsed before the job starts, so a missing text column is
          reported here rather than failing in the background.
        </p>
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
              <div>
                <dt className="text-xs text-neutral-500">Splitter</dt>
                <dd className="font-medium">{job.result.splitter ?? "unknown"}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-neutral-500">Source</dt>
                <dd className="font-mono text-xs break-all">{job.result.path}</dd>
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

      {history.length > 0 && (
        <Card
          title="Ingested files"
          subtitle="Every saved copy, by content hash. Delete clears its vectors from the store, its saved copy, and this record."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-neutral-500">
                  <th className="pb-2 pr-4">Filename</th>
                  <th className="pb-2 pr-4">Ingested</th>
                  <th className="pb-2 pr-4">Documents</th>
                  <th className="pb-2 pr-4">SHA-256</th>
                  <th className="pb-2 pr-4"></th>
                </tr>
              </thead>
              <tbody>
                {history.map((entry) => (
                  <tr key={entry.id} className="border-t border-neutral-200 dark:border-neutral-800">
                    <td className="py-2 pr-4">{entry.filename}</td>
                    <td className="py-2 pr-4 text-xs text-neutral-500">
                      {new Date(entry.created_at).toLocaleString()}
                    </td>
                    <td className="py-2 pr-4">{entry.documents ?? "unknown"}</td>
                    <td className="py-2 pr-4 font-mono text-xs break-all text-neutral-500">
                      {entry.sha256}
                    </td>
                    <td className="py-2 pr-4">
                      <button
                        onClick={() => onDelete(entry)}
                        disabled={busy || deletingId === entry.id}
                        className="text-xs text-red-600 dark:text-red-400 hover:underline
                          disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {deletingId === entry.id ? "Deleting..." : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
