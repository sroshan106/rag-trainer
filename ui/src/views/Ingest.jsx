import { useEffect, useRef, useState } from "react";
import {
  UploadCloud,
  FileText,
  Search,
  MoreVertical,
  Trash2,
  Copy,
  Check,
  AlertCircle,
  Clock,
  Sparkles,
} from "lucide-react";
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

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function timeAgo(dateString) {
  if (!dateString) return "";
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return dateString;
  const seconds = Math.floor((new Date() - date) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function Ingest() {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [history, setHistory] = useState([]);
  const [deletingId, setDeletingId] = useState(null);
  const [splitters, setSplitters] = useState([]);
  const [splitter, setSplitter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [copiedSha, setCopiedSha] = useState(null);
  const [openMenuId, setOpenMenuId] = useState(null);

  const cancelledRef = useRef(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    ingestSplitters()
      .then(({ splitters: available, default: def }) => {
        setSplitters(available);
        setSplitter(def);
      })
      .catch(() => {});
  }, []);

  function refreshHistory() {
    ingestHistory()
      .then(setHistory)
      .catch(() => {});
  }

  useEffect(refreshHistory, []);

  const busy = submitting || (job !== null && BUSY_STATUSES.includes(job.status));

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
      .catch(() => {});
    return () => {
      ignore = true;
      cancelledRef.current = true;
    };
  }, []);

  // Close menus when clicking outside
  useEffect(() => {
    function handleClickOutside() {
      setOpenMenuId(null);
    }
    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
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

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    if (busy) return;
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.name.endsWith(".csv")) {
      setFile(droppedFile);
    } else {
      setError("Please drop a valid .csv file");
    }
  }

  function copySha(sha, id) {
    navigator.clipboard.writeText(sha);
    setCopiedSha(id);
    setTimeout(() => setCopiedSha(null), 1500);
  }

  // Filtered documents
  const filteredHistory = history.filter((item) =>
    item.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Compute aggregate stats
  const totalDocuments = history.reduce((acc, cur) => acc + (cur.documents || 0), 0);
  const totalChunks = history.reduce(
    (acc, cur) => acc + (cur.chunk_ids?.length || cur.documents || 0),
    0
  );
  const totalSizeBytes = history.reduce((acc, cur) => acc + (cur.size_bytes || 0), 0);

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-100">Ingest</h1>
        <p className="text-sm text-slate-400 mt-1">Add documents to your knowledge base</p>
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

      {/* Upload Box matching mockup */}
      <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
        {/* Dropzone area */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`group flex flex-col items-center justify-center rounded-xl border border-dashed py-8 px-4 transition-all duration-200 cursor-pointer ${
            isDragging
              ? "border-blue-500 bg-blue-950/20"
              : "border-slate-800 hover:border-slate-700 bg-slate-900/30 hover:bg-slate-900/50"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            disabled={busy}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="hidden"
          />
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800/80 text-slate-400 group-hover:text-blue-400 group-hover:bg-slate-800 transition-colors mb-3">
            <UploadCloud className="h-6 w-6" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            {file ? file.name : "Drop files here or click to browse"}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {file ? `${formatBytes(file.size)} • Ready to ingest` : "CSV files up to 50MB"}
          </p>

          <button
            type="button"
            className="mt-4 px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700/80 transition-colors"
          >
            {file ? "Change File" : "Choose File"}
          </button>
        </div>

        {/* Processing mode + Upload button row */}
        <div className="flex flex-wrap items-center justify-between gap-4 mt-5 pt-4 border-t border-slate-800/80">
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-slate-400">Processing mode</span>
            {splitters.length > 0 ? (
              <div className="relative">
                <select
                  value={splitter}
                  onChange={(e) => setSplitter(e.target.value)}
                  disabled={busy}
                  className="appearance-none rounded-lg border border-slate-700/80 bg-slate-900/90 text-slate-200 px-3 py-1.5 pr-8 text-xs font-medium focus:outline-none focus:border-blue-500 transition-colors cursor-pointer"
                >
                  {splitters.map((s) => (
                    <option key={s} value={s} className="bg-[#111726] text-slate-200">
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                  <span className="text-[10px]">▼</span>
                </div>
              </div>
            ) : (
              <span className="text-xs text-slate-500">Default (Recursive)</span>
            )}
          </div>

          <button
            type="button"
            onClick={onUpload}
            disabled={busy || !file}
            className="flex items-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            <Sparkles className="h-4 w-4" />
            <span>{busy ? "Ingesting..." : "Upload & Ingest"}</span>
          </button>
        </div>
      </div>

      {/* In-Flight Job Status Card */}
      {job && (
        <Card title="Ingestion Progress" className="border-blue-900/60 bg-blue-950/20">
          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2.5">
              <StatusBadge status={job.status} />
              <span className="text-xs font-medium text-slate-300">{job.message}</span>
            </div>
            <span className="text-xs font-mono font-semibold text-blue-400">
              {Math.round((job.progress ?? 0) * 100)}%
            </span>
          </div>

          <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-300 rounded-full"
              style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
            />
          </div>

          {job.status === "done" && job.result && (
            <div className="mt-4 grid grid-cols-3 gap-3 pt-3 border-t border-slate-800/80 text-xs">
              <div>
                <span className="text-slate-400">Documents: </span>
                <span className="font-semibold text-slate-200">{job.result.documents ?? "--"}</span>
              </div>
              <div>
                <span className="text-slate-400">Chunks: </span>
                <span className="font-semibold text-slate-200">{job.result.chunks ?? "--"}</span>
              </div>
              <div>
                <span className="text-slate-400">Splitter: </span>
                <span className="font-semibold text-slate-200">{job.result.splitter ?? "--"}</span>
              </div>
            </div>
          )}

          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-rose-400 bg-rose-950/40 p-3 rounded-lg border border-rose-900/60 whitespace-pre-wrap font-mono">
              {job.error}
            </pre>
          )}
        </Card>
      )}

      {/* Knowledge Base Section matching mockup */}
      <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
        {/* Knowledge Base Header with Summary & Search */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h2 className="text-base font-bold text-slate-100">Knowledge base</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {history.length > 0
                ? `${history.length} documents • ${totalChunks > 0 ? totalChunks.toLocaleString() : totalDocuments.toLocaleString()} chunks • ${formatBytes(totalSizeBytes)}`
                : "No documents ingested yet"}
            </p>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="rounded-lg border border-slate-700/80 bg-slate-900/90 pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 w-full sm:w-56 transition-colors"
            />
          </div>
        </div>

        {/* Document List */}
        {filteredHistory.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-xs">
            {searchQuery ? "No matching documents found." : "No documents ingested yet. Upload a CSV above to start."}
          </div>
        ) : (
          <div className="flex flex-col gap-2.5">
            {filteredHistory.map((entry) => {
              const isDeleting = deletingId === entry.id;
              const chunksCount = entry.chunk_ids?.length || entry.documents || 0;
              const isMenuOpen = openMenuId === entry.id;

              return (
                <div
                  key={entry.id}
                  className="flex items-center justify-between p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:bg-slate-900/80 hover:border-slate-700/80 transition-all duration-150"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-950/70 border border-blue-800/50 text-blue-400">
                      <FileText className="h-5 w-5" />
                    </div>

                    <div className="min-w-0">
                      <div className="font-semibold text-sm text-slate-100 truncate">
                        {entry.filename}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-2 text-xs text-slate-400 mt-0.5">
                        <span>{chunksCount.toLocaleString()} chunks</span>
                        <span>•</span>
                        <span>{formatBytes(entry.size_bytes)}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3 inline text-slate-500" />
                          Indexed {timeAgo(entry.created_at)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="relative shrink-0 ml-3" onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      onClick={() => setOpenMenuId(isMenuOpen ? null : entry.id)}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                      title="Actions"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </button>

                    {isMenuOpen && (
                      <div className="absolute right-0 top-full mt-1 z-20 w-44 rounded-xl border border-slate-700/80 bg-[#151c2d] p-1 shadow-xl text-xs">
                        <button
                          type="button"
                          onClick={() => {
                            copySha(entry.sha256, entry.id);
                            setOpenMenuId(null);
                          }}
                          className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-left text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
                        >
                          {copiedSha === entry.id ? (
                            <Check className="h-3.5 w-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="h-3.5 w-3.5 text-slate-400" />
                          )}
                          <span>{copiedSha === entry.id ? "SHA Copied" : "Copy SHA-256"}</span>
                        </button>
                        <button
                          type="button"
                          disabled={busy || isDeleting}
                          onClick={() => {
                            setOpenMenuId(null);
                            onDelete(entry);
                          }}
                          className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-left text-rose-400 hover:bg-rose-950/50 transition-colors"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          <span>{isDeleting ? "Deleting..." : "Delete document"}</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

