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
  Eye,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import {
  uploadAndIngest,
  activeIngest,
  ingestHistory,
  ingestSplitters,
  deleteIngestedFile,
  pollJob,
  cancelJob,
} from "../api.js";

const BUSY_STATUSES = ["pending", "running"];
const ACCEPTED_EXTENSIONS = [".csv", ".txt", ".md", ".json", ".jsonl", ".pdf"];

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${["B", "KB", "MB", "GB"][i]}`;
}

function timeAgo(dateString) {
  if (!dateString) return "";
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return dateString;
  const s = Math.floor((new Date() - date) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} hr ago`;
  return `${Math.floor(s / 86400)}d ago`;
}


const CITATION_HINTS = [
  "source_url",
  "source",
  "url",
  "link",
  "href",
  "document_index",
  "document_id",
  "passage_id",
  "doc_id",
  "row_id",
  "index",
  "id",
  "_id",
  "idx",
];

function parseCsvHeaders(text) {
  if (!text) return [];
  const firstLine = text.split(/\r?\n/)[0];
  if (!firstLine) return [];
  const columns = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < firstLine.length; i++) {
    const char = firstLine[i];
    if (char === '"') {
      if (inQuotes && firstLine[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      columns.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  columns.push(current.trim());
  return columns.filter((c) => c.length > 0);
}

export default function Ingest() {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [csvColumns, setCsvColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [citationColumns, setCitationColumns] = useState([]);
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
    ingestHistory().then(setHistory).catch(() => {});
  }, []);

  const refreshHistory = () => ingestHistory().then(setHistory).catch(() => {});
  const busy = submitting || (job !== null && BUSY_STATUSES.includes(job.status));

  useEffect(() => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) {
      setCsvColumns([]);
      setSelectedColumns([]);
      setCitationColumns([]);
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result || "";
      const headers = parseCsvHeaders(content);
      const cited = headers.filter((h) => CITATION_HINTS.includes(h.toLowerCase()));
      setCsvColumns(headers);


      setSelectedColumns(headers.filter((h) => !cited.includes(h)));
      setCitationColumns(cited);
    };
    reader.readAsText(file.slice(0, 65536));
  }, [file]);

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

  useEffect(() => {
    const handleClickOutside = () => setOpenMenuId(null);
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

  function onCancelIngest() {
    if (!job) return;
    setError(null);
    cancelJob(job.id).catch((err) => setError(err.message));
  }

  async function onDelete(entry) {
    if (!window.confirm(`Delete ${entry.filename} and its vectors? This can't be undone.`)) return;
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
    const isCsv = chosen.name.toLowerCase().endsWith(".csv");
    const colsToIndex = isCsv && selectedColumns.length > 0 ? selectedColumns : null;
    const colsToCite = isCsv && citationColumns.length > 0 ? citationColumns : null;

    setFile(null);
    setCsvColumns([]);
    setSelectedColumns([]);
    setCitationColumns([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    track(() => uploadAndIngest(chosen, splitter || null, colsToIndex, colsToCite));
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    if (busy) return;
    const dropped = e.dataTransfer.files?.[0];
    const ext = dropped ? `.${dropped.name.split(".").pop().toLowerCase()}` : "";
    if (dropped && ACCEPTED_EXTENSIONS.includes(ext)) {
      setFile(dropped);
    } else {
      setError(`Please drop a supported file (${ACCEPTED_EXTENSIONS.join(", ")})`);
    }
  }

  function copySha(sha, id) {
    navigator.clipboard.writeText(sha);
    setCopiedSha(id);
    setTimeout(() => setCopiedSha(null), 1500);
  }

  const filteredHistory = history.filter((item) =>
    item.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalDocuments = history.reduce((acc, cur) => acc + (cur.documents || 0), 0);
  const totalChunks = history.reduce((acc, cur) => acc + (cur.chunk_ids?.length || cur.documents || 0), 0);
  const totalSizeBytes = history.reduce((acc, cur) => acc + (cur.size_bytes || 0), 0);

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">

      <div>
        <h1 className="text-2xl font-bold tracking-tight text-ink">Ingest</h1>
        <p className="text-sm text-ink-3 mt-1">Add documents to your knowledge base</p>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Error: </span>
            {error}
          </div>
        </div>
      )}


      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">

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
              ? "border-accent-line bg-accent-soft"
              : "border-hairline hover:border-ink-4 bg-surface-2 hover:bg-surface-3"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            disabled={busy}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="hidden"
          />
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-2 text-ink-3 group-hover:text-accent group-hover:bg-surface-2 transition-colors mb-3">
            <UploadCloud className="h-6 w-6" />
          </div>
          <p className="text-sm font-medium text-ink">
            {file ? file.name : "Drop files here or click to browse"}
          </p>
          <p className="text-xs text-ink-4 mt-1">
            {file
              ? `${formatBytes(file.size)} • Ready to ingest`
              : `CSV, TXT, MD, JSON, JSONL, PDF up to 50MB`}
          </p>

          <button
            type="button"
            className="mt-4 px-4 py-1.5 rounded-lg bg-surface-2 hover:bg-surface-3 text-xs font-medium text-ink border border-hairline transition-colors"
          >
            {file ? "Change File" : "Choose File"}
          </button>
        </div>


        {file && csvColumns.length > 0 && (
          <div className="mt-4 rounded-xl border border-hairline bg-surface-2 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-ink">Fields to Index</span>
                  <span
                    className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                      selectedColumns.length === 0
                        ? "bg-amber-50 text-amber-700 border-amber-200"
                        : "bg-accent-soft text-accent border-accent-line"
                    }`}
                  >
                    {selectedColumns.length} of {csvColumns.length} selected
                  </span>
                </div>
                <p className="text-[11px] text-ink-3 mt-0.5">
                  Index columns are concatenated and embedded. Citation columns (e.g. an id or URL) are
                  stored as metadata and shown as the source when this data is used in an answer.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedColumns([...csvColumns])}
                  className="text-[11px] text-accent hover:text-accent font-medium px-2.5 py-1 rounded-lg bg-accent-soft hover:bg-accent-strong-soft border border-accent-line transition-colors"
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedColumns([])}
                  className="text-[11px] text-ink-3 hover:text-ink-2 font-medium px-2.5 py-1 rounded-lg bg-surface-2 hover:bg-surface-2 border border-hairline transition-colors"
                >
                  Deselect All
                </button>
              </div>
            </div>

            <div className="flex items-center gap-4 text-[10px] text-ink-4 font-semibold uppercase tracking-wide mb-1.5 px-2">
              <span className="w-full" />
              <span className="w-10 text-center shrink-0">Index</span>
              <span className="w-10 text-center shrink-0">Cite</span>
            </div>
            <div className="flex flex-col gap-1.5 pt-1 max-h-48 overflow-y-auto pr-1">
              {csvColumns.map((col) => {
                const isIndexed = selectedColumns.includes(col);
                const isCited = citationColumns.includes(col);
                return (
                  <div
                    key={col}
                    className={`group flex items-center gap-4 px-2 py-2 rounded-lg border text-xs transition-colors ${
                      isIndexed || isCited
                        ? "border-hairline bg-surface text-ink shadow-xs"
                        : "border-hairline/60 bg-transparent text-ink-3 hover:bg-surface"
                    }`}
                  >
                    <span className="truncate font-mono text-[11px] flex-1" title={col}>
                      {col}
                    </span>
                    <label className="w-10 flex items-center justify-center shrink-0 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={isIndexed}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedColumns((prev) => [...prev, col]);
                          } else {
                            setSelectedColumns((prev) => prev.filter((c) => c !== col));
                          }
                        }}
                        title="Concatenate and embed this column"
                        className="rounded border-hairline bg-surface text-accent focus:ring-2 focus:ring-accent/40 focus:ring-offset-0 h-4 w-4 cursor-pointer accent-accent"
                      />
                    </label>
                    <label className="w-10 flex items-center justify-center shrink-0 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={isCited}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setCitationColumns((prev) => [...prev, col]);
                          } else {
                            setCitationColumns((prev) => prev.filter((c) => c !== col));
                          }
                        }}
                        title="Use this column as the citation/source shown with answers"
                        className="rounded border-hairline bg-surface text-emerald-700 focus:ring-2 focus:ring-emerald-500/40 focus:ring-offset-0 h-4 w-4 cursor-pointer accent-emerald-500"
                      />
                    </label>
                  </div>
                );
              })}
            </div>
            {selectedColumns.length === 0 && (
              <p className="text-[11px] text-amber-700 mt-2 font-medium flex items-center gap-1.5">
                <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                At least one field should be selected to produce searchable embeddings.
              </p>
            )}
            {citationColumns.length === 0 && (
              <p className="text-[11px] text-ink-4 mt-1">
                No citation columns selected -- answers sourced from this file won't show an id or link.
              </p>
            )}
          </div>
        )}


        <div className="flex flex-wrap items-center justify-between gap-4 mt-5 pt-4 border-t border-hairline">
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-ink-3">Processing mode</span>
            {splitters.length > 0 ? (
              <div className="relative">
                <select
                  value={splitter}
                  onChange={(e) => setSplitter(e.target.value)}
                  disabled={busy}
                  className="appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-medium focus:outline-none focus:border-accent-line transition-colors cursor-pointer"
                >
                  {splitters.map((s) => (
                    <option key={s} value={s} className="bg-surface text-ink">
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                  <span className="text-[10px]">▼</span>
                </div>
              </div>
            ) : (
              <span className="text-xs text-ink-4">Default (Recursive)</span>
            )}
          </div>

          <button
            type="button"
            onClick={onUpload}
            disabled={busy || !file || (csvColumns.length > 0 && selectedColumns.length === 0)}
            className="flex items-center gap-2 rounded-lg bg-accent hover:bg-accent-strong text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            <Sparkles className="h-4 w-4" />
            <span>{busy ? "Ingesting..." : "Upload & Ingest"}</span>
          </button>
        </div>
      </div>


      {job && (
        <Card title="Ingestion Progress" className="border-accent-line bg-accent-soft">
          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2.5">
              <StatusBadge status={job.status} />
              <span className="text-xs font-medium text-ink-2">{job.message}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold text-accent">
                {Math.round((job.progress ?? 0) * 100)}%
              </span>
              {BUSY_STATUSES.includes(job.status) && (
                <button
                  type="button"
                  onClick={onCancelIngest}
                  title="Cancel ingestion"
                  className="inline-flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-lg text-rose-700 hover:text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 transition-all cursor-pointer"
                >
                  <X className="h-3.5 w-3.5 text-rose-700" />
                  <span>Cancel</span>
                </button>
              )}
            </div>
          </div>

          <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent to-accent-strong transition-all duration-300 rounded-full"
              style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
            />
          </div>

          {job.status === "done" && job.result && (
            <div className="mt-4 grid grid-cols-3 gap-3 pt-3 border-t border-hairline text-xs">
              <div>
                <span className="text-ink-3">Documents: </span>
                <span className="font-semibold text-ink">{job.result.documents ?? "--"}</span>
              </div>
              <div>
                <span className="text-ink-3">Chunks: </span>
                <span className="font-semibold text-ink">{job.result.chunks ?? "--"}</span>
              </div>
              <div>
                <span className="text-ink-3">Splitter: </span>
                <span className="font-semibold text-ink">{job.result.splitter ?? "--"}</span>
              </div>
            </div>
          )}

          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-rose-700 bg-rose-50 p-3 rounded-lg border border-rose-200 whitespace-pre-wrap font-mono">
              {job.error}
            </pre>
          )}
        </Card>
      )}


      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h2 className="text-base font-bold text-ink">Knowledge base</h2>
            <p className="text-xs text-ink-3 mt-0.5">
              {history.length > 0
                ? `${history.length} documents • ${totalChunks > 0 ? totalChunks.toLocaleString() : totalDocuments.toLocaleString()} chunks • ${formatBytes(totalSizeBytes)}`
                : "No documents ingested yet"}
            </p>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-ink-3" aria-hidden="true" />
            <input
              type="text"
              aria-label="Search documents"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="rounded-lg border border-hairline bg-surface pl-8 pr-3 py-1.5 text-xs text-ink placeholder-ink-4 focus:outline-none focus:border-accent-line w-full sm:w-56 transition-colors"
            />
          </div>
        </div>


        {filteredHistory.length === 0 ? (
          <div className="py-12 text-center text-ink-4 text-xs">
            {searchQuery ? "No matching documents found." : "No documents ingested yet. Upload a file above to start."}
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
                  className="flex items-center justify-between p-3.5 rounded-xl border border-hairline bg-surface-2 hover:bg-surface-2 hover:border-hairline transition-all duration-150"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent-soft border border-accent-line text-accent">
                      <FileText className="h-5 w-5" />
                    </div>

                    <div className="min-w-0">
                      <div className="font-semibold text-sm text-ink truncate">
                        {entry.filename}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-2 text-xs text-ink-3 mt-0.5">
                        <span>{chunksCount.toLocaleString()} chunks</span>
                        <span>•</span>
                        <span>{formatBytes(entry.size_bytes)}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3 inline text-ink-4" />
                          Indexed {timeAgo(entry.created_at)}
                        </span>
                      </div>
                      {entry.index_columns && entry.index_columns.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1 mt-1.5">
                          <span className="text-[10px] text-ink-4 font-medium">Fields:</span>
                          {entry.index_columns.map((col) => (
                            <span
                              key={col}
                              className="px-1.5 py-0.2 rounded bg-surface-2 text-accent font-mono text-[10px] border border-hairline"
                            >
                              {col}
                            </span>
                          ))}
                        </div>
                      )}
                      {entry.citation_columns && entry.citation_columns.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1 mt-1">
                          <span className="text-[10px] text-ink-4 font-medium">Cited by:</span>
                          {entry.citation_columns.map((col) => (
                            <span
                              key={col}
                              className="px-1.5 py-0.2 rounded bg-surface-2 text-emerald-700 font-mono text-[10px] border border-hairline"
                            >
                              {col}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="relative shrink-0 ml-3" onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      onClick={() => setOpenMenuId(isMenuOpen ? null : entry.id)}
                      className="p-1.5 rounded-lg text-ink-3 hover:text-ink hover:bg-surface-2 transition-colors cursor-pointer"
                      title="Actions"
                      aria-label={`Actions for ${entry.filename}`}
                      aria-haspopup="menu"
                      aria-expanded={isMenuOpen}
                    >
                      <MoreVertical className="h-4 w-4" aria-hidden="true" />
                    </button>

                    {isMenuOpen && (
                      <div className="absolute right-0 top-full mt-1 z-20 w-44 rounded-xl border border-hairline bg-surface p-1 shadow-xl text-xs">
                        <Link
                          to={`/documents/${entry.id}`}
                          onClick={() => setOpenMenuId(null)}
                          className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-left text-ink-2 hover:bg-surface-2 hover:text-ink transition-colors"
                        >
                          <Eye className="h-3.5 w-3.5 text-ink-3" />
                          <span>View document</span>
                        </Link>
                        <button
                          type="button"
                          onClick={() => {
                            copySha(entry.sha256, entry.id);
                            setOpenMenuId(null);
                          }}
                          className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-left text-ink-2 hover:bg-surface-2 hover:text-ink transition-colors"
                        >
                          {copiedSha === entry.id ? (
                            <Check className="h-3.5 w-3.5 text-emerald-700" />
                          ) : (
                            <Copy className="h-3.5 w-3.5 text-ink-3" />
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
                          className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-left text-rose-700 hover:bg-rose-50 transition-colors"
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
