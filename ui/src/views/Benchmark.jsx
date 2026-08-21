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
  GitCompare,
  Clock,
  Zap,
  Plus,
  Trash2,
  UploadCloud,
  FileSpreadsheet,
  Check,
  X,
  Layers,
} from "lucide-react";

import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import SourceList from "../components/SourceList.jsx";
import {
  startBenchmark,
  benchmarkHistory,
  benchmarkModels,
  activeBenchmark,
  cancelJob,
  pollJob,
  compareQuery,
  getBenchmarkTestFiles,
  uploadBenchmarkTestFile,
  deleteBenchmarkTestFile,
} from "../api.js";

const QUESTION_ALIASES = ["question", "query", "prompt", "q", "question_text"];
const ANSWER_ALIASES = ["answer", "ground_truth", "reference", "expected", "target", "a", "expected_answer"];
const DOC_INDEX_ALIASES = ["document_index", "doc_index", "doc_id", "index", "document_id"];

function findMatchingHeader(headers, aliases) {
  const lowered = headers.map((h) => h.trim().toLowerCase());
  for (const alias of aliases) {
    const idx = lowered.indexOf(alias);
    if (idx !== -1) return headers[idx];
  }
  return "";
}

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

  const [testSuites, setTestSuites] = useState([]);
  const [selectedSuites, setSelectedSuites] = useState([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadHeaders, setUploadHeaders] = useState([]);
  const [questionCol, setQuestionCol] = useState("");
  const [answerCol, setAnswerCol] = useState("");
  const [docIndexCol, setDocIndexCol] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [deletingSuiteId, setDeletingSuiteId] = useState(null);
  const uploadFileInputRef = useRef(null);

  const [compareText, setCompareText] = useState("");
  const [compareModel, setCompareModel] = useState("");
  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState(null);

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
        setCompareModel((prev) => prev || available[0] || "");
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

  const loadTestSuites = () => {
    getBenchmarkTestFiles()
      .then((suites) => {
        setTestSuites(suites);
        setSelectedSuites(suites.map((s) => s.id));
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadTestSuites();
  }, []);

  function handleUploadFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploadFile(f);
    setUploadError(null);
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result || "";
      const headers = parseCsvHeaders(content);
      setUploadHeaders(headers);
      setQuestionCol(findMatchingHeader(headers, QUESTION_ALIASES) || headers[0] || "");
      setAnswerCol(findMatchingHeader(headers, ANSWER_ALIASES) || "");
      setDocIndexCol(findMatchingHeader(headers, DOC_INDEX_ALIASES) || "");
    };
    reader.readAsText(f.slice(0, 65536));
  }

  async function onSaveCustomSuite(e) {
    e.preventDefault();
    if (!uploadFile || !questionCol) {
      setUploadError("Please select a file and a question column");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const created = await uploadBenchmarkTestFile(uploadFile, {
        question_col: questionCol,
        answer_col: answerCol || null,
        doc_index_col: docIndexCol || null,
      });
      setShowUploadModal(false);
      setUploadFile(null);
      setUploadHeaders([]);
      setQuestionCol("");
      setAnswerCol("");
      setDocIndexCol("");
      getBenchmarkTestFiles().then((suites) => {
        setTestSuites(suites);
        setSelectedSuites((prev) => [...prev, created.id]);
      });
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function onDeleteSuite(id) {
    if (!window.confirm("Delete this benchmark test suite?")) return;
    setDeletingSuiteId(id);
    try {
      await deleteBenchmarkTestFile(id);
      setTestSuites((prev) => prev.filter((s) => s.id !== id));
      setSelectedSuites((prev) => prev.filter((sid) => sid !== id));
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingSuiteId(null);
    }
  }

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
        test_files: selectedSuites.length > 0 ? selectedSuites : null,
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

  async function onCompare() {
    const text = compareText.trim();
    if (!text || !compareModel || compareLoading) return;
    setCompareLoading(true);
    setCompareError(null);
    try {
      setCompareResult(await compareQuery(text, compareModel));
    } catch (err) {
      setCompareError(err.message);
      setCompareResult(null);
    } finally {
      setCompareLoading(false);
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
        <h1 className="text-2xl font-bold tracking-tight text-ink">Benchmark Suite</h1>
        <p className="text-sm text-ink-3 mt-1">
          Evaluate pipeline answer overlap, refusal precision, and latencies across test suites
        </p>
      </div>

      {modelsLoaded && models.length === 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p>
              No chat model is downloaded yet.{" "}
              <Link to="/settings" className="font-semibold underline hover:text-amber-700">
                Download one in Settings
              </Link>{" "}
              first to run benchmarks.
            </p>
          </div>
        </div>
      )}

      {/* Config Card */}
      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-hairline">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink">
            <Sliders className="h-4 w-4 text-accent" />
            <span>Benchmark Configuration</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-4 font-medium">Presets:</span>
            <button
              type="button"
              onClick={() => applyPreset("quick")}
              disabled={running}
              className="px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink-2 text-xs font-medium transition-colors"
            >
              ⚡ Quick Test (5)
            </button>
            <button
              type="button"
              onClick={() => applyPreset("full")}
              disabled={running}
              className="px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink-2 text-xs font-medium transition-colors"
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
                className="w-full appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-2 pr-8 text-xs font-mono font-medium focus:outline-none focus:border-accent-line transition-colors cursor-pointer"
              >
                {models.map((m) => (
                  <option key={m} value={m} className="bg-surface text-ink">
                    {m}
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
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
              className="w-full rounded-lg border border-hairline bg-surface text-ink px-3 py-2 text-xs font-medium focus:outline-none focus:border-accent-line"
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
              className="w-full rounded-lg border border-hairline bg-surface text-ink px-3 py-2 text-xs font-medium focus:outline-none focus:border-accent-line"
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
              className="w-full rounded-lg border border-hairline bg-surface text-ink px-3 py-2 text-xs font-medium focus:outline-none focus:border-accent-line"
            />
          </Field>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-hairline">
          <label className="flex items-center gap-2 text-xs font-medium text-ink-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useCache}
              disabled={running}
              onChange={(e) => setUseCache(e.target.checked)}
              className="rounded border-hairline bg-surface text-accent focus:ring-0"
            />
            <span>Use caching (skip repeating queries if cached)</span>
          </label>

          <div className="flex items-center gap-2">
            <button
              onClick={onStart}
              disabled={running || !model || testSuites.length === 0 || selectedSuites.length === 0}
              className="flex items-center gap-2 rounded-lg bg-accent hover:bg-accent-strong text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              <Play className="h-4 w-4" />
              <span>{running ? "Running..." : "Run Benchmark"}</span>
            </button>
            {running && (
              <button
                onClick={onStop}
                disabled={stopping}
                className="flex items-center gap-1.5 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink px-4 py-2 text-sm font-medium transition-colors"
              >
                <Square className="h-3.5 w-3.5" />
                <span>{stopping ? "Stopping..." : "Stop"}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Test Suites Management Card */}
      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-hairline">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-accent" />
            <span className="text-sm font-semibold text-ink">Evaluation Test Suites</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded-md bg-accent-soft text-accent border border-accent-line">
              {selectedSuites.length} of {testSuites.length} selected
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSelectedSuites(testSuites.map((s) => s.id))}
              className="px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink-2 text-xs font-medium transition-colors"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={() => setSelectedSuites([])}
              className="px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink-3 text-xs font-medium transition-colors"
            >
              Deselect All
            </button>
            <button
              type="button"
              onClick={() => {
                setShowUploadModal(true);
                setUploadError(null);
              }}
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-accent hover:bg-accent-strong text-white text-xs font-medium transition-colors shadow-sm"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>Upload Custom Suite</span>
            </button>
          </div>
        </div>

        {testSuites.length === 0 ? (
          <div className="py-8 text-center text-xs text-ink-4">
            No test suites available. Click "Upload Custom Suite" to add a CSV test set.
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {testSuites.map((suite) => {
              const isSelected = selectedSuites.includes(suite.id);
              const isDeleting = deletingSuiteId === suite.id;

              return (
                <div
                  key={suite.id}
                  className={`flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl border transition-all ${
                    isSelected
                      ? "border-hairline bg-surface-2"
                      : "border-hairline bg-surface-2 opacity-75"
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      disabled={running}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedSuites((prev) => [...prev, suite.id]);
                        } else {
                          setSelectedSuites((prev) => prev.filter((id) => id !== suite.id));
                        }
                      }}
                      className="rounded border-hairline bg-surface text-accent focus:ring-0 h-4 w-4 cursor-pointer"
                    />

                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-2 border border-hairline text-ink-2">
                      <FileSpreadsheet className="h-4 w-4" />
                    </div>

                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-xs text-ink truncate">
                          {suite.name}
                        </span>
                        <span className="px-1.5 py-0.2 rounded text-[10px] font-medium border bg-accent-soft text-accent border-accent-line capitalize">
                          {suite.suite_type.replace("_", " ")}
                        </span>
                      </div>

                      <div className="flex flex-wrap items-center gap-x-2 text-[11px] text-ink-3 mt-0.5">
                        <span>{suite.questions} questions</span>
                        {suite.question_col && (
                          <>
                            <span>•</span>
                            <span className="text-ink-4 font-mono text-[10px]">
                              Q: <span className="text-ink-2">{suite.question_col}</span>
                              {suite.answer_col && (
                                <> | A: <span className="text-ink-2">{suite.answer_col}</span></>
                              )}
                              {suite.doc_index_col && (
                                <> | Doc: <span className="text-ink-2">{suite.doc_index_col}</span></>
                              )}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <button
                    type="button"
                    disabled={running || isDeleting}
                    onClick={() => onDeleteSuite(suite.id)}
                    className="p-1.5 rounded-lg text-ink-3 hover:text-rose-700 hover:bg-rose-50 transition-colors"
                    title="Delete test suite"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Upload Custom Suite Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/25 backdrop-blur-sm">
          <div className="relative w-full max-w-lg rounded-2xl border border-hairline bg-surface p-6 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-hairline">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="h-5 w-5 text-accent" />
                <h3 className="text-sm font-bold text-ink">Upload Benchmark Test Suite</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowUploadModal(false)}
                className="p-1 rounded-lg text-ink-3 hover:text-ink hover:bg-surface-2 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {uploadError && (
              <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-rose-700 shrink-0 mt-0.5" />
                <div>{uploadError}</div>
              </div>
            )}

            <form onSubmit={onSaveCustomSuite} className="mt-4 flex flex-col gap-4">
              {/* File Drop / Select Area */}
              <div>
                <label className="text-xs font-medium text-ink-2 block mb-1.5">
                  Select CSV File
                </label>
                <input
                  ref={uploadFileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleUploadFileChange}
                  className="hidden"
                />
                <div
                  onClick={() => uploadFileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center rounded-xl border border-dashed border-hairline bg-surface-2 hover:bg-surface-2 p-4 cursor-pointer transition-colors"
                >
                  <UploadCloud className="h-6 w-6 text-ink-3 mb-1" />
                  <p className="text-xs font-medium text-ink">
                    {uploadFile ? uploadFile.name : "Click to select benchmark CSV"}
                  </p>
                  <p className="text-[10px] text-ink-4 mt-0.5">
                    {uploadFile ? `${uploadHeaders.length} columns detected` : "CSV format up to 50MB"}
                  </p>
                </div>
              </div>

              {/* Column Mapping Section */}
              {uploadHeaders.length > 0 && (
                <div className="rounded-xl border border-hairline bg-surface-2 p-3.5 flex flex-col gap-3">
                  <span className="text-xs font-semibold text-ink">Column Matching</span>
                  <p className="text-[11px] text-ink-3">
                    Map CSV headers to benchmark question and answer fields.
                  </p>

                  <Field label="Question Column (Required)">
                    <div className="relative">
                      <select
                        value={questionCol}
                        onChange={(e) => setQuestionCol(e.target.value)}
                        required
                        className="w-full appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-mono focus:outline-none focus:border-accent-line cursor-pointer"
                      >
                        {uploadHeaders.map((h) => (
                          <option key={h} value={h}>
                            {h}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                        <span className="text-[10px]">▼</span>
                      </div>
                    </div>
                  </Field>

                  <Field label="Answer Column (Optional - for ground truth answer scoring)">
                    <div className="relative">
                      <select
                        value={answerCol}
                        onChange={(e) => setAnswerCol(e.target.value)}
                        className="w-full appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-mono focus:outline-none focus:border-accent-line cursor-pointer"
                      >
                        <option value="">(None - No Answer / Refusal evaluation)</option>
                        {uploadHeaders.map((h) => (
                          <option key={h} value={h}>
                            {h}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                        <span className="text-[10px]">▼</span>
                      </div>
                    </div>
                  </Field>

                  <Field label="Document Identifier / Index Column (Optional - for recall@k scoring)">
                    <div className="relative">
                      <select
                        value={docIndexCol}
                        onChange={(e) => setDocIndexCol(e.target.value)}
                        className="w-full appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-mono focus:outline-none focus:border-accent-line cursor-pointer"
                      >
                        <option value="">(None)</option>
                        {uploadHeaders.map((h) => (
                          <option key={h} value={h}>
                            {h}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                        <span className="text-[10px]">▼</span>
                      </div>
                    </div>
                  </Field>
                </div>
              )}

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-hairline">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  disabled={uploading}
                  className="px-4 py-2 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink-2 text-xs font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading || !uploadFile || !questionCol}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-strong text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploading ? (
                    <span>Uploading...</span>
                  ) : (
                    <>
                      <Check className="h-3.5 w-3.5" />
                      <span>Save & Add Suite</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Error: </span>
            {error}
          </div>
        </div>
      )}

      {/* Retrieval Impact Check */}
      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink mb-1">
          <GitCompare className="h-4 w-4 text-accent" />
          <span>Retrieval Impact Check</span>
        </div>
        <p className="text-xs text-ink-3 mb-4">
          Run one query two ways on the same model -- grounded (through retrieval, reranking, and your
          documents) and direct (the model alone, no context) -- to see what retrieval actually
          contributes. Not saved to Ask history.
        </p>

        <div className="flex flex-col gap-3">
          <textarea
            rows={2}
            value={compareText}
            onChange={(e) => setCompareText(e.target.value)}
            placeholder="Ask something your documents can answer..."
            disabled={compareLoading}
            className="w-full resize-none rounded-xl border border-hairline bg-surface px-4 py-3 text-sm text-ink placeholder-ink-4 focus:outline-none focus:border-accent-line focus:ring-1 focus:ring-accent/40 transition-all"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-ink-3">Model</span>
              <div className="relative">
                <select
                  value={compareModel}
                  onChange={(e) => setCompareModel(e.target.value)}
                  disabled={compareLoading || models.length === 0}
                  className="appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-mono font-medium focus:outline-none focus:border-accent-line transition-colors cursor-pointer"
                >
                  {models.map((m) => (
                    <option key={m} value={m} className="bg-surface text-ink">
                      {m}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                  <span className="text-[10px]">▼</span>
                </div>
              </div>
            </div>
            <button
              type="button"
              onClick={onCompare}
              disabled={compareLoading || !compareText.trim() || !compareModel}
              className="flex items-center gap-2 rounded-lg bg-accent hover:bg-accent-strong text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              <GitCompare className="h-4 w-4" />
              <span>{compareLoading ? "Comparing..." : "Compare"}</span>
            </button>
          </div>
        </div>

        {compareError && (
          <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-rose-700 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold">Error: </span>
              {compareError}
            </div>
          </div>
        )}

        {compareResult && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <ComparePanel
              label="Grounded (with retrieval)"
              answer={compareResult.grounded.answer}
              badges={[
                compareResult.grounded.refused ? "refused" : null,
                compareResult.grounded.confidence != null
                  ? `confidence ${(compareResult.grounded.confidence * 100).toFixed(0)}%`
                  : null,
              ].filter(Boolean)}
              timings={[
                ["total", compareResult.grounded.latency_ms],
                ["rerank", compareResult.grounded.rerank_ms],
                ["llm", compareResult.grounded.generate_ms],
              ]}
            >
              <SourceList citations={compareResult.grounded.citations} title="Sources" compact />
            </ComparePanel>
            <ComparePanel
              label="Direct (no retrieval)"
              answer={compareResult.direct.answer}
              timings={[
                ["total", compareResult.direct.latency_ms],
                compareResult.direct.tokens_per_sec != null
                  ? ["tok/s", compareResult.direct.tokens_per_sec, ""]
                  : null,
              ].filter(Boolean)}
            />
          </div>
        )}
      </div>

      {/* Current Run Card */}
      {job && (
        <Card title="Current Run" className="border-accent-line bg-accent-soft">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-3">
              <StatusBadge status={job.status} />
              <span className="text-xs font-medium text-ink-2">{job.message}</span>
            </div>
            <RunParams params={job.params} />
          </div>

          {running && <ProgressBar value={job.progress} />}
          {job.result && <ResultsTable results={job.result} partial={job.status !== "done"} />}
          {job.status === "failed" && (
            <pre className="mt-3 text-xs text-rose-700 bg-rose-50 p-3 rounded-lg border border-rose-200 whitespace-pre-wrap font-mono">
              {job.error}
            </pre>
          )}
        </Card>
      )}

      {/* History Card */}
      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">
        <h2 className="text-base font-bold text-ink mb-1">Benchmark History</h2>
        <p className="text-xs text-ink-3 mb-4">Past evaluations from this session, newest first</p>

        {history.length === 0 ? (
          <div className="py-12 text-center text-ink-4 text-xs">
            No benchmark runs yet. Configure parameters above and click "Run Benchmark".
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-ink-3 border-b border-hairline pb-2">
                  <th className="pb-2.5 pr-4 font-semibold">Date & Time</th>
                  <th className="pb-2.5 pr-4 font-semibold">Model</th>
                  <th className="pb-2.5 pr-4 font-semibold">Status</th>
                  <th className="pb-2.5 pr-4 font-semibold">Config</th>
                  <th className="pb-2.5 pr-4 font-semibold">Results Summary</th>
                  <th className="pb-2.5 text-right font-semibold"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {history.map((h) => {
                  const isExpanded = expandedIds.has(h.id);
                  return (
                    <Fragment key={h.id}>
                      <tr
                        onClick={() => toggleExpand(h.id)}
                        className="hover:bg-surface-2 cursor-pointer transition-colors"
                      >
                        <td className="py-3 pr-4 whitespace-nowrap text-ink-3">
                          {new Date(h.created_at * 1000).toLocaleString()}
                        </td>
                        <td className="py-3 pr-4 whitespace-nowrap">
                          <span className="font-mono font-medium px-2 py-0.5 rounded-md bg-accent-soft text-accent border border-accent-line">
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
                            <span className="text-ink-4 italic">{h.message || "—"}</span>
                          )}
                        </td>
                        <td className="py-3 text-right whitespace-nowrap">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpand(h.id);
                            }}
                            className="text-accent hover:text-accent font-medium inline-flex items-center gap-1"
                          >
                            <span>{isExpanded ? "Hide" : "Details"}</span>
                            {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                          </button>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr className="bg-surface-2">
                          <td colSpan={6} className="px-4 py-3.5">
                            <div className="flex flex-col gap-2">
                              {h.message && h.status !== "done" && (
                                <div className="text-xs text-ink-3 italic mb-1">
                                  Status message: {h.message}
                                </div>
                              )}
                              {h.result ? (
                                <ResultsTable results={h.result} partial={h.status !== "done"} />
                              ) : (
                                <p className="text-xs text-ink-4">No score data recorded.</p>
                              )}
                              {h.status === "failed" && h.error && (
                                <pre className="mt-2 text-xs text-rose-700 bg-rose-50 p-3 rounded-lg border border-rose-200 whitespace-pre-wrap font-mono">
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
  if (suitesWithData.length === 0) return <span className="text-xs text-ink-4">0 answered</span>;

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {suitesWithData.map((r) => {
        const shortName = r.name
          .replace(/\.csv$/i, "")
          .replace(/_/g, " ");

        const metricEntries = Object.entries(r).filter(([k]) => k !== "name" && k !== "n");

        return (
          <span key={r.name} className="inline-flex items-center gap-1.5 bg-surface px-2 py-0.5 rounded-md border border-hairline">
            <span className="font-semibold text-ink-2 capitalize">{shortName}:</span>
            <span className="text-ink-3">
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
        <span className="px-2 py-0.5 rounded-md bg-surface text-ink-2 border border-hairline">
          {workers}w
        </span>
      )}
      <span className="px-2 py-0.5 rounded-md bg-surface text-ink-2 border border-hairline">
        sample: {sample != null && sample !== "" ? sample : "all"}
      </span>
      {chunk_size != null && (
        <span className="px-2 py-0.5 rounded-md bg-surface text-ink-2 border border-hairline">
          chunk: {chunk_size}
        </span>
      )}
      {use_cache != null && (
        <span className="px-2 py-0.5 rounded-md bg-surface text-ink-2 border border-hairline">
          {use_cache ? "cache" : "no-cache"}
        </span>
      )}
    </div>
  );
}

function ComparePanel({ label, answer, badges = [], timings = [], children }) {
  return (
    <div className="rounded-xl border border-hairline bg-surface-2 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-ink">{label}</span>
        {badges.map((badge) => (
          <span
            key={badge}
            className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
          >
            {badge}
          </span>
        ))}
      </div>
      <p className="text-sm text-ink whitespace-pre-wrap leading-relaxed">{answer}</p>
      <div className="flex flex-wrap items-center gap-2 mt-3 text-[11px] font-mono text-ink-3">
        {timings.map(([tag, value, unit = "ms"]) => (
          value == null ? null : (
            <span
              key={tag}
              className="flex items-center gap-1 bg-surface px-2 py-0.5 rounded-md border border-hairline"
            >
              {tag === "total" ? <Clock className="h-3 w-3 text-ink-4" /> : <Zap className="h-3 w-3 text-ink-4" />}
              {tag} {unit === "ms" ? Math.round(value) : value}{unit}
            </span>
          )
        ))}
      </div>
      {children}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function ProgressBar({ value }) {
  return (
    <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden mb-3">
      <div
        className="h-full rounded-full bg-gradient-to-r from-accent to-accent-strong transition-all duration-300"
        style={{ width: `${Math.round((value ?? 0) * 100)}%` }}
      />
    </div>
  );
}

function ResultsTable({ results, partial = false }) {
  return (
    <div className="overflow-x-auto">
      {partial && (
        <p className="text-xs text-amber-700 font-medium mb-2">
          Partial — scored over the questions answered so far.
        </p>
      )}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-ink-3 border-b border-hairline pb-2">
            <th className="py-2 pr-4 font-semibold">Test Suite</th>
            <th className="py-2 pr-4 font-semibold">n</th>
            <th className="py-2 pr-4 font-semibold">Metrics</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-hairline font-mono">
          {results.map((r) => (
            <tr key={r.name} className="hover:bg-surface-2">
              <td className="py-2 pr-4 font-semibold text-ink">{r.name}</td>
              <td className="py-2 pr-4 text-ink-3">{r.n}</td>
              <td className="py-2 pr-4 text-ink-2">
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

