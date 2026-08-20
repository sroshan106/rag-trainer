import { useEffect, useRef, useState } from "react";
import {
  DownloadCloud,
  CheckCircle2,
  Cpu,
  Database,
  Layers,
  Zap,
  HardDrive,
  AlertCircle,
  AlertTriangle,
  Trash2,
  X,
  Loader2,
} from "lucide-react";
import StatusBadge from "../components/StatusBadge.jsx";
import { listModelCatalog, pullModel, pullHistory, pollJob, deleteModel } from "../api.js";

const DEFAULT_MODEL_METADATA = {
  "llama3.2:3b": {
    min_vram: "3 GB",
    disk_size: "~2.0 GB",
    params: "3.2B",
    context: "128k ctx",
    description: "Fast, balanced general QA with low latency and strong instruction following.",
  },
  "llama3.2:1b": {
    min_vram: "1.5 GB",
    disk_size: "~1.3 GB",
    params: "1.2B",
    context: "128k ctx",
    description: "Ultra-lightweight edge model with minimal memory footprint.",
  },
  "qwen3:4b": {
    min_vram: "3.5 GB",
    disk_size: "~2.6 GB",
    params: "4B",
    context: "32k ctx",
    description: "Strong reasoning, multilingual support, and coding capabilities.",
  },
  "qwen2.5:3b": {
    min_vram: "3 GB",
    disk_size: "~1.9 GB",
    params: "3.1B",
    context: "32k ctx",
    description: "High instruction following, structured output, and coding ability.",
  },
  "gemma2:2b": {
    min_vram: "2.5 GB",
    disk_size: "~1.6 GB",
    params: "2.6B",
    context: "8k ctx",
    description: "Google lightweight conversational and knowledge retrieval model.",
  },
  "phi3.5": {
    min_vram: "3.5 GB",
    disk_size: "~2.2 GB",
    params: "3.8B",
    context: "128k ctx",
    description: "Microsoft compact reasoning model with high benchmark performance.",
  },
  "nomic-embed-text": {
    min_vram: "500 MB",
    disk_size: "~274 MB",
    params: "137M",
    context: "8192 ctx • 768 dim",
    description: "Default embedding model for vectorstore retrieval with large context window.",
  },
  "all-minilm": {
    min_vram: "250 MB",
    disk_size: "~120 MB",
    params: "33M",
    context: "256 ctx • 384 dim",
    description: "Extremely fast, lightweight sentence embeddings.",
  },
  "bge-m3": {
    min_vram: "1.5 GB",
    disk_size: "~1.2 GB",
    params: "568M",
    context: "8192 ctx • 1024 dim",
    description: "Multilingual embeddings supporting dense, sparse, and multi-vector search.",
  },
  "bge-small": {
    min_vram: "200 MB",
    disk_size: "~67 MB",
    params: "24M",
    context: "512 ctx • 384 dim",
    description: "Minimal memory footprint embedding model for resource-constrained setups.",
  },
  "mxbai-embed-large": {
    min_vram: "1.0 GB",
    disk_size: "~670 MB",
    params: "335M",
    context: "512 ctx • 1024 dim",
    description: "High retrieval accuracy representation model for English corpus.",
  },
  "snowflake-arctic-embed": {
    min_vram: "1.0 GB",
    disk_size: "~669 MB",
    params: "335M",
    context: "512 ctx • 1024 dim",
    description: "Enterprise-grade retrieval embedding model by Snowflake.",
  },
  "cross-encoder/ms-marco-MiniLM-L-6-v2": {
    min_vram: "300 MB",
    disk_size: "~80 MB",
    params: "22.7M",
    context: "512 ctx",
    description: "Fast cross-encoder reranker for passage re-ranking and noise filtering.",
  },
  "cross-encoder/ms-marco-MiniLM-L-12-v2": {
    min_vram: "400 MB",
    disk_size: "~130 MB",
    params: "33M",
    context: "512 ctx",
    description: "12-layer variant of MiniLM offering higher precision while remaining fast.",
  },
  "BAAI/bge-reranker-base": {
    min_vram: "1.5 GB",
    disk_size: "~1.1 GB",
    params: "278M",
    context: "512 ctx",
    description: "High-accuracy multilingual cross-encoder reranker (100+ languages).",
  },
  "BAAI/bge-reranker-large": {
    min_vram: "2.5 GB",
    disk_size: "~2.2 GB",
    params: "560M",
    context: "512 ctx",
    description: "Top-tier accuracy reranker for demanding retrieval benchmarks.",
  },
  "mixedbread-ai/mxbai-rerank-base-v1": {
    min_vram: "800 MB",
    disk_size: "~500 MB",
    params: "135M",
    context: "512 ctx",
    description: "State-of-the-art English reranking precision designed for RAG pipelines.",
  },
  "jinaai/jina-reranker-v2-base-multilingual": {
    min_vram: "1.2 GB",
    disk_size: "~560 MB",
    params: "278M",
    context: "8192 ctx",
    description: "Supports long-context reranking up to 8k tokens and multi-language queries.",
  },
};

const DEFAULT_RERANK_CATALOG = [
  "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "cross-encoder/ms-marco-MiniLM-L-12-v2",
  "BAAI/bge-reranker-base",
  "BAAI/bge-reranker-large",
  "mixedbread-ai/mxbai-rerank-base-v1",
  "jinaai/jina-reranker-v2-base-multilingual",
];

const TABS = [
  { key: "chat", label: "Chat Models", icon: Cpu },
  { key: "embed", label: "Embeddings", icon: Database },
  { key: "rerank", label: "Rerankers", icon: Layers },
];

export default function Settings() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [jobs, setJobs] = useState({});
  const [activeTab, setActiveTab] = useState("chat");
  const [modelToDelete, setModelToDelete] = useState(null);
  const [deletingModel, setDeletingModel] = useState(null);
  const cancelledRef = useRef(false);

  function refresh() {
    listModelCatalog().then(setData).catch((err) => setError(err.message));
  }

  useEffect(() => {
    cancelledRef.current = false;
    refresh();

    pullHistory()
      .then((history) => {
        for (const j of history) {
          if (["pending", "running"].includes(j.status) && j.result?.model) {
            const m = j.result.model;
            setJobs((prev) => ({ ...prev, [m]: j }));
            pollJob(j.id, {
              onUpdate: (up) => setJobs((prev) => ({ ...prev, [m]: up })),
              isCancelled: () => cancelledRef.current,
            }).then((finalJob) => {
              if (finalJob?.status === "done" && !cancelledRef.current) {
                refresh();
              }
            });
          }
        }
      })
      .catch(() => {});

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

  async function confirmDelete(model) {
    setError(null);
    setDeletingModel(model);
    try {
      await deleteModel(model);
      setModelToDelete(null);
      refresh();
    } catch (err) {
      setError(err.message);
      setModelToDelete(null);
    } finally {
      setDeletingModel(null);
    }
  }

  const infoMap = { ...DEFAULT_MODEL_METADATA, ...(data?.model_info || {}) };
  const rerankModels = data?.rerank_models || DEFAULT_RERANK_CATALOG;

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-100">Model Management</h1>
        <p className="text-sm text-slate-400 mt-1">
          Download and manage LLM generators, embedding models, and cross-encoders
        </p>
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

      {/* Tab Selector */}
      <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[#111726]/90 border border-slate-800 self-start">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? "bg-blue-600 text-white shadow-sm shadow-blue-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Panels */}
      {activeTab === "chat" && (
        <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
          <div className="mb-4 pb-3 border-b border-slate-800/80">
            <h2 className="text-base font-bold text-slate-100">Chat Generation Models</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Downloaded models are instantly available in Ask and Benchmark views
            </p>
          </div>
          <ModelTable
            rows={data?.catalog.map((m) => ({
              name: m,
              installed: data.installed.includes(m),
            }))}
            infoMap={infoMap}
            job={(m) => jobs[m]}
            onDownload={download}
            onDeleteRequest={(row, info) => setModelToDelete({ name: row.name, active: row.active, info })}
            deletingModel={deletingModel}
          />
        </div>
      )}

      {activeTab === "embed" && (
        <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
          <div className="mb-4 pb-3 border-b border-slate-800/80">
            <h2 className="text-base font-bold text-slate-100">Vector Embedding Models</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {data
                ? `Active vector embedding model: ${data.active_embed_model || "nomic-embed-text"}`
                : "Downloadable models for vectorizing document chunks"}
            </p>
          </div>
          <ModelTable
            rows={data?.embed_models.map((m) => ({
              name: m,
              installed: data.embed_installed.includes(m),
              active: (data.active_embed_model || "nomic-embed-text") === m,
            }))}
            infoMap={infoMap}
            job={(m) => jobs[m]}
            onDownload={download}
            onDeleteRequest={(row, info) => setModelToDelete({ name: row.name, active: row.active, info })}
            deletingModel={deletingModel}
          />
        </div>
      )}

      {activeTab === "rerank" && (
        <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
          <div className="mb-4 pb-3 border-b border-slate-800/80">
            <h2 className="text-base font-bold text-slate-100">Cross-Encoder Reranker Models</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {data
                ? `Status: ${data.rerank_enabled ? "Enabled" : "Disabled"} (Active model: ${data.active_rerank_model || data.rerank_model})`
                : "Pre-download cross-encoder models from HuggingFace to cache locally"}
            </p>
          </div>
          <ModelTable
            rows={rerankModels.map((m) => {
              const isInstalled = Array.isArray(data?.rerank_installed)
                ? data.rerank_installed.includes(m)
                : data?.rerank_installed && m === data?.rerank_model;
              return {
                name: m,
                installed: isInstalled,
                active: (data?.active_rerank_model || data?.rerank_model) === m,
              };
            })}
            infoMap={infoMap}
            job={(m) => jobs[m]}
            onDownload={download}
            onDeleteRequest={(row, info) => setModelToDelete({ name: row.name, active: row.active, info })}
            deletingModel={deletingModel}
          />
        </div>
      )}

      {/* Delete Confirmation Warning Modal */}
      {modelToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
          <div
            className="relative w-full max-w-md rounded-2xl border border-slate-800 bg-[#111726] p-6 shadow-2xl shadow-black/80"
            role="dialog"
            aria-modal="true"
          >
            <button
              type="button"
              onClick={() => setModelToDelete(null)}
              disabled={!!deletingModel}
              className="absolute top-4 right-4 text-slate-400 hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-800/60 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>

            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 shrink-0">
                <AlertTriangle className="h-6 w-6 text-amber-400" />
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-bold text-slate-100">Delete Model</h3>
                <p className="text-sm text-slate-300 mt-1">
                  Are you sure you want to delete <span className="font-mono font-semibold text-amber-300 bg-slate-800/80 px-1.5 py-0.5 rounded">{modelToDelete.name}</span> from local disk?
                </p>
                {modelToDelete.info?.disk_size && (
                  <p className="text-xs text-slate-400 mt-1.5">
                    This will remove approximately <span className="font-mono text-slate-300 font-medium">{modelToDelete.info.disk_size}</span> of disk space.
                  </p>
                )}
              </div>
            </div>

            {modelToDelete.active && (
              <div className="mt-4 flex items-start gap-2.5 p-3 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs">
                <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-amber-200">Active Model Warning:</span> This model is currently configured as active. Deleting it may cause operations to fail until another active model is configured.
                </div>
              </div>
            )}

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                type="button"
                disabled={!!deletingModel}
                onClick={() => setModelToDelete(null)}
                className="px-4 py-2 text-xs font-medium rounded-lg text-slate-300 hover:text-white bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 transition-all cursor-pointer disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!!deletingModel}
                onClick={() => confirmDelete(modelToDelete.name)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition-all shadow-sm shadow-rose-600/30 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {deletingModel ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Delete Model</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ModelTable({ rows, infoMap, job, onDownload, onDeleteRequest, deletingModel }) {
  if (!rows) return <div className="py-8 text-center text-xs text-slate-500">Loading catalog...</div>;
  return (
    <div className="flex flex-col divide-y divide-slate-800/70">
      {rows.map((row) => (
        <ModelRow
          key={row.name}
          row={row}
          info={infoMap?.[row.name]}
          job={job(row.name)}
          onDownload={onDownload}
          onDeleteRequest={onDeleteRequest}
          deletingModel={deletingModel}
        />
      ))}
    </div>
  );
}

function ModelRow({ row, info, job, onDownload, onDeleteRequest, deletingModel }) {
  const busy = job && ["pending", "running"].includes(job.status);
  const minVram = info?.min_vram;
  const diskSize = info?.disk_size;
  const params = info?.params;
  const context = info?.context;
  const description = info?.description;
  const isDeleting = deletingModel === row.name;

  return (
    <div className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-bold text-slate-100">{row.name}</span>

          {row.active && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
              Active
            </span>
          )}
          {minVram && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono font-medium bg-blue-950/60 text-blue-300 border border-blue-800/60">
              <Zap className="h-3 w-3" />
              {minVram}
            </span>
          )}
          {diskSize && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono text-slate-400 bg-slate-900/80 border border-slate-800">
              <HardDrive className="h-3 w-3 text-slate-500" />
              {diskSize}
            </span>
          )}
          {params && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono text-slate-400 bg-slate-900/80 border border-slate-800">
              {params}
            </span>
          )}
        </div>

        {(context || description) && (
          <div className="mt-1 text-xs text-slate-400 flex flex-wrap items-center gap-x-2 gap-y-0.5">
            {context && <span className="font-mono text-slate-300">{context}</span>}
            {context && description && <span className="text-slate-600">•</span>}
            {description && <span>{description}</span>}
          </div>
        )}

        {busy && (
          <div className="mt-2.5 max-w-sm">
            <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all"
                style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
              />
            </div>
            <div className="text-[11px] text-slate-400 mt-1 truncate">{job.message}</div>
          </div>
        )}

        {job && !busy && job.status !== "done" && (
          <div className="text-xs mt-1.5">
            <StatusBadge status={job.status} />
            {job.error && <span className="ml-2 text-rose-400">{job.error}</span>}
          </div>
        )}
      </div>

      <div className="flex-shrink-0 flex items-center gap-2 self-start sm:self-center">
        {row.installed ? (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/80">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              <span>Installed</span>
            </span>
            <button
              type="button"
              disabled={busy || isDeleting}
              onClick={() => onDeleteRequest?.(row, info)}
              title="Delete model from disk"
              className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-lg text-rose-400 hover:text-rose-200 bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/50 hover:border-rose-700/80 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
            >
              <Trash2 className="h-3.5 w-3.5 text-rose-400" />
              <span>{isDeleting ? "Deleting..." : "Delete"}</span>
            </button>
          </div>
        ) : (
          <button
            type="button"
            disabled={busy}
            onClick={() => onDownload(row.name)}
            className="inline-flex items-center gap-1.5 text-xs font-medium px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-sm shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            <DownloadCloud className="h-3.5 w-3.5" />
            <span>{busy ? "Downloading..." : "Download"}</span>
          </button>
        )}
      </div>
    </div>
  );
}


