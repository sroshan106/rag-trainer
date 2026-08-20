import { useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { listModelCatalog, pullModel, pullHistory, pollJob } from "../api.js";

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

    // Reconnect to in-flight pulls (survives page refresh mid-download).
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

  const infoMap = { ...DEFAULT_MODEL_METADATA, ...(data?.model_info || {}) };
  const rerankModels = data?.rerank_models || DEFAULT_RERANK_CATALOG;

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
          infoMap={infoMap}
          job={(m) => jobs[m]}
          onDownload={download}
        />
      </Card>

      <Card
        title="Embedding models"
        subtitle={
          data
            ? `Downloadable for vectorization and retrieval (active embedder: ${data.active_embed_model || "nomic-embed-text"}).`
            : "Downloadable for vectorization and retrieval."
        }
      >
        <ModelTable
          rows={data?.embed_models.map((m) => ({
            name: m,
            installed: data.embed_installed.includes(m),
            active: (data.active_embed_model || "nomic-embed-text") === m,
          }))}
          infoMap={infoMap}
          job={(m) => jobs[m]}
          onDownload={download}
        />
      </Card>

      <Card
        title="Reranker models"
        subtitle={
          data
            ? `${data.rerank_enabled ? "Enabled" : "Disabled"} via RAG_RERANK (active: ${data.active_rerank_model || data.rerank_model}). Pre-download any reranker from HuggingFace to cache locally.`
            : "Pre-download cross-encoder models from HuggingFace to cache locally."
        }
      >
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
        />
      </Card>
    </div>
  );
}

function ModelTable({ rows, infoMap, job, onDownload }) {
  if (!rows) return <div className="text-sm text-neutral-500">Loading...</div>;
  return (
    <div className="flex flex-col divide-y divide-neutral-200 dark:divide-neutral-800">
      {rows.map((row) => (
        <ModelRow
          key={row.name}
          row={row}
          info={infoMap?.[row.name]}
          job={job(row.name)}
          onDownload={onDownload}
        />
      ))}
    </div>
  );
}

function ModelRow({ row, info, job, onDownload }) {
  const busy = job && ["pending", "running"].includes(job.status);
  const minVram = info?.min_vram;
  const diskSize = info?.disk_size;
  const params = info?.params;
  const context = info?.context;
  const description = info?.description;

  return (
    <div className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-semibold text-neutral-900 dark:text-neutral-100">
            {row.name}
          </span>
          {row.active && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
              active
            </span>
          )}
          {minVram && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800">
              Min VRAM: {minVram}
            </span>
          )}
          {diskSize && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-normal bg-neutral-100 text-neutral-600 border border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700">
              Disk: {diskSize}
            </span>
          )}
          {params && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-normal bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
              {params}
            </span>
          )}
        </div>

        {(context || description) && (
          <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400 flex flex-wrap items-center gap-x-2 gap-y-0.5">
            {context && <span className="font-mono text-neutral-600 dark:text-neutral-300">{context}</span>}
            {context && description && <span className="text-neutral-300 dark:text-neutral-600">•</span>}
            {description && <span>{description}</span>}
          </div>
        )}

        {busy && (
          <div className="mt-2">
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
          <div className="text-xs mt-1.5">
            <StatusBadge status={job.status} />
            {job.error && <span className="ml-2 text-red-600 dark:text-red-400">{job.error}</span>}
          </div>
        )}
      </div>

      <div className="flex-shrink-0 flex items-center gap-2 self-start sm:self-center">
        {row.installed ? (
          <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-green-100 text-green-700 dark:bg-green-900/60 dark:text-green-300">
            installed
          </span>
        ) : (
          <button
            type="button"
            disabled={busy}
            onClick={() => onDownload(row.name)}
            className="text-sm font-medium px-3.5 py-1.5 rounded-md bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 hover:bg-neutral-800 dark:hover:bg-neutral-200 transition-colors disabled:opacity-50"
          >
            {busy ? "Downloading..." : "Download"}
          </button>
        )}
      </div>
    </div>
  );
}
