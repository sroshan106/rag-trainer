import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Copy,
  Check,
  RotateCcw,
  Trash2,
  Clock,
  Cpu,
  AlertTriangle,
  XCircle,
  HelpCircle,
  Sparkles,
} from "lucide-react";
import SourceList from "./SourceList.jsx";
import StageIndicator from "./StageIndicator.jsx";

function formatWhen(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const s = Math.floor((new Date() - date) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return date.toLocaleDateString();
}

function Timing({ entry }) {
  if (entry.latency_ms == null) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-ink-3">
      <span className="flex items-center gap-1 bg-surface px-2 py-0.5 rounded-md border border-hairline">
        <Clock className="h-3 w-3 text-ink-4" />
        {Math.round(entry.latency_ms)}ms total
      </span>
      {entry.rerank_ms != null && (
        <span className="bg-surface px-2 py-0.5 rounded-md border border-hairline">
          rerank {Math.round(entry.rerank_ms)}ms
        </span>
      )}
      {entry.generate_ms != null && (
        <span className="bg-surface px-2 py-0.5 rounded-md border border-hairline">
          llm {Math.round(entry.generate_ms)}ms
        </span>
      )}
    </div>
  );
}

export default function Exchange({
  entry,
  live = false,
  stage,
  stageDetail,
  models = [],
  onCancel,
  onRerun,
  onDelete,
  onReuse,
}) {
  const [rerunModel, setRerunModel] = useState(entry.model ?? models[0] ?? "");
  const [copied, setCopied] = useState(false);

  const streaming = live && Boolean(entry.answer);
  const when = formatWhen(entry.created_at);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(entry.answer ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  return (
    <section className="rounded-2xl border border-hairline bg-surface p-5 shadow-card transition-all duration-200">
      <div className="flex items-start justify-between gap-4 pb-3 border-b border-hairline">
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent border border-accent-line mt-0.5">
            <HelpCircle className="h-4 w-4" />
          </div>
          <button
            type="button"
            onClick={() => onReuse?.(entry.query)}
            title="Click to put this question back in the input box"
            className="text-left text-sm font-semibold text-ink hover:text-accent transition-colors leading-snug cursor-pointer"
          >
            {entry.query}
          </button>
        </div>

        <div className="shrink-0 flex items-center gap-2 text-right text-xs">
          {entry.model && (
            <span className="flex items-center gap-1 font-mono text-[11px] font-medium px-2 py-0.5 rounded-md bg-accent-soft text-accent border border-accent-line">
              <Cpu className="h-3 w-3" />
              {entry.model}
            </span>
          )}
          {when && <span className="text-[11px] text-ink-3">{when}</span>}
        </div>
      </div>

      {live && (
        <div className="mt-3.5">
          <StageIndicator stage={stage} detail={stageDetail} streaming={streaming} />
        </div>
      )}

      {entry.status === "pending" && !live && (
        <div className="mt-3 flex items-center gap-2 p-3 rounded-xl bg-surface-2 border border-hairline text-xs text-ink-3 animate-pulse">
          <Sparkles className="h-4 w-4 text-accent" />
          <span>Processing query in background...</span>
        </div>
      )}

      {entry.answer && (
        <div className="mt-3.5 text-sm text-ink leading-relaxed prose prose-invert max-w-none prose-p:my-2 prose-pre:bg-surface-2 prose-pre:border prose-pre:border-hairline prose-code:text-accent prose-code:bg-surface prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.answer}</ReactMarkdown>
          {streaming && <span className="inline-block h-4 w-1.5 ml-1 bg-accent animate-pulse align-middle" />}
        </div>
      )}

      {entry.status === "error" && (
        <div className="mt-3 flex items-center gap-2 p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700">
          <XCircle className="h-4 w-4 text-rose-700 shrink-0" />
          <span>This query encountered an error before producing an answer.</span>
        </div>
      )}

      {entry.status === "cancelled" && (
        <div className="mt-3 flex items-center gap-2 p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-700">
          <AlertTriangle className="h-4 w-4 text-amber-700 shrink-0" />
          <span>Cancelled{entry.answer ? " mid-answer." : " before generation."}</span>
        </div>
      )}

      {entry.refused && (
        <div className="mt-3 flex items-center gap-2 p-3 rounded-xl bg-surface-2 border border-hairline text-xs text-ink-3">
          <AlertTriangle className="h-4 w-4 text-ink-4 shrink-0" />
          <span>No chunk cleared the relevance grader — the knowledge base does not cover this question.</span>
        </div>
      )}

      <SourceList citations={entry.citations} />

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-hairline pt-3 text-xs">
        <Timing entry={entry} />

        <div className="flex items-center gap-2 ml-auto">
          {live ? (
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink text-xs font-medium transition-colors"
            >
              Cancel
            </button>
          ) : (
            <>
              {entry.answer && (
                <button
                  type="button"
                  onClick={onCopy}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-2 text-ink-2 hover:text-ink transition-colors"
                >
                  {copied ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-700" />
                      <span className="text-emerald-700">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3 text-ink-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              )}

              {onRerun && models.length > 0 && (
                <div className="flex items-center gap-1 bg-surface p-0.5 rounded-lg border border-hairline">
                  <select
                    value={rerunModel}
                    onChange={(e) => setRerunModel(e.target.value)}
                    className="bg-transparent text-ink-2 px-2 py-0.5 text-xs focus:outline-none cursor-pointer"
                  >
                    {models.map((m) => (
                      <option key={m} value={m} className="bg-surface text-ink">
                        {m}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => onRerun(entry.query, rerunModel)}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-surface-2 hover:bg-surface-3 text-ink text-xs transition-colors"
                  >
                    <RotateCcw className="h-3 w-3" />
                    <span>Re-run</span>
                  </button>
                </div>
              )}

              {onDelete && (
                <button
                  type="button"
                  onClick={() => onDelete(entry)}
                  className="p-1 rounded-lg text-ink-4 hover:text-rose-700 hover:bg-rose-50 transition-colors"
                  title="Delete from history"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

