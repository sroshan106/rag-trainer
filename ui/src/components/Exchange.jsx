import { useState } from "react";
import SourceList from "./SourceList.jsx";
import StageIndicator from "./StageIndicator.jsx";

// One question and its answer -- the unit the Ask view is a list of. The same
// component renders the in-flight exchange and every stored one, so a query
// doesn't visibly change shape the moment it finishes.

function formatWhen(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function Timing({ entry }) {
  if (entry.latency_ms == null) return null;
  const parts = [`${Math.round(entry.latency_ms)} ms total`];
  if (entry.rerank_ms != null) parts.push(`rerank ${Math.round(entry.rerank_ms)} ms`);
  if (entry.generate_ms != null) parts.push(`llm ${Math.round(entry.generate_ms)} ms`);
  return <span className="text-neutral-500">{parts.join("  ·  ")}</span>;
}

const ACTION = "text-xs text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100";

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
  const pending = entry.status === "pending";
  const failed = entry.status === "error";
  const cancelled = entry.status === "cancelled";

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(entry.answer ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard is blocked outside a secure context -- nothing to recover.
    }
  }

  return (
    <section className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
      <div className="flex items-start justify-between gap-4">
        <button
          type="button"
          onClick={() => onReuse?.(entry.query)}
          title="Put this question back in the box"
          className="text-left text-sm font-medium leading-snug hover:underline"
        >
          {entry.query}
        </button>
        <div className="shrink-0 text-[11px] text-neutral-400 text-right">
          {entry.model && <div className="font-mono">{entry.model}</div>}
          {formatWhen(entry.created_at) && <div>{formatWhen(entry.created_at)}</div>}
        </div>
      </div>

      {live && (
        <div className="mt-3">
          <StageIndicator stage={stage} detail={stageDetail} streaming={streaming} />
        </div>
      )}
      {pending && !live && (
        // A pending row this client isn't streaming -- another tab, or the
        // CLI. There are no stage events to follow, only the row's status.
        <p className="mt-3 animate-pulse text-xs text-neutral-500">
          Running (started elsewhere)…
        </p>
      )}

      {entry.answer && (
        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">
          {entry.answer}
          {streaming && (
            <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-neutral-500 align-text-bottom" />
          )}
        </p>
      )}

      {failed && (
        <p className="mt-3 text-sm text-red-600 dark:text-red-400">
          This query failed before it produced an answer.
        </p>
      )}
      {cancelled && (
        <p className="mt-2 text-xs text-neutral-500">
          Cancelled{entry.answer ? " mid-answer." : " before any text arrived."}
        </p>
      )}
      {entry.refused && (
        <p className="mt-2 text-xs text-neutral-500">
          No sources survived grading — the collection doesn&apos;t cover this.
        </p>
      )}

      <SourceList sources={entry.sources} />

      <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-neutral-100 dark:border-neutral-800 pt-2 text-xs">
        <Timing entry={entry} />
        <div className="ml-auto flex items-center gap-3">
          {live ? (
            <button type="button" onClick={onCancel} className={ACTION}>
              Cancel
            </button>
          ) : (
            <>
              {entry.answer && (
                <button type="button" onClick={onCopy} className={ACTION}>
                  {copied ? "Copied" : "Copy"}
                </button>
              )}
              {onRerun && models.length > 0 && (
                <span className="flex items-center gap-1">
                  <select
                    value={rerunModel}
                    onChange={(e) => setRerunModel(e.target.value)}
                    className="rounded border border-neutral-200 dark:border-neutral-800 bg-transparent px-1 py-0.5 text-[11px] dark:[color-scheme:dark]"
                  >
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => onRerun(entry.query, rerunModel)}
                    className={ACTION}
                  >
                    Re-run
                  </button>
                </span>
              )}
              {onDelete && (
                <button
                  type="button"
                  onClick={() => onDelete(entry)}
                  className="text-xs text-neutral-500 hover:text-red-600 dark:hover:text-red-400"
                >
                  Delete
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
