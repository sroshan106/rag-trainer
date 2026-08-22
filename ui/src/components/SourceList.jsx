import { useState } from "react";
import { ExternalLink, FileText } from "lucide-react";
import DocumentModal from "./DocumentModal.jsx";

function hostnameOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}


export default function SourceList({ citations, sources, title = "Sources", compact = false }) {
  const [open, setOpen] = useState(null);

  const entries = (citations?.length ? citations : sources || []).map((entry, i) =>
    typeof entry === "string"
      ? { legacyUrl: entry, key: `legacy-${i}` }
      : { ...entry, key: `${entry.file_id}-${entry.unit_index}-${i}` }
  );

  if (entries.length === 0) return null;

  const chip =
    "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-hairline bg-surface-2 text-xs max-w-full transition-all";

  return (
    <div className={compact ? "" : "mt-4 pt-3.5 border-t border-hairline"}>
      {!compact && (
        <div className="flex items-center gap-2 mb-2.5">
          <FileText className="h-3.5 w-3.5 text-ink-3" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-3">
            {title} ({entries.length})
          </h3>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {entries.map((entry, i) => {
          const number = (
            <span className="font-mono text-[10px] text-ink-4 font-semibold">
              [{i + 1}]
            </span>
          );

          if (entry.legacyUrl) {
            return (
              <a
                key={entry.key}
                href={entry.legacyUrl}
                target="_blank"
                rel="noreferrer noopener"
                title={entry.legacyUrl}
                className={`${chip} text-accent hover:text-accent hover:bg-surface-2 hover:border-hairline`}
              >
                {number}
                <span className="truncate max-w-xs">{hostnameOf(entry.legacyUrl)}</span>
                <ExternalLink className="h-3 w-3 shrink-0 opacity-70" />
              </a>
            );
          }

          const fieldEntries = Object.entries(entry.fields || {});
          const fieldSummary = fieldEntries.map(([k, v]) => `${k}: ${v}`).join(", ");

          return (
            <button
              key={entry.key}
              type="button"
              onClick={() => setOpen(entry)}
              title={[`${entry.filename} — ${entry.label}`, fieldSummary].filter(Boolean).join("\n")}
              className={`${chip} text-ink-2 hover:text-ink hover:bg-surface-2 hover:border-hairline cursor-pointer`}
            >
              {number}
              <span className="truncate max-w-xs">{entry.filename}</span>
              <span className="shrink-0 font-mono text-[10px] text-accent">
                {entry.label}
              </span>
              {fieldEntries.length > 0 && (
                <span className="shrink-0 font-mono text-[10px] text-emerald-700 truncate max-w-[10rem]">
                  {fieldSummary}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {open && (
        <DocumentModal
          fileId={open.file_id}
          focusIndex={open.unit_index}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
