import { ExternalLink, FileText } from "lucide-react";

export default function SourceList({ sources, title = "Sources", compact = false }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className={compact ? "" : "mt-4 pt-3.5 border-t border-slate-800/80"}>
      {!compact && (
        <div className="flex items-center gap-2 mb-2.5">
          <FileText className="h-3.5 w-3.5 text-slate-400" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            {title} ({sources.length})
          </h3>
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        {sources.map((src, i) => {
          const isLink = typeof src === "string" && /^https?:\/\//.test(src);
          let hostname = "";
          if (isLink) {
            try {
              hostname = new URL(src).hostname.replace(/^www\./, "");
            } catch {
              hostname = src;
            }
          }

          return isLink ? (
            <a
              key={i}
              href={src}
              target="_blank"
              rel="noreferrer"
              title={src}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-slate-800 bg-slate-900/60 hover:bg-slate-800 hover:border-slate-700 text-xs text-blue-400 hover:text-blue-300 transition-all max-w-full"
            >
              <span className="font-mono text-[10px] text-slate-500 font-semibold">[{i + 1}]</span>
              <span className="truncate max-w-xs">{hostname || src}</span>
              <ExternalLink className="h-3 w-3 shrink-0 opacity-70" />
            </a>
          ) : (
            <div
              key={i}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-slate-800 bg-slate-900/60 text-xs text-slate-400 max-w-full"
            >
              <span className="font-mono text-[10px] text-slate-500 font-semibold">[{i + 1}]</span>
              <span className="truncate max-w-xs">{src}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

