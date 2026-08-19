// Citations render as links because the corpus is web-scraped and the URL is
// the only part of a source that means anything outside this machine. Rows
// without a usable URL still render, as plain text rather than a dead link.
//
// `compact` drops the heading/border, for use inside something that already
// has its own layout -- e.g. a table cell.
export default function SourceList({ sources, title = "Sources", compact = false }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className={compact ? "" : "mt-4 pt-3 border-t border-neutral-200 dark:border-neutral-800"}>
      {!compact && (
        <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500 mb-2">
          {title}
        </h3>
      )}
      <ul className="text-xs space-y-1">
        {sources.map((src, i) => {
          const isLink = typeof src === "string" && /^https?:\/\//.test(src);
          return (
            <li key={i} className="flex gap-2">
              <span className="text-neutral-400 shrink-0">[{i + 1}]</span>
              {isLink ? (
                <a
                  href={src}
                  target="_blank"
                  rel="noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline break-all"
                >
                  {src}
                </a>
              ) : (
                <span className="break-all text-neutral-500">{src}</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
