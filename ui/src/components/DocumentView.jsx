import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ExternalLink, FileText, Loader2 } from "lucide-react";
import { documentMeta, documentUnit, documentUnits } from "../api.js";

const PAGE_SIZE = 25;

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${["B", "KB", "MB", "GB"][i]}`;
}

function Unit({ unit, highlighted }) {
  return (
    <div
      className={`rounded-xl border p-3.5 transition-colors ${
        highlighted
          ? "border-blue-700/80 bg-blue-950/30"
          : "border-slate-800/80 bg-slate-900/40"
      }`}
    >
      <div className="flex items-center justify-between gap-3 mb-2">
        <span className="text-[11px] font-mono font-semibold text-blue-400 uppercase tracking-wide">
          {unit.label}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          {unit.fields &&
            Object.entries(unit.fields).map(([field, value]) =>
              value === unit.url ? null : (
                <span key={field} className="text-[11px] font-mono text-slate-500" title={field}>
                  {field}: {value}
                </span>
              )
            )}
          {!unit.fields && unit.key && (
            <span className="text-[11px] font-mono text-slate-500" title="Dataset identifier">
              id {unit.key}
            </span>
          )}
          {unit.url && (
            <a
              href={unit.url}
              target="_blank"
              rel="noreferrer noopener"
              className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-blue-400 transition-colors"
            >
              <ExternalLink className="h-3 w-3" />
              Original
            </a>
          )}
        </div>
      </div>
      <p className="text-sm text-slate-300 whitespace-pre-wrap break-words leading-relaxed">
        {unit.text}
      </p>
    </div>
  );
}

/**
 * Renders a stored document as the ingest pipeline read it.
 *
 * `focusIndex` is a unit index from a citation. That unit is fetched directly
 * rather than paged to: it can sit 38,000 rows into a file, and paging there
 * would mean dozens of round trips.
 */
export default function DocumentView({ fileId, focusIndex = null, compact = false }) {
  const [meta, setMeta] = useState(null);
  const [units, setUnits] = useState([]);
  const [focused, setFocused] = useState(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [exhausted, setExhausted] = useState(false);

  const focusRef = useRef(null);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError(null);
    setUnits([]);
    setFocused(null);
    setOffset(0);
    setExhausted(false);

    const wanted =
      focusIndex === null
        ? Promise.resolve(null)
        : documentUnit(fileId, focusIndex).catch(() => null);

    Promise.all([documentMeta(fileId), wanted, documentUnits(fileId, { limit: PAGE_SIZE })])
      .then(([info, focusUnit, page]) => {
        if (ignore) return;
        setMeta(info);
        setFocused(focusUnit);
        setUnits(page);
        setOffset(page.length);
        setExhausted(page.length < PAGE_SIZE);
      })
      .catch((err) => !ignore && setError(err.message))
      .finally(() => !ignore && setLoading(false));

    return () => {
      ignore = true;
    };
  }, [fileId, focusIndex]);

  useEffect(() => {
    if (focused && focusRef.current) {
      focusRef.current.scrollIntoView({ block: "nearest" });
    }
  }, [focused]);

  const loadMore = useCallback(() => {
    setLoadingMore(true);
    documentUnits(fileId, { offset, limit: PAGE_SIZE })
      .then((page) => {
        setUnits((current) => [...current, ...page]);
        setOffset((current) => current + page.length);
        if (page.length < PAGE_SIZE) setExhausted(true);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoadingMore(false));
  }, [fileId, offset]);

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 py-12 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading document...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-rose-800/80 bg-rose-950/40 p-4 text-sm text-rose-300">
        <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
        <div>{error}</div>
      </div>
    );
  }

  // The focused unit may also appear in the first page; show it once.
  const rest = focused ? units.filter((u) => u.index !== focused.index) : units;

  return (
    <div className="flex flex-col gap-4">
      {meta && (
        <div className="flex items-center gap-3 pb-3 border-b border-slate-800/80">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-950/70 border border-blue-800/50 text-blue-400">
            <FileText className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-sm text-slate-100 truncate">
              {meta.filename}
            </div>
            <div className="flex flex-wrap items-center gap-x-2 text-xs text-slate-400 mt-0.5">
              <span>
                {meta.units?.toLocaleString() ?? "?"} {meta.unit_kind}s
              </span>
              <span>•</span>
              <span>{meta.chunks.toLocaleString()} chunks</span>
              <span>•</span>
              <span>{formatBytes(meta.size_bytes)}</span>
            </div>
            {meta.index_columns && meta.index_columns.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 mt-2">
                <span className="text-[11px] text-slate-400 font-medium">Indexed fields:</span>
                {meta.index_columns.map((col) => (
                  <span
                    key={col}
                    className="px-2 py-0.5 rounded-md bg-blue-950/60 text-blue-300 font-mono text-[11px] border border-blue-800/60"
                  >
                    {col}
                  </span>
                ))}
              </div>
            )}
            {meta.citation_columns && meta.citation_columns.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                <span className="text-[11px] text-slate-400 font-medium">Cited by:</span>
                {meta.citation_columns.map((col) => (
                  <span
                    key={col}
                    className="px-2 py-0.5 rounded-md bg-emerald-950/60 text-emerald-300 font-mono text-[11px] border border-emerald-800/60"
                  >
                    {col}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {focused && (
        <div ref={focusRef} className="flex flex-col gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            Cited {focused.kind}
          </span>
          <Unit unit={focused} highlighted />
        </div>
      )}

      {focusIndex !== null && !focused && (
        <div className="rounded-xl border border-amber-800/60 bg-amber-950/30 p-3 text-xs text-amber-300">
          The cited {meta?.unit_kind ?? "unit"} {focusIndex} is no longer in this document.
        </div>
      )}

      {!compact && (
        <div className="flex flex-col gap-2.5">
          {focused && rest.length > 0 && (
            <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mt-2">
              Document
            </span>
          )}
          {rest.map((unit) => (
            <Unit key={unit.index} unit={unit} />
          ))}

          {rest.length === 0 && !focused && (
            <div className="py-12 text-center text-xs text-slate-500">
              This document has no readable content.
            </div>
          )}

          {!exhausted && (
            <button
              type="button"
              onClick={loadMore}
              disabled={loadingMore}
              className="mt-2 self-center rounded-lg border border-slate-700/80 bg-slate-900/90 px-4 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white transition-colors disabled:opacity-50"
            >
              {loadingMore ? "Loading..." : `Load ${PAGE_SIZE} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
