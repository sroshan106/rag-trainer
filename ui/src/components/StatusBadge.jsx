const STYLES = {
  pending: "bg-surface-2 text-ink-2 border-hairline",
  running: "bg-accent-soft text-accent border-accent-line animate-pulse",
  done: "bg-emerald-50 text-emerald-700 border-emerald-200",
  failed: "bg-rose-50 text-rose-700 border-rose-200",
  cancelled: "bg-amber-50 text-amber-700 border-amber-200",
};

const DOT_COLORS = {
  pending: "bg-ink-4",
  running: "bg-accent animate-ping",
  done: "bg-emerald-400",
  failed: "bg-rose-400",
  cancelled: "bg-amber-400",
};

export default function StatusBadge({ status }) {
  const badgeClass = STYLES[status] || "bg-surface-2 text-ink-2 border-hairline";
  const dotClass = DOT_COLORS[status] || "bg-ink-4";

  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-0.5 rounded-full border ${badgeClass}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
      <span className="capitalize">{status}</span>
    </span>
  );
}

