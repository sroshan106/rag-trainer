const STYLES = {
  pending: "bg-slate-800 text-slate-300 border-slate-700",
  running: "bg-blue-950/80 text-blue-400 border-blue-800/80 animate-pulse",
  done: "bg-emerald-950/80 text-emerald-300 border-emerald-800/70",
  failed: "bg-rose-950/80 text-rose-300 border-rose-800/70",
  cancelled: "bg-amber-950/80 text-amber-300 border-amber-800/70",
};

const DOT_COLORS = {
  pending: "bg-slate-400",
  running: "bg-blue-400 animate-ping",
  done: "bg-emerald-400",
  failed: "bg-rose-400",
  cancelled: "bg-amber-400",
};

export default function StatusBadge({ status }) {
  const badgeClass = STYLES[status] || "bg-slate-800 text-slate-300 border-slate-700";
  const dotClass = DOT_COLORS[status] || "bg-slate-400";

  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-0.5 rounded-full border ${badgeClass}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
      <span className="capitalize">{status}</span>
    </span>
  );
}

