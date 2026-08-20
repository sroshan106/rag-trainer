import { Search, CheckCircle2, Sparkles, ArrowRight } from "lucide-react";

const STAGES = [
  { key: "retrieve", label: "Retrieving", icon: Search },
  { key: "grade", label: "Grading", icon: CheckCircle2 },
  { key: "generate", label: "Generating", icon: Sparkles },
];

function describe(stage, detail) {
  if (!detail) return null;
  if (stage === "grade") return `${detail.retrieved} candidates`;
  if (stage === "generate") return `${detail.kept} of ${detail.retrieved} kept`;
  return null;
}

export default function StageIndicator({ stage, detail, streaming }) {
  const current = STAGES.findIndex((s) => s.key === stage);
  const note = describe(stage, detail);

  return (
    <div className="flex flex-wrap items-center gap-2 p-2.5 rounded-xl border border-slate-800 bg-slate-900/60 text-xs">
      {STAGES.map((s, i) => {
        const Icon = s.icon;
        const done = i < current || (streaming && s.key === "generate");
        const active = i === current && !done;

        return (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && <ArrowRight className="h-3 w-3 text-slate-600" />}
            <span
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-all ${
                active
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/40 font-medium"
                  : done
                    ? "bg-emerald-950/40 text-emerald-400 border border-emerald-800/40"
                    : "text-slate-500 bg-slate-900/40 border border-slate-800/60"
              }`}
            >
              <Icon
                className={`h-3.5 w-3.5 ${
                  active ? "animate-pulse text-blue-400" : done ? "text-emerald-400" : "text-slate-500"
                }`}
              />
              <span>{s.label}</span>
              {active && <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-ping" />}
            </span>
          </div>
        );
      })}
      {note && (
        <span className="ml-auto text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-md border border-slate-700/60">
          {note}
        </span>
      )}
    </div>
  );
}

