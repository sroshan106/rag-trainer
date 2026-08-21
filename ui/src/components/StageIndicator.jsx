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
    <div className="flex flex-wrap items-center gap-2 p-2.5 rounded-xl border border-hairline bg-surface-2 text-xs">
      {STAGES.map((s, i) => {
        const Icon = s.icon;
        const done = i < current || (streaming && s.key === "generate");
        const active = i === current && !done;

        return (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && <ArrowRight className="h-3 w-3 text-ink-4" />}
            <span
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-all ${
                active
                  ? "bg-accent text-accent border border-accent-line font-medium"
                  : done
                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    : "text-ink-4 bg-surface-2 border border-hairline"
              }`}
            >
              <Icon
                className={`h-3.5 w-3.5 ${
                  active ? "animate-pulse text-accent" : done ? "text-emerald-700" : "text-ink-4"
                }`}
              />
              <span>{s.label}</span>
              {active && <span className="h-1.5 w-1.5 rounded-full bg-accent animate-ping" />}
            </span>
          </div>
        );
      })}
      {note && (
        <span className="ml-auto text-[11px] font-mono text-ink-3 bg-surface-2 px-2 py-0.5 rounded-md border border-hairline">
          {note}
        </span>
      )}
    </div>
  );
}

