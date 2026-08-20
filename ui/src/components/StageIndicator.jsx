// The three graph nodes, shown as they run. Retrieval and grading are fast;
// generation is the one that can take minutes on this hardware -- so the value
// here is less "progress" than "which slow thing am I waiting on", plus the
// counts each stage produced (20 fetched, 5 kept) as proof it is doing work.
const STAGES = [
  { key: "retrieve", label: "Retrieving" },
  { key: "grade", label: "Grading" },
  { key: "generate", label: "Generating" },
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
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {STAGES.map((s, i) => {
        const done = i < current || (streaming && s.key === "generate");
        const active = i === current && !done;
        return (
          <span key={s.key} className="flex items-center gap-2">
            {i > 0 && <span className="text-neutral-300 dark:text-neutral-700">·</span>}
            <span
              className={
                done
                  ? "text-neutral-500"
                  : active
                    ? "font-medium text-neutral-900 dark:text-neutral-100"
                    : "text-neutral-400 dark:text-neutral-600"
              }
            >
              {active && (
                <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500 align-middle" />
              )}
              {s.label}
            </span>
          </span>
        );
      })}
      {note && <span className="text-neutral-500">— {note}</span>}
    </div>
  );
}
