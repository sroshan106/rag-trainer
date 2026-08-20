const STYLES = {
  pending: "bg-neutral-200 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300",
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
  done: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  // Distinct from failed: a cancelled job stopped on request and its partial
  // result is still worth reading.
  cancelled: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
};

export default function StatusBadge({ status }) {
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${STYLES[status] ?? ""}`}>
      {status}
    </span>
  );
}
