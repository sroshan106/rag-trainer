export default function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section
      className={`rounded-2xl border border-hairline/70 bg-surface p-5 shadow-card ${className}`}
    >
      {(title || subtitle || action) && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            {title && (
              <h2 className="text-[15px] font-semibold text-ink">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="text-[13px] text-ink-3 mt-1 leading-relaxed">{subtitle}</p>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

