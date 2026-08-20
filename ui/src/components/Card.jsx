export default function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section
      className={`rounded-xl border border-slate-800/90 bg-[#111726]/85 backdrop-blur-md p-5 shadow-sm transition-all duration-200 ${className}`}
    >
      {(title || subtitle || action) && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            {title && (
              <h2 className="text-base font-semibold tracking-tight text-slate-100">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{subtitle}</p>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

