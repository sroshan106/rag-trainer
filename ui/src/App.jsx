import { lazy, Suspense } from "react";
import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { MessageSquare, UploadCloud, Activity, BarChart3, Settings as SettingsIcon, Sparkles, Loader2 } from "lucide-react";

const Ask = lazy(() => import("./views/Ask.jsx"));
const Ingest = lazy(() => import("./views/Ingest.jsx"));
const SystemView = lazy(() => import("./views/System.jsx"));
const Benchmark = lazy(() => import("./views/Benchmark.jsx"));
const Settings = lazy(() => import("./views/Settings.jsx"));
const DocumentPage = lazy(() => import("./views/Document.jsx"));

const NAV = [
  { to: "/ask", label: "Ask", icon: MessageSquare },
  { to: "/ingest", label: "Ingest", icon: UploadCloud },
  { to: "/system", label: "System", icon: Activity },
  { to: "/benchmark", label: "Benchmark", icon: BarChart3 },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

function linkClass({ isActive }) {
  return [
    "flex items-center gap-2 px-2.5 sm:px-3.5 py-2 sm:py-1.5 rounded-[10px] text-sm font-medium transition-colors duration-150 cursor-pointer",
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
    isActive
      ? "bg-accent text-white shadow-sm shadow-accent/25"
      : "text-ink-3 hover:text-ink hover:bg-surface",
  ].join(" ");
}

function ViewFallback() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] text-ink-3 gap-3">
      <Loader2 className="h-6 w-6 animate-spin text-accent" />
      <span className="text-xs uppercase tracking-wider font-semibold">Loading view...</span>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-canvas text-ink antialiased">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:rounded-lg focus:bg-accent focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-white"
      >
        Skip to content
      </a>
      {/*
        Floating chrome, not an opaque strip: content scrolls underneath a
        translucent material. The bottom edge is a hairline at 60% rather than
        a full divider so the bar reads as a layer above the page, not a seam.
      */}
      <header className="chrome-blur sticky top-0 z-50 border-b border-hairline/60 bg-canvas/72 backdrop-blur-xl backdrop-saturate-150">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-4 py-2.5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-accent to-cyan-400 shadow-md shadow-accent/20 text-white">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <span className="font-semibold tracking-[-0.015em] text-ink text-base">RAG Engine</span>
              <span className="hidden sm:inline-block ml-2 text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-surface-3/70 text-ink-3">
                v1.0
              </span>
            </div>
          </div>

          <nav
            aria-label="Primary"
            className="flex items-center gap-1 bg-surface-3/70 p-1 rounded-xl"
          >
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink key={item.to} to={item.to} className={linkClass} title={item.label} aria-label={item.label}>
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className="hidden sm:inline">{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>

      <main id="main-content" className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        <Suspense fallback={<ViewFallback />}>
          <Routes>
            <Route path="/" element={<Navigate to="/ask" replace />} />
            <Route path="/ask" element={<Ask />} />
            <Route path="/ingest" element={<Ingest />} />
            {/* Not in NAV: reached from a citation or a knowledge-base entry. */}
            <Route path="/documents/:fileId" element={<DocumentPage />} />
            <Route path="/system" element={<SystemView />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/ask" replace />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
