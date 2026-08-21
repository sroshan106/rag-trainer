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
    "flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150",
    isActive
      ? "bg-blue-600 text-white shadow-sm shadow-blue-500/20"
      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60",
  ].join(" ");
}

function ViewFallback() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-400 gap-3">
      <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
      <span className="text-xs uppercase tracking-wider font-semibold">Loading view...</span>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100 antialiased">
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-[#0b0f19]/85 backdrop-blur-md">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-4 py-2.5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 shadow-md shadow-blue-500/20 text-white">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <span className="font-bold tracking-tight text-slate-100 text-base">RAG Engine</span>
              <span className="hidden sm:inline-block ml-2 text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                v2.0
              </span>
            </div>
          </div>

          <nav className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800/80">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink key={item.to} to={item.to} className={linkClass}>
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
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
