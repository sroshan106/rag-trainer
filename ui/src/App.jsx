import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import Ask from "./views/Ask.jsx";
import Ingest from "./views/Ingest.jsx";
import SystemView from "./views/System.jsx";
import Benchmark from "./views/Benchmark.jsx";

const NAV = [
  { to: "/ask", label: "Ask" },
  { to: "/ingest", label: "Ingest" },
  { to: "/system", label: "System" },
  { to: "/benchmark", label: "Benchmark" },
];

function linkClass({ isActive }) {
  return [
    "px-3 py-2 rounded-md text-sm font-medium transition-colors",
    isActive
      ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
      : "text-neutral-600 hover:bg-neutral-200 dark:text-neutral-400 dark:hover:bg-neutral-800",
  ].join(" ");
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-neutral-200 dark:border-neutral-800">
        <div className="max-w-6xl mx-auto flex items-center gap-6 px-4 py-3">
          <span className="font-semibold tracking-tight">RAG Dashboard</span>
          <nav className="flex gap-1">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} className={linkClass}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/ask" replace />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="/system" element={<SystemView />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="*" element={<Navigate to="/ask" replace />} />
        </Routes>
      </main>
    </div>
  );
}
