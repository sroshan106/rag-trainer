import { useCallback, useEffect, useState } from "react";
import Card from "../components/Card.jsx";
import SourceList from "../components/SourceList.jsx";
import { runQuery, queryModels, queryHistory, clearHistory } from "../api.js";

function formatWhen(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

const MODEL_STORAGE_KEY = "rag:ask:model";
const PENDING_POLL_MS = 2000;

export default function Ask() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");

  useEffect(() => {
    queryModels()
      .then(({ models: available, default: def }) => {
        setModels(available);
        const saved = localStorage.getItem(MODEL_STORAGE_KEY);
        setModel(saved && available.includes(saved) ? saved : def);
      })
      .catch(() => {
        // No API yet -- the select just stays empty and the server picks its default.
      });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const rows = await queryHistory(25);
      setHistory(rows);
      setHistoryError(null);
      return rows;
    } catch (err) {
      setHistoryError(err.message);
      return null;
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // The pending row is written to Postgres by the server the moment a query
  // starts (before the graph runs), so it shows up here for every tab/client
  // that reads history -- surviving a reload since it isn't client state at
  // all. Poll while anything is still pending so it flips to done/error
  // without the user having to act.
  const hasPending = history.some((entry) => entry.status === "pending");
  useEffect(() => {
    if (!hasPending) return undefined;
    const interval = setInterval(refresh, PENDING_POLL_MS);
    return () => clearInterval(interval);
  }, [hasPending, refresh]);

  async function onSubmit(e) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const submitted = runQuery(query.trim(), model || null);
      // The server inserts the pending row as its first step, before the
      // graph runs -- refresh shortly after firing so it shows up promptly
      // instead of waiting for the whole request to resolve.
      setTimeout(refresh, 200);
      const res = await submitted;
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      await refresh();
    }
  }

  async function onClear() {
    if (!window.confirm("Delete every saved question and answer?")) return;
    try {
      await clearHistory();
      refresh();
    } catch (err) {
      setHistoryError(err.message);
    }
  }

  const refused = result && result.sources.length === 0;

  return (
    <div className="flex flex-col gap-4">
      <Card title="Ask" subtitle="Query the ingested collection through the RAG pipeline.">
        <form onSubmit={onSubmit} className="flex gap-2">
          <input
            className="flex-1 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
            placeholder="What would you like to know?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          {models.length > 0 && (
            <select
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                localStorage.setItem(MODEL_STORAGE_KEY, e.target.value);
              }}
              disabled={loading}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm disabled:opacity-50 dark:[color-scheme:dark]"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          )}
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {loading ? "Asking..." : "Ask"}
          </button>
        </form>
      </Card>

      {error && (
        <Card title="Error" className="border-red-300 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </Card>
      )}

      {result && (
        <Card
          title="Answer"
          subtitle={refused ? "No sources were cited -- likely a refusal." : undefined}
        >
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{result.answer}</p>
          <SourceList sources={result.sources} />
        </Card>
      )}

      <Card
        title="Saved questions"
        subtitle="Every answered query is stored in Postgres with its citations."
      >
        {historyError && (
          <p className="text-sm text-red-600 dark:text-red-400">{historyError}</p>
        )}

        {!historyError && history.length === 0 && (
          <p className="text-sm text-neutral-500">Nothing asked yet.</p>
        )}

        {history.length > 0 && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-neutral-500">
                    <th className="pb-2 pr-4">Question</th>
                    <th className="pb-2 pr-4">Answer</th>
                    <th className="pb-2 pr-4">Citations</th>
                    <th className="pb-2 pr-4">Latency</th>
                    <th className="pb-2 pr-4">Model</th>
                    <th className="pb-2 pr-4">Asked</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((entry) => (
                    <tr
                      key={entry.id}
                      className={`border-t border-neutral-200 dark:border-neutral-800 align-top ${
                        entry.status === "pending" ? "animate-pulse" : ""
                      }`}
                    >
                      <td className="py-2 pr-4 font-medium max-w-xs">{entry.query}</td>
                      {entry.status === "pending" ? (
                        <td className="py-2 pr-4 text-neutral-500 whitespace-nowrap" colSpan={4}>
                          Asking...
                        </td>
                      ) : entry.status === "error" ? (
                        <td className="py-2 pr-4 text-red-600 dark:text-red-400 whitespace-nowrap" colSpan={4}>
                          Failed
                        </td>
                      ) : (
                        <>
                          <td className="py-2 pr-4 text-neutral-600 dark:text-neutral-300 whitespace-pre-wrap max-w-md">
                            {entry.answer}
                            {entry.refused && (
                              <span className="ml-2 text-[11px] text-neutral-500">(refused)</span>
                            )}
                          </td>
                          <td className="py-2 pr-4 min-w-[16rem]">
                            <SourceList sources={entry.sources} compact />
                          </td>
                          <td className="py-2 pr-4 text-xs text-neutral-500 whitespace-nowrap">
                            {entry.latency_ms != null ? `${Math.round(entry.latency_ms)} ms` : "—"}
                          </td>
                          <td className="py-2 pr-4 text-xs text-neutral-500 whitespace-nowrap">
                            {entry.model ?? "—"}
                          </td>
                        </>
                      )}
                      <td className="py-2 pr-4 text-xs text-neutral-500 whitespace-nowrap">
                        {formatWhen(entry.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button
              onClick={onClear}
              className="mt-4 rounded-md border border-neutral-300 dark:border-neutral-700 px-3 py-1.5 text-xs"
            >
              Clear history
            </button>
          </>
        )}
      </Card>
    </div>
  );
}
