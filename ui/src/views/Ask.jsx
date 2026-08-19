import { useState } from "react";
import Card from "../components/Card.jsx";
import { runQuery } from "../api.js";

export default function Ask() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runQuery(query.trim());
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
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
        <Card title="Answer" subtitle={refused ? "No sources were cited -- likely a refusal." : undefined}>
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{result.answer}</p>

          {result.sources.length > 0 && (
            <div className="mt-4 pt-3 border-t border-neutral-200 dark:border-neutral-800">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500 mb-2">
                Sources
              </h3>
              <ul className="text-xs space-y-1">
                {result.sources.map((src, i) => (
                  <li key={i}>
                    <a
                      href={src}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-600 dark:text-blue-400 hover:underline break-all"
                    >
                      {src}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
