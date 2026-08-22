import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Sparkles,
  Trash2,
  AlertCircle,
  AlertTriangle,
  Database,
} from "lucide-react";
import Exchange from "../components/Exchange.jsx";
import {
  streamQuery,
  queryModels,
  queryHistory,
  clearHistory,
  deleteHistoryEntry,
  collectionStatus,
} from "../api.js";

const MODEL_STORAGE_KEY = "rag:ask:model";
const PENDING_POLL_MS = 2000;

const SUGGESTIONS = [
  "Summarize the main findings in the dataset.",
  "What are the key concepts and topics discussed?",
  "What limitations or caveats are mentioned?",
  "Compare the primary methodologies in the documents.",
];

export default function Ask() {
  const [query, setQuery] = useState("");
  const [live, setLive] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [collection, setCollection] = useState(null);

  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const loading = live !== null;

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
    let ignore = false;
    queryModels().then(({ models: available }) => {
      if (ignore) return;
      setModels(available);
      setModelsLoaded(true);
      const saved = localStorage.getItem(MODEL_STORAGE_KEY);
      setModel(saved && available.includes(saved) ? saved : available[0] || "");
    }).catch(() => {
      if (!ignore) setModelsLoaded(true);
    });
    collectionStatus().then((s) => {
      if (!ignore) setCollection(s);
    }).catch(() => {});
    queryHistory(25).then((rows) => {
      if (!ignore) {
        setHistory(rows);
        setHistoryError(null);
      }
    }).catch((err) => {
      if (!ignore) setHistoryError(err.message);
    });
    return () => { ignore = true; };
  }, []);

  const hasPending = history.some((entry) => entry.status === "pending");
  useEffect(() => {
    if (!hasPending) return undefined;
    const interval = setInterval(refresh, PENDING_POLL_MS);
    return () => clearInterval(interval);
  }, [hasPending, refresh]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const ask = useCallback(
    async (text, chosenModel) => {
      const trimmed = text.trim();
      if (!trimmed || abortRef.current) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setError(null);
      setLive({
        query: trimmed,
        model: chosenModel || null,
        answer: "",
        citations: [],
        stage: "retrieve",
        stageDetail: null,
      });

      try {
        await streamQuery(trimmed, {
          model: chosenModel || null,
          signal: controller.signal,
          onEvent: (e) => {
            if (e.type === "stage") setLive(c => c && { ...c, stage: e.stage, stageDetail: e.detail ?? null });
            else if (e.type === "token") setLive(c => c && { ...c, answer: c.answer + e.text });
            else if (e.type === "done") setLive(c => c && { ...c, ...e, stage: null });
            else if (e.type === "error") setError(e.detail);
          },
        });
      } catch (err) {
        if (err.name !== "AbortError") setError(err.message);
      } finally {
        abortRef.current = null;
        await refresh();
        setLive(null);
      }
    },
    [refresh],
  );

  function onSubmit(e) {
    e?.preventDefault();
    const text = query.trim();
    if (!text || !model || loading) return;
    setQuery("");
    ask(text, model);
  }

  function onModelChange(next) {
    setModel(next);
    localStorage.setItem(MODEL_STORAGE_KEY, next);
  }

  function reuse(text) {
    setQuery(text);
    inputRef.current?.focus();
  }

  async function onDelete(entry) {
    try {
      await deleteHistoryEntry(entry.id);
      setHistory((rows) => rows.filter((row) => row.id !== entry.id));
    } catch (err) {
      setHistoryError(err.message);
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

  useEffect(() => {
    function onKeyDown(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      } else if (event.key === "Escape" && abortRef.current) {
        event.preventDefault();
        cancel();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cancel]);

  function onInputKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    } else if (event.key === "ArrowUp" && !query) {
      const last = history[0]?.query;
      if (last) {
        event.preventDefault();
        setQuery(last);
      }
    }
  }

  const rows = live
    ? history.filter((entry) => !(entry.status === "pending" && entry.query === live.query))
    : history;

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">

      <div>
        <h1 className="text-2xl font-bold tracking-tight text-ink">Ask Question</h1>
        <p className="text-sm text-ink-3 mt-1">Query your knowledge base through the RAG pipeline</p>
      </div>


      <div className="rounded-2xl border border-hairline bg-surface p-5 shadow-card">
        <form onSubmit={onSubmit} className="flex flex-col gap-3">
          <div className="relative">
            <textarea
              ref={inputRef}
              autoFocus
              rows={2}
              className="w-full resize-none rounded-xl border border-hairline bg-surface px-4 py-3 text-sm text-ink placeholder-ink-4 focus:outline-none focus:border-accent-line focus:ring-1 focus:ring-accent/40 transition-all font-normal"
              placeholder="What would you like to know from your documents? (⌘K)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onInputKeyDown}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-ink-3">Model</span>
              {models.length > 0 ? (
                <div className="relative">
                  <select
                    value={model}
                    onChange={(e) => onModelChange(e.target.value)}
                    disabled={loading}
                    className="appearance-none rounded-lg border border-hairline bg-surface text-ink px-3 py-1.5 pr-8 text-xs font-mono font-medium focus:outline-none focus:border-accent-line transition-colors cursor-pointer"
                  >
                    {models.map((m) => (
                      <option key={m} value={m} className="bg-surface text-ink">
                        {m}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-ink-3">
                    <span className="text-[10px]">▼</span>
                  </div>
                </div>
              ) : (
                <span className="text-xs text-amber-700">No model loaded</span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {loading ? (
                <button
                  type="button"
                  onClick={cancel}
                  className="px-4 py-2 rounded-lg border border-hairline bg-surface-2 hover:bg-surface-3 text-ink text-xs font-medium transition-colors"
                >
                  Cancel
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!query.trim() || !model}
                  className="flex items-center gap-2 rounded-lg bg-accent hover:bg-accent-strong text-white px-5 py-2 text-sm font-medium transition-all shadow-sm shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Ask</span>
                </button>
              )}
            </div>
          </div>
        </form>


        {rows.length === 0 && !live && (
          <div className="mt-4 pt-4 border-t border-hairline">
            <div className="text-xs font-medium text-ink-3 mb-2.5 flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-accent" />
              <span>Suggested questions:</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SUGGESTIONS.map((suggestion, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setQuery(suggestion);
                    inputRef.current?.focus();
                  }}
                  className="text-left text-xs p-2.5 rounded-xl border border-hairline bg-surface-2 hover:bg-surface-2 hover:border-hairline text-ink-2 transition-all leading-snug"
                >
                  "{suggestion}"
                </button>
              ))}
            </div>
          </div>
        )}
      </div>


      {modelsLoaded && models.length === 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p>
              No chat model is downloaded yet.{" "}
              <Link to="/settings" className="font-semibold underline hover:text-amber-700">
                Download one in Settings
              </Link>{" "}
              to start asking questions.
            </p>
          </div>
        </div>
      )}

      {collection?.empty && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700 flex items-start gap-3">
          <Database className="h-5 w-5 text-amber-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p>
              Knowledge base is empty.{" "}
              <Link to="/ingest" className="font-semibold underline hover:text-amber-700">
                Ingest a dataset
              </Link>{" "}
              first to get answers with grounded citations.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-700 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Error: </span>
            {error}
          </div>
        </div>
      )}


      {live && (
        <Exchange
          entry={live}
          live
          stage={live.stage}
          stageDetail={live.stageDetail}
          onCancel={cancel}
        />
      )}


      {historyError && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          History unavailable: {historyError}
        </div>
      )}


      {rows.map((entry) => (
        <Exchange
          key={entry.id}
          entry={entry}
          models={models}
          onRerun={(text, chosenModel) => ask(text, chosenModel)}
          onDelete={onDelete}
          onReuse={reuse}
        />
      ))}

      {rows.length > 0 && (
        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-hairline bg-surface-2 hover:bg-rose-50 hover:border-rose-200 text-ink-3 hover:text-rose-700 text-xs transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Clear all history</span>
          </button>
        </div>
      )}
    </div>
  );
}
