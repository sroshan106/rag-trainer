import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Card from "../components/Card.jsx";
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

// The view is a transcript: the in-flight exchange on top, then everything
// Postgres has, newest first. The answer no longer lives in its own card --
// rendering it twice (once fresh, once as a history row) was what made a
// finished query appear to jump.
export default function Ask() {
  const [query, setQuery] = useState("");
  const [live, setLive] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [collection, setCollection] = useState(null);

  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const loading = live !== null;

  useEffect(() => {
    queryModels()
      .then(({ models: available, default: def }) => {
        setModels(available);
        const saved = localStorage.getItem(MODEL_STORAGE_KEY);
        setModel(saved && available.includes(saved) ? saved : def);
      })
      .catch(() => {
        // No API yet -- the select stays empty and the server picks its default.
      });
    collectionStatus()
      .then(setCollection)
      .catch(() => {
        // Can't tell empty from stocked -- say nothing rather than guess.
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

  // Pending rows can come from another tab or the CLI -- this client only sees
  // its own query through the stream. Poll while any of them are unresolved so
  // they flip to done without the user having to act.
  const hasPending = history.some((entry) => entry.status === "pending");
  useEffect(() => {
    if (!hasPending) return undefined;
    const interval = setInterval(refresh, PENDING_POLL_MS);
    return () => clearInterval(interval);
  }, [hasPending, refresh]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // Abort on unmount too: navigating away should stop the generation, not
  // leave Ollama producing an answer for a view that no longer exists.
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
        sources: [],
        stage: "retrieve",
        stageDetail: null,
      });

      try {
        await streamQuery(trimmed, {
          model: chosenModel || null,
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === "stage") {
              setLive((cur) =>
                cur && { ...cur, stage: event.stage, stageDetail: event.detail ?? null },
              );
            } else if (event.type === "token") {
              setLive((cur) => cur && { ...cur, answer: cur.answer + event.text });
            } else if (event.type === "done") {
              setLive((cur) => cur && { ...cur, ...event, stage: null });
            } else if (event.type === "error") {
              setError(event.detail);
            }
          },
        });
      } catch (err) {
        // An abort is the Cancel button doing its job, not a failure -- the
        // server has already recorded the row as cancelled.
        if (err.name !== "AbortError") setError(err.message);
      } finally {
        abortRef.current = null;
        // Load the stored row before dropping the live one, so the finished
        // answer never blinks out between the two.
        await refresh();
        setLive(null);
      }
    },
    [refresh],
  );

  function onSubmit(e) {
    e.preventDefault();
    const text = query.trim();
    if (!text || loading) return;
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

  // Cmd/Ctrl+K focuses the box from anywhere; Escape cancels a running query.
  // Both are global rather than bound to the input, since the answer is what
  // the user is looking at while a query runs.
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

  // Up-arrow in an empty box recalls the last question, the way a shell does.
  function onInputKeyDown(event) {
    if (event.key !== "ArrowUp" || query) return;
    const last = history[0]?.query;
    if (!last) return;
    event.preventDefault();
    setQuery(last);
  }

  // The live row is also in Postgres as a pending row; hide that copy rather
  // than show the same question twice.
  const rows = live
    ? history.filter((entry) => !(entry.status === "pending" && entry.query === live.query))
    : history;

  return (
    <div className="flex flex-col gap-4">
      <Card title="Ask" subtitle="Query the ingested collection through the RAG pipeline.">
        <form onSubmit={onSubmit} className="flex gap-2">
          <input
            ref={inputRef}
            autoFocus
            className="flex-1 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
            placeholder="What would you like to know?    (⌘K)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKeyDown}
          />
          {models.length > 0 && (
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
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
          {loading ? (
            <button
              type="button"
              onClick={cancel}
              className="rounded-md border border-neutral-300 dark:border-neutral-700 px-4 py-2 text-sm font-medium"
            >
              Cancel
            </button>
          ) : (
            <button
              type="submit"
              disabled={!query.trim()}
              className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              Ask
            </button>
          )}
        </form>
      </Card>

      {collection?.empty && (
        <Card className="border-amber-300 dark:border-amber-800">
          <p className="text-sm">
            Nothing has been ingested yet, so every question will come back
            unanswered.{" "}
            <Link to="/ingest" className="font-medium underline">
              Upload a dataset
            </Link>{" "}
            first.
          </p>
        </Card>
      )}

      {error && (
        <Card title="Error" className="border-red-300 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </Card>
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
        <Card title="History unavailable" className="border-red-300 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{historyError}</p>
        </Card>
      )}

      {!historyError && rows.length === 0 && !live && (
        <Card>
          <p className="text-sm text-neutral-500">Nothing asked yet.</p>
        </Card>
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
        <div>
          <button
            onClick={onClear}
            className="rounded-md border border-neutral-300 dark:border-neutral-700 px-3 py-1.5 text-xs"
          >
            Clear history
          </button>
        </div>
      )}
    </div>
  );
}
