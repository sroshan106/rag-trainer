import { useEffect } from "react";
import { X, Maximize2 } from "lucide-react";
import { Link } from "react-router-dom";
import DocumentView from "./DocumentView.jsx";

/**
 * Peek at one cited unit without leaving the answer.
 *
 * Opens compact -- only the cited unit, not the whole document -- since the
 * question being answered is "what did that citation actually say".
 */
export default function DocumentModal({ fileId, focusIndex = null, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!fileId) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-2xl border border-slate-800 bg-[#111726] p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-end gap-1 mb-2">
          <Link
            to={`/documents/${fileId}${focusIndex !== null ? `?unit=${focusIndex}` : ""}`}
            onClick={onClose}
            title="Open full document"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <Maximize2 className="h-4 w-4" />
          </Link>
          <button
            type="button"
            onClick={onClose}
            title="Close"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <DocumentView fileId={fileId} focusIndex={focusIndex} compact />
      </div>
    </div>
  );
}
