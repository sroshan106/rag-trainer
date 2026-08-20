import { ArrowLeft } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import DocumentView from "../components/DocumentView.jsx";

export default function DocumentPage() {
  const { fileId } = useParams();
  const [params] = useSearchParams();

  const unit = params.get("unit");
  const focusIndex = unit === null || unit === "" ? null : Number(unit);

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      <div>
        <Link
          to="/ingest"
          className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors mb-3"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Knowledge base
        </Link>
        <h1 className="text-2xl font-bold tracking-tight text-slate-100">Document</h1>
        <p className="text-sm text-slate-400 mt-1">
          The text as it was ingested and embedded
        </p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-6 shadow-sm">
        <DocumentView
          fileId={fileId}
          focusIndex={Number.isNaN(focusIndex) ? null : focusIndex}
        />
      </div>
    </div>
  );
}
