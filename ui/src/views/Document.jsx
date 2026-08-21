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
          className="inline-flex items-center gap-1.5 text-xs text-ink-3 hover:text-ink transition-colors mb-3"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Knowledge base
        </Link>
        <h1 className="text-2xl font-bold tracking-tight text-ink">Document</h1>
        <p className="text-sm text-ink-3 mt-1">
          The text as it was ingested and embedded
        </p>
      </div>

      <div className="rounded-2xl border border-hairline bg-surface p-6 shadow-card">
        <DocumentView
          fileId={fileId}
          focusIndex={Number.isNaN(focusIndex) ? null : focusIndex}
        />
      </div>
    </div>
  );
}
