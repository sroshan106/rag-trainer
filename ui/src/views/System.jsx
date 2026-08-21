import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Cpu, Zap, Activity, Database, AlertTriangle } from "lucide-react";
import Card from "../components/Card.jsx";
import { metricsStreamUrl } from "../api.js";

const HISTORY_LENGTH = 120;

// Three hues that stay distinguishable on a white plot and hold 3:1 against it
// as 2px strokes. Legend, line, and tooltip all read from this one list.
const SERIES = [
  { key: "cpu", label: "CPU", color: "#5e5ce6" },
  { key: "vram", label: "VRAM", color: "#e5484d" },
  { key: "gpuUtil", label: "GPU Util", color: "#0d9488" },
];

function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-hairline bg-surface p-3 shadow-xl text-xs font-mono">
        {payload.map((p) => (
          <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
            <span className="text-ink-3 capitalize">{p.name}:</span>
            <span className="font-semibold text-ink">{p.value?.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

export default function SystemView() {
  const [frame, setFrame] = useState(null);
  const [history, setHistory] = useState([]);
  const [connected, setConnected] = useState(false);
  const tRef = useRef(0);

  useEffect(() => {
    const source = new EventSource(metricsStreamUrl());

    source.addEventListener("open", () => setConnected(true));
    source.addEventListener("error", () => setConnected(false));
    source.addEventListener("metrics", (event) => {
      const data = JSON.parse(event.data);
      setFrame(data);
      setConnected(true);
      tRef.current += 1;
      const gpu = data.gpu?.[0];
      setHistory((prev) => {
        const next = [
          ...prev,
          {
            t: tRef.current,
            cpu: data.cpu.percent,
            mem: data.memory.percent,
            vram: gpu ? gpu.memory_pct : null,
            gpuUtil: gpu ? gpu.utilization_pct : null,
          },
        ];
        return next.length > HISTORY_LENGTH ? next.slice(next.length - HISTORY_LENGTH) : next;
      });
    });

    return () => source.close();
  }, []);

  const gpu = frame?.gpu?.[0];

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-ink">System Monitor</h1>
          <p className="text-sm text-ink-3 mt-1">Real-time hardware telemetry and VRAM tracking</p>
        </div>

        <div
          role="status"
          className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-hairline bg-surface self-start sm:self-auto"
        >
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-emerald-500 motion-safe:animate-pulse" : "bg-ink-4"
            }`}
          />
          <span className="text-xs font-medium text-ink-2">
            {connected ? "Live • 1 sample/sec" : "Connecting to telemetry..."}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Gauge
          label="CPU Load"
          value={frame?.cpu.percent}
          unit="%"
          icon={Cpu}
          color="accent"
        />
        <Gauge
          label="RAM Usage"
          value={frame?.memory.percent}
          unit="%"
          icon={Database}
          color="teal"
          detail={frame ? `${(frame.memory.used_mb / 1024).toFixed(1)} / ${(frame.memory.total_mb / 1024).toFixed(1)} GB` : null}
        />
        <Gauge
          label="VRAM"
          value={gpu?.memory_pct}
          unit="%"
          icon={Zap}
          color="rose"
          warn={gpu && gpu.memory_pct > 85}
          detail={gpu ? `${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB` : "No GPU detected"}
        />
        <Gauge
          label="GPU Utilization"
          value={gpu?.utilization_pct}
          unit="%"
          icon={Activity}
          color="amber"
        />
      </div>

      {/* Real-time Chart */}
      <Card
        title="Hardware Metrics over Time"
        subtitle="Tracking CPU, GPU utilization, and VRAM memory usage continuously."
      >
        <div className="flex items-center gap-4 text-xs font-medium mb-3 text-ink-2">
          {SERIES.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              {s.label} %
            </span>
          ))}
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8ed" />
              <XAxis dataKey="t" tick={false} stroke="#c7c7cc" />
              <YAxis domain={[0, 100]} stroke="#c7c7cc" tick={{ fontSize: 11, fill: "#6e6e73" }} />
              <Tooltip content={<CustomTooltip />} />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  stroke={s.color}
                  strokeWidth={2}
                  dot={false}
                  name={s.label}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Memory & Storage Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="System Memory & Swap">
          {frame ? (
            <div className="flex flex-col gap-3 text-xs">
              <div>
                <div className="flex justify-between text-ink-3 mb-1">
                  <span>RAM Used</span>
                  <span className="font-semibold text-ink">
                    {frame.memory.used_mb.toFixed(0)} / {frame.memory.total_mb.toFixed(0)} MB ({frame.memory.percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden">
                  <div
                    className="h-full bg-accent rounded-full transition-all"
                    style={{ width: `${Math.min(frame.memory.percent, 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-ink-3 mb-1">
                  <span>Swap Space</span>
                  <span className="font-semibold text-ink">
                    {frame.memory.swap_used_mb.toFixed(0)} MB ({frame.memory.swap_percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden">
                  <div
                    className="h-full bg-teal-600 rounded-full transition-all"
                    style={{ width: `${Math.min(frame.memory.swap_percent, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-ink-4">Loading memory stats...</div>
          )}
        </Card>

        <Card title="Disk Storage">
          {frame ? (
            <div className="flex flex-col gap-3 text-xs">
              <div>
                <div className="flex justify-between text-ink-3 mb-1">
                  <span>Disk Capacity</span>
                  <span className="font-semibold text-ink">
                    {frame.disk.used_gb.toFixed(1)} / {frame.disk.total_gb.toFixed(1)} GB ({frame.disk.percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden">
                  <div
                    className="h-full bg-emerald-600 rounded-full transition-all"
                    style={{ width: `${Math.min(frame.disk.percent, 100)}%` }}
                  />
                </div>
              </div>
              <div className="pt-2 text-ink-3">
                <span>Free Available Storage: </span>
                <span className="font-semibold text-emerald-700">{frame.disk.free_gb.toFixed(1)} GB</span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-ink-4">Loading disk stats...</div>
          )}
        </Card>
      </div>
    </div>
  );
}

const GAUGE_BARS = {
  accent: "bg-accent",
  rose: "bg-rose-500",
  teal: "bg-teal-600",
  amber: "bg-amber-500",
};

function Gauge({ label, value, unit, icon: Icon, color = "accent", warn, detail }) {
  const display = value == null ? "--" : value.toFixed(0);
  const pct = typeof value === "number" ? Math.min(Math.max(value, 0), 100) : 0;
  const barColor = warn ? "bg-rose-500" : GAUGE_BARS[color] || "bg-accent";

  return (
    <div className="rounded-2xl border border-hairline/70 bg-surface p-4 shadow-card flex flex-col justify-between">
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-[13px] font-medium text-ink-3">{label}</span>
        {Icon && <Icon className="h-4 w-4 text-ink-4" aria-hidden="true" />}
      </div>
      <div>
        <div
          className={`flex items-center gap-1.5 text-2xl font-semibold font-mono tracking-[-0.02em] ${warn ? "text-rose-700" : "text-ink"}`}
        >
          {warn && <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />}
          <span>
            {display}
            {value != null && <span className="text-sm font-normal text-ink-3 ml-0.5">{unit}</span>}
          </span>
          {warn && <span className="sr-only">Warning: value is high</span>}
        </div>
        {value != null && (
          <div className="h-1.5 w-full rounded-full bg-surface-2 overflow-hidden mt-2">
            <div className={`h-full rounded-full transition-all duration-300 ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
        )}
      </div>
      {detail && <div className="text-[11px] font-mono text-ink-3 mt-2 truncate">{detail}</div>}
    </div>
  );
}

