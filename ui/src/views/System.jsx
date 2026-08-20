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
import { Cpu, Zap, Activity, Database } from "lucide-react";
import Card from "../components/Card.jsx";
import { metricsStreamUrl } from "../api.js";

const HISTORY_LENGTH = 120;

function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-slate-700/80 bg-[#151c2d] p-3 shadow-xl text-xs font-mono">
        {payload.map((p) => (
          <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
            <span className="text-slate-400 capitalize">{p.name}:</span>
            <span className="font-semibold text-slate-100">{p.value?.toFixed(1)}%</span>
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
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100">System Monitor</h1>
          <p className="text-sm text-slate-400 mt-1">Real-time hardware telemetry and VRAM tracking</p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-slate-800 bg-[#111726]/80 self-start sm:self-auto">
          <span
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-emerald-400 animate-pulse" : "bg-slate-500"
            }`}
          />
          <span className="text-xs font-medium text-slate-300">
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
          color="blue"
        />
        <Gauge
          label="RAM Usage"
          value={frame?.memory.percent}
          unit="%"
          icon={Database}
          color="indigo"
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
          color="purple"
        />
      </div>

      {/* Real-time Chart */}
      <Card
        title="Hardware Metrics over Time"
        subtitle="Tracking CPU, GPU utilization, and VRAM memory usage continuously."
      >
        <div className="flex items-center gap-4 text-xs font-medium mb-3">
          <span className="flex items-center gap-1.5 text-blue-400">
            <span className="h-2 w-2 rounded-full bg-blue-500" /> CPU %
          </span>
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="h-2 w-2 rounded-full bg-rose-500" /> VRAM %
          </span>
          <span className="flex items-center gap-1.5 text-purple-400">
            <span className="h-2 w-2 rounded-full bg-purple-500" /> GPU Util %
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="t" tick={false} stroke="#475569" />
              <YAxis domain={[0, 100]} stroke="#475569" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="cpu"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="CPU"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="vram"
                stroke="#f43f5e"
                strokeWidth={2}
                dot={false}
                name="VRAM"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="gpuUtil"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
                name="GPU Util"
                isAnimationActive={false}
              />
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
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>RAM Used</span>
                  <span className="font-semibold text-slate-200">
                    {frame.memory.used_mb.toFixed(0)} / {frame.memory.total_mb.toFixed(0)} MB ({frame.memory.percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all"
                    style={{ width: `${Math.min(frame.memory.percent, 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Swap Space</span>
                  <span className="font-semibold text-slate-200">
                    {frame.memory.swap_used_mb.toFixed(0)} MB ({frame.memory.swap_percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-purple-500 rounded-full transition-all"
                    style={{ width: `${Math.min(frame.memory.swap_percent, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-500">Loading memory stats...</div>
          )}
        </Card>

        <Card title="Disk Storage">
          {frame ? (
            <div className="flex flex-col gap-3 text-xs">
              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Disk Capacity</span>
                  <span className="font-semibold text-slate-200">
                    {frame.disk.used_gb.toFixed(1)} / {frame.disk.total_gb.toFixed(1)} GB ({frame.disk.percent}%)
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all"
                    style={{ width: `${Math.min(frame.disk.percent, 100)}%` }}
                  />
                </div>
              </div>
              <div className="pt-2 text-slate-400">
                <span>Free Available Storage: </span>
                <span className="font-semibold text-emerald-400">{frame.disk.free_gb.toFixed(1)} GB</span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-500">Loading disk stats...</div>
          )}
        </Card>
      </div>
    </div>
  );
}

function Gauge({ label, value, unit, icon: Icon, color = "blue", warn, detail }) {
  const display = value == null ? "--" : value.toFixed(0);
  const pct = typeof value === "number" ? Math.min(Math.max(value, 0), 100) : 0;
  const barColor = warn ? "bg-rose-500" : { rose: "bg-rose-500", indigo: "bg-indigo-500", purple: "bg-purple-500" }[color] || "bg-blue-500";

  return (
    <div className="rounded-2xl border border-slate-800 bg-[#111726]/90 backdrop-blur-md p-4 shadow-sm flex flex-col justify-between">
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-medium text-slate-400">{label}</span>
        {Icon && <Icon className="h-4 w-4 text-slate-500" />}
      </div>
      <div>
        <div className={`text-2xl font-bold font-mono tracking-tight ${warn ? "text-rose-400" : "text-slate-100"}`}>
          {display}{value != null && <span className="text-sm font-normal text-slate-400 ml-0.5">{unit}</span>}
        </div>
        {value != null && (
          <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden mt-2">
            <div className={`h-full rounded-full transition-all duration-300 ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
        )}
      </div>
      {detail && <div className="text-[11px] font-mono text-slate-400 mt-2 truncate">{detail}</div>}
    </div>
  );
}

