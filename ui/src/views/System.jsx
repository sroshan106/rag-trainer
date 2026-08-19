import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import Card from "../components/Card.jsx";
import { metricsStreamUrl } from "../api.js";

// How many samples of history to keep for the chart -- 120 frames at 1Hz is
// two minutes, enough to see a VRAM eviction cliff without the chart
// scrolling too fast to read.
const HISTORY_LENGTH = 120;

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
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${connected ? "bg-green-500" : "bg-neutral-400"}`}
        />
        <span className="text-xs text-neutral-500">
          {connected ? "live -- 1 frame/sec" : "connecting..."}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Gauge label="CPU" value={frame?.cpu.percent} unit="%" />
        <Gauge label="Memory" value={frame?.memory.percent} unit="%" />
        <Gauge
          label="VRAM"
          value={gpu?.memory_pct}
          unit="%"
          warn={gpu && gpu.memory_pct > 85}
          detail={gpu ? `${gpu.memory_used_mb.toFixed(0)} / ${gpu.memory_total_mb.toFixed(0)} MB` : "no GPU detected"}
        />
        <Gauge label="GPU utilization" value={gpu?.utilization_pct} unit="%" />
      </div>

      <Card title="CPU / VRAM over time" subtitle="Correlating a latency spike with a VRAM event is the diagnostic that matters most on this hardware.">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-neutral-200 dark:stroke-neutral-800" />
              <XAxis dataKey="t" tick={false} label={{ value: "time →", position: "insideBottom", offset: -2, fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="cpu" stroke="#3b82f6" dot={false} name="CPU %" isAnimationActive={false} />
              <Line type="monotone" dataKey="vram" stroke="#ef4444" dot={false} name="VRAM %" isAnimationActive={false} />
              <Line type="monotone" dataKey="gpuUtil" stroke="#a855f7" dot={false} name="GPU util %" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Memory">
          {frame && (
            <dl className="text-sm grid grid-cols-2 gap-y-1">
              <dt className="text-neutral-500">Used</dt>
              <dd>{frame.memory.used_mb.toFixed(0)} / {frame.memory.total_mb.toFixed(0)} MB</dd>
              <dt className="text-neutral-500">Swap used</dt>
              <dd>{frame.memory.swap_used_mb.toFixed(0)} MB ({frame.memory.swap_percent}%)</dd>
            </dl>
          )}
        </Card>
        <Card title="Disk">
          {frame && (
            <dl className="text-sm grid grid-cols-2 gap-y-1">
              <dt className="text-neutral-500">Free</dt>
              <dd>{frame.disk.free_gb.toFixed(1)} GB</dd>
              <dt className="text-neutral-500">Used</dt>
              <dd>{frame.disk.used_gb.toFixed(1)} / {frame.disk.total_gb.toFixed(1)} GB ({frame.disk.percent}%)</dd>
            </dl>
          )}
        </Card>
      </div>
    </div>
  );
}

function Gauge({ label, value, unit, warn, detail }) {
  const display = value === undefined || value === null ? "--" : value.toFixed(0);
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
      <div className="text-xs text-neutral-500 mb-1">{label}</div>
      <div className={`text-2xl font-semibold ${warn ? "text-red-600 dark:text-red-400" : ""}`}>
        {display}
        {value !== undefined && value !== null && <span className="text-sm font-normal">{unit}</span>}
      </div>
      {detail && <div className="text-xs text-neutral-500 mt-1">{detail}</div>}
    </div>
  );
}
