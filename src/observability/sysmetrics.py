"""Host CPU/RAM/disk/GPU collectors for the System dashboard view.

Browsers expose no API for host resource utilization (checked -- see
ui_plan.md), so metrics are gathered server-side and streamed over SSE. GPU
telemetry uses NVML directly (via pynvml) rather than shelling out to
nvidia-smi per sample -- no subprocess per frame, structured values instead
of parsed CSV. VRAM headroom gets top billing deliberately: on a 4GB card
running llama3.2:3b alongside nomic-embed-text, VRAM exhaustion is what
causes Ollama to evict a model and generation latency to fall off a cliff,
not a gradual slowdown.

Degrades gracefully when NVML is unavailable (no NVIDIA driver, no GPU, or
pynvml not installed) -- the ``gpu`` field is simply omitted rather than the
whole collector failing.
"""

import shutil

import psutil

try:
    import pynvml

    _NVML_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without pynvml installed
    pynvml = None
    _NVML_IMPORT_ERROR = exc

_nvml_ready = False
_nvml_failed = False


def _ensure_nvml() -> bool:
    """Initialize NVML once. Returns False permanently if it can't be used here."""
    global _nvml_ready, _nvml_failed
    if _nvml_ready:
        return True
    if _nvml_failed or pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        _nvml_ready = True
        return True
    except Exception:
        _nvml_failed = True
        return False


def collect_gpu() -> list[dict] | None:
    """One dict per GPU device, or None if NVML is unavailable on this host."""
    if not _ensure_nvml():
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            device = {
                "index": i,
                "name": name,
                "utilization_pct": util.gpu,
                "memory_used_mb": round(mem.used / 1_048_576, 1),
                "memory_total_mb": round(mem.total / 1_048_576, 1),
                "memory_pct": round(100 * mem.used / mem.total, 1) if mem.total else 0.0,
            }
            try:
                device["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass
            try:
                device["power_watts"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:
                pass
            devices.append(device)
        return devices
    except Exception:
        # A transient NVML error (e.g. driver hiccup) should degrade this
        # frame, not take down the whole metrics stream.
        return None


def collect_cpu() -> dict:
    return {
        "percent": psutil.cpu_percent(interval=None),
        "per_core": psutil.cpu_percent(interval=None, percpu=True),
        "load_avg": _load_avg(),
    }


def _load_avg() -> list[float] | None:
    try:
        return list(psutil.getloadavg())
    except (AttributeError, OSError):  # pragma: no cover - not available on Windows
        return None


def collect_memory() -> dict:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_mb": round(vm.total / 1_048_576, 1),
        "used_mb": round(vm.used / 1_048_576, 1),
        "percent": vm.percent,
        "swap_used_mb": round(swap.used / 1_048_576, 1),
        "swap_percent": swap.percent,
    }


def collect_disk(path: str = "/") -> dict:
    usage = shutil.disk_usage(path)
    io = psutil.disk_io_counters()
    disk = {
        "total_gb": round(usage.total / 1_073_741_824, 1),
        "used_gb": round(usage.used / 1_073_741_824, 1),
        "free_gb": round(usage.free / 1_073_741_824, 1),
        "percent": round(100 * usage.used / usage.total, 1) if usage.total else 0.0,
    }
    if io is not None:
        disk["read_bytes"] = io.read_bytes
        disk["write_bytes"] = io.write_bytes
    return disk


def collect_all() -> dict:
    """One frame of the metrics stream -- everything the System view renders."""
    frame = {
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "disk": collect_disk(),
    }
    gpu = collect_gpu()
    if gpu is not None:
        frame["gpu"] = gpu
    return frame
