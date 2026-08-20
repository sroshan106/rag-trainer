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
import time

import psutil

from src.observability.logging import log

try:
    import pynvml

    _NVML_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without pynvml installed
    pynvml = None
    _NVML_IMPORT_ERROR = exc

# Every failure NVML itself reports is an ``NVMLError`` subclass. Bound to a
# name here so the handlers below stay valid even when pynvml is absent (where
# no NVML call can raise in the first place).
_NVMLError: type[BaseException] = getattr(pynvml, "NVMLError", Exception)

_NVML_RETRY_COOLDOWN_SECONDS = 30.0

_nvml_ready = False
_nvml_last_fail_time: float | None = None


def _ensure_nvml() -> bool:
    """Initialize NVML, retrying after a cooldown if a prior attempt failed.

    A failed nvmlInit() at process startup (driver not yet loaded, GPU
    momentarily busy, container device not mounted yet, ...) is often
    transient -- latching it permanently would show "no GPU detected" for
    the rest of the process lifetime even once the GPU becomes reachable.
    """
    global _nvml_ready, _nvml_last_fail_time
    if _nvml_ready:
        return True
    if pynvml is None:
        return False
    if _nvml_last_fail_time is not None:
        if time.monotonic() - _nvml_last_fail_time < _NVML_RETRY_COOLDOWN_SECONDS:
            return False
    try:
        pynvml.nvmlInit()
        _nvml_ready = True
        _nvml_last_fail_time = None
        return True
    except _NVMLError:
        # The expected outcome on a host with no NVIDIA driver or no GPU --
        # stays quiet, since that is a configuration, not a fault.
        _nvml_last_fail_time = time.monotonic()
        return False
    except Exception as exc:  # noqa: BLE001 - anything else is a real surprise
        log("error", "NVML init raised unexpectedly", error=repr(exc))
        _nvml_last_fail_time = time.monotonic()
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
            # Temperature and power are genuinely optional readings -- plenty
            # of cards (and most virtualized ones) report NVML_ERROR_NOT_
            # SUPPORTED for them -- so a missing field here is not a fault.
            try:
                device["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except _NVMLError:
                pass
            try:
                device["power_watts"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except _NVMLError:
                pass
            devices.append(device)
        return devices
    except _NVMLError as exc:
        # NVML initialized fine, so this is a real fault (driver hiccup, device
        # fell off the bus), not an absent GPU -- degrade this frame rather
        # than take down the metrics stream, but say why: silently returning
        # None here is what made a broken GPU look like no GPU at all.
        log("warning", "GPU metrics unavailable: NVML query failed", error=str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 - unexpected, but must not kill the stream
        log("error", "GPU metrics collector raised unexpectedly", error=repr(exc))
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
