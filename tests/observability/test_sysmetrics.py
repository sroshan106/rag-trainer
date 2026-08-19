from src.observability import sysmetrics


def test_collect_all_has_required_keys():
    frame = sysmetrics.collect_all()

    assert "cpu" in frame
    assert "memory" in frame
    assert "disk" in frame
    assert 0 <= frame["memory"]["percent"] <= 100


def test_collect_gpu_returns_none_or_list():
    result = sysmetrics.collect_gpu()

    assert result is None or isinstance(result, list)


def test_collect_gpu_degrades_gracefully_when_nvml_missing(monkeypatch):
    monkeypatch.setattr(sysmetrics, "_ensure_nvml", lambda: False)

    assert sysmetrics.collect_gpu() is None
