"""Test cho karaokeforge.utils.gpu.detect_gpu (PR-A1). Xem plan §5.1 (G1-G6)."""

from __future__ import annotations

import sys

import pytest

from tests.helpers_a1 import install_fake_torch


def test_g1_no_torch_returns_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    from karaokeforge.utils.gpu import detect_gpu

    assert detect_gpu() == {"available": False, "name": None, "vram_gb": 0.0}


def test_g2_cuda_not_available_returns_unavailable(monkeypatch):
    install_fake_torch(monkeypatch, cuda=False)
    from karaokeforge.utils.gpu import detect_gpu

    result = detect_gpu()
    assert result == {"available": False, "name": None, "vram_gb": 0.0}


def test_g3_cuda_t4_returns_expected_dict(monkeypatch):
    install_fake_torch(monkeypatch, cuda=True, name="Tesla T4", vram_bytes=15_843_721_216)
    from karaokeforge.utils.gpu import detect_gpu

    result = detect_gpu()
    assert result["available"] is True
    assert result["name"] == "Tesla T4"
    assert result["vram_gb"] == pytest.approx(15.84, abs=0.01)


def test_g4_dict_has_exact_keys_and_types(monkeypatch):
    install_fake_torch(monkeypatch, cuda=True)
    from karaokeforge.utils.gpu import detect_gpu

    result = detect_gpu()
    assert set(result.keys()) == {"available", "name", "vram_gb"}
    assert isinstance(result["available"], bool)
    assert result["name"] is None or isinstance(result["name"], str)
    assert isinstance(result["vram_gb"], float)


def test_g5_get_device_properties_raises_does_not_propagate(monkeypatch):
    fakes = install_fake_torch(monkeypatch, cuda=True)
    fakes.get_device_properties.side_effect = RuntimeError("driver error")
    from karaokeforge.utils.gpu import detect_gpu

    result = detect_gpu()
    assert result == {"available": False, "name": None, "vram_gb": 0.0}


def test_g6_import_succeeds_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "karaokeforge.utils.gpu", raising=False)
    import karaokeforge.utils.gpu  # noqa: F401 — chỉ cần import không raise
