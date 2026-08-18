"""Test cho karaokeforge.pipeline.transcriber — mock whisperx/torch qua sys.modules.

Bảng REQ: xem docs/plans/PR-A2A3.md §5.4 (T-01..T-20).
Fixture mock đặt trong chính file này (không tạo conftest.py — G1 DECISIONS.md).
"""
from __future__ import annotations

import logging
import sys
import types

import pytest

from tests.helpers_a2a3 import assert_valid_lyrics


SAMPLE_RATE = 16000
AUDIO_SECONDS = 10


class FakeModel:
    """Giả lập whisperx model trả về từ load_model()."""

    def __init__(self, segments=None, language="vi", calls=None):
        self._segments = segments if segments is not None else [
            {"start": 0.0, "end": 1.0, "text": "nguoi yeu"},
            {"start": 1.0, "end": 2.0, "text": "toi"},
        ]
        self._language = language
        self.calls = calls if calls is not None else []

    def transcribe(self, audio, **kwargs):
        self.calls.append(("transcribe", audio, kwargs))
        return {"segments": self._segments, "language": self._language}


def _default_align_segments(segments, *_a, **_k):
    out = []
    for seg in segments:
        text = seg.get("text", "")
        tokens = text.split() or ["x"]
        n = len(tokens)
        step = (seg["end"] - seg["start"]) / n if n else 0.0
        words = []
        for i, tok in enumerate(tokens):
            words.append({
                "word": tok,
                "start": seg["start"] + i * step,
                "end": seg["start"] + (i + 1) * step,
                "score": 0.9,
            })
        out.append({"start": seg["start"], "end": seg["end"], "text": text, "words": words})
    return {"segments": out}


@pytest.fixture
def call_log():
    return {}


@pytest.fixture
def fake_whisperx(monkeypatch, call_log):
    wx = types.ModuleType("whisperx")

    def load_audio(path):
        call_log["load_audio"] = path
        return [0.0] * (SAMPLE_RATE * AUDIO_SECONDS)

    def load_model(*args, **kwargs):
        call_log["load_model_args"] = args
        call_log["load_model_kwargs"] = kwargs
        model = FakeModel()
        call_log["model"] = model
        return model

    def load_align_model(**kwargs):
        call_log["load_align_model_kwargs"] = kwargs
        return object(), {}

    def align(segments, model, metadata, audio, device, **kwargs):
        call_log["align_segments"] = segments
        call_log["align_device"] = device
        align_fn = call_log.get("align_impl", _default_align_segments)
        return align_fn(segments)

    wx.load_audio = load_audio
    wx.load_model = load_model
    wx.load_align_model = load_align_model
    wx.align = align

    torch = types.ModuleType("torch")
    empty_cache_calls = []

    def empty_cache():
        empty_cache_calls.append(True)

    # detect_gpu() dùng torch.cuda.is_available/get_device_properties. Mặc định
    # có GPU (Colab cấp T4); test có thể lật call_log["gpu_available"]=False.
    call_log.setdefault("gpu_available", True)
    _props = types.SimpleNamespace(name="Tesla T4", total_memory=15_843_721_216)

    torch.cuda = types.SimpleNamespace(
        empty_cache=empty_cache,
        is_available=lambda: call_log["gpu_available"],
        get_device_properties=lambda idx=0: _props,
    )
    call_log["empty_cache_calls"] = empty_cache_calls
    call_log["torch"] = torch

    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return wx


def test_t01_import_safe_without_whisperx_or_torch():
    for name in ("whisperx", "torch", "karaokeforge.pipeline.transcriber"):
        sys.modules.pop(name, None)
    import karaokeforge.pipeline.transcriber  # noqa: F401

    assert "whisperx" not in sys.modules
    assert "torch" not in sys.modules


def test_t02_happy_path_schema(fake_whisperx):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert isinstance(out, list)
    for seg in out:
        assert set(seg.keys()) == {"start", "end", "text", "words"}
        for w in seg["words"]:
            assert set(w.keys()) == {"start", "end", "word", "confidence"}
    assert_valid_lyrics(out)


def test_t03_load_model_called_correctly(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi", whisper_model="large-v3")
    args = call_log["load_model_args"]
    kwargs = call_log["load_model_kwargs"]
    assert args[0] == "large-v3"
    assert args[1] == "cuda"
    assert kwargs["compute_type"] == "float16"
    assert kwargs["language"] == "vi"


def test_t04_transcribe_called_with_batch_size(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    model = call_log["model"]
    _, _, kwargs = model.calls[0]
    assert kwargs["batch_size"] == 16


def test_t05_load_align_model_called_correctly(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    kwargs = call_log["load_align_model_kwargs"]
    assert kwargs["language_code"] == "vi"
    assert kwargs["device"] == "cuda"


def test_t06_score_mapped_to_confidence_no_extra_keys(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def align_impl(segments):
        out = []
        for seg in segments:
            out.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": [
                    {"word": "nguoi", "start": seg["start"], "end": seg["end"],
                     "score": 0.87, "speaker": "SPEAKER_00"},
                ],
            })
        return {"segments": out}

    call_log["align_impl"] = align_impl
    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="en")
    word = out[0]["words"][0]
    assert word["confidence"] == 0.87
    assert "score" not in word
    assert "speaker" not in word


def test_t07_custom_whisper_model_passed_through(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", whisper_model="medium")
    assert call_log["load_model_args"][0] == "medium"


def test_t08_user_lyrics_text_reaches_align(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align(
        "audio.wav", language="vi", user_lyrics="dong1\ndong2",
    )
    align_segments = call_log["align_segments"]
    assert [s["text"] for s in align_segments] == ["dong1", "dong2"]


def test_t09_vietnamese_fix_applied_when_language_vi(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def align_impl(segments):
        return {"segments": [{
            "start": 0.0, "end": 1.0, "text": "nguoi",
            "words": [{"word": "nguoi", "start": 0.0, "end": 1.0, "score": 0.9}],
        }]}

    call_log["align_impl"] = align_impl
    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert out[0]["text"] == "người"
    assert out[0]["words"][0]["word"] == "người"


def test_t10_no_vietnamese_fix_when_language_not_vi(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def align_impl(segments):
        return {"segments": [{
            "start": 0.0, "end": 1.0, "text": "nguoi",
            "words": [{"word": "nguoi", "start": 0.0, "end": 1.0, "score": 0.9}],
        }]}

    call_log["align_impl"] = align_impl
    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="en")
    assert out[0]["text"] == "nguoi"
    assert out[0]["words"][0]["word"] == "nguoi"


def test_t11_empty_cache_called_happy_path(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert len(call_log["empty_cache_calls"]) >= 1


def test_t12_align_exception_falls_back_and_does_not_raise(fake_whisperx, call_log, monkeypatch):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def broken_align(*a, **k):
        raise RuntimeError("align boom")

    monkeypatch.setattr(sys.modules["whisperx"], "align", broken_align)

    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert isinstance(out, list)
    assert_valid_lyrics(out)
    assert len(call_log["empty_cache_calls"]) >= 1


def test_t13_load_model_exception_propagates_after_cleanup(fake_whisperx, call_log, monkeypatch):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def broken_load_model(*a, **k):
        raise RuntimeError("load boom")

    monkeypatch.setattr(sys.modules["whisperx"], "load_model", broken_load_model)

    with pytest.raises(RuntimeError, match="load boom"):
        LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert len(call_log["empty_cache_calls"]) >= 1


def test_t14_missing_word_timings_interpolated(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def align_impl(segments):
        return {"segments": [{
            "start": 0.0, "end": 2.0, "text": "a b",
            "words": [
                {"word": "a", "start": 0.0, "end": 1.0, "score": 0.9},
                {"word": "b", "score": None},
            ],
        }]}

    call_log["align_impl"] = align_impl
    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="en")
    for w in out[0]["words"]:
        assert isinstance(w["start"], float)
        assert isinstance(w["end"], float)


def test_t15_no_speech_no_lyrics_raises_exact_message(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    model = FakeModel(segments=[])
    sys.modules["whisperx"].load_model = lambda *a, **k: model

    with pytest.raises(RuntimeError, match=r"^Không nhận dạng được giọng hát\. Hãy paste lời\.$"):
        LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")


def test_t16_no_speech_with_user_lyrics_synthesizes_segments(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    model = FakeModel(segments=[])
    sys.modules["whisperx"].load_model = lambda *a, **k: model

    out = LyricsTranscriber().transcribe_and_align(
        "audio.wav", language="vi", user_lyrics="dong1\ndong2\ndong3\ndong4",
    )
    align_segments = call_log["align_segments"]
    assert len(align_segments) == 4
    duration = (SAMPLE_RATE * AUDIO_SECONDS) / SAMPLE_RATE
    assert align_segments[0]["start"] == 0.0
    assert align_segments[-1]["end"] == pytest.approx(duration, abs=1e-6)
    assert isinstance(out, list)


def test_t17_whitespace_only_segments_treated_as_no_speech(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    model = FakeModel(segments=[{"start": 0.0, "end": 1.0, "text": "   "}])
    sys.modules["whisperx"].load_model = lambda *a, **k: model

    with pytest.raises(RuntimeError):
        LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")


def test_t18_language_mismatch_warns_and_continues(fake_whisperx, call_log, caplog):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    model = FakeModel(language="en")
    sys.modules["whisperx"].load_model = lambda *a, **k: model

    with caplog.at_level(logging.WARNING):
        out = LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")

    assert "en" in caplog.text
    assert "vi" in caplog.text
    assert isinstance(out, list)
    assert len(out) > 0


def test_t19_output_rounded_to_3_decimals(fake_whisperx, call_log):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    def align_impl(segments):
        return {"segments": [{
            "start": 0.1234567, "end": 1.9999999, "text": "a",
            "words": [{"word": "a", "start": 0.1234567, "end": 1.9999999, "score": 0.9}],
        }]}

    call_log["align_impl"] = align_impl
    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="en")
    assert out[0]["start"] == round(0.1234567, 3)
    assert out[0]["end"] == round(1.9999999, 3)


def test_t20_output_is_list_not_raw_result(fake_whisperx):
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    out = LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")
    assert isinstance(out, list)
    assert not isinstance(out, dict)


# ---------------------------------------------------------------------------
# PR-PIPE-FIX Fix 4 — device/compute_type từ Config + detect_gpu (hết hardcode)
# ---------------------------------------------------------------------------


def test_t21_no_gpu_uses_cpu_and_int8(fake_whisperx, call_log):
    """Không có GPU (Colab không cấp) → device="cpu", compute_type CPU-safe ("int8")
    thay vì "cuda"/float16 hard-code làm whisperx chết khó hiểu."""
    call_log["gpu_available"] = False
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")

    assert call_log["load_model_args"][1] == "cpu"
    assert call_log["load_model_kwargs"]["compute_type"] == "int8"
    assert call_log["load_align_model_kwargs"]["device"] == "cpu"
    assert call_log["align_device"] == "cpu"


def test_t22_compute_type_read_from_config_when_gpu(fake_whisperx, call_log, monkeypatch):
    """Có GPU → device="cuda", compute_type đọc từ Config.WHISPER_COMPUTE_TYPE
    (auto_select_models hạ int8 phải có hiệu lực)."""
    from karaokeforge.config import Config
    from karaokeforge.pipeline.transcriber import LyricsTranscriber

    monkeypatch.setattr(Config, "WHISPER_COMPUTE_TYPE", "int8_float16")

    LyricsTranscriber().transcribe_and_align("audio.wav", language="vi")

    assert call_log["load_model_args"][1] == "cuda"
    assert call_log["load_model_kwargs"]["compute_type"] == "int8_float16"
