"""Test cho karaokeforge.utils.audio (PR-A1). Xem plan §5.2 (A1-A17)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from tests.helpers_a1 import make_wav


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------


def test_a1_wave_fallback_mono_44100(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    path = make_wav(str(tmp_path / "a1.wav"), seconds=1.5, sample_rate=44100, channels=1)

    duration = audio.get_audio_duration(path)
    assert duration == pytest.approx(1.5, abs=0.02)


def test_a2_wave_fallback_stereo_22050(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    path = make_wav(str(tmp_path / "a2.wav"), seconds=0.5, sample_rate=22050, channels=2)

    duration = audio.get_audio_duration(path)
    assert duration == pytest.approx(0.5, abs=0.02)


def test_a3_ffprobe_available_used(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)
    called = {}

    def fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="12.345\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = make_wav(str(tmp_path / "a3.wav"), seconds=1.0)

    duration = audio.get_audio_duration(path)
    assert duration == 12.345
    assert "cmd" in called


def test_a4_ffprobe_bad_output_falls_back_to_wave(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="N/A\n", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = make_wav(str(tmp_path / "a4.wav"), seconds=0.75)

    duration = audio.get_audio_duration(path)
    assert duration == pytest.approx(0.75, abs=0.02)


def test_a5_ffprobe_filenotfound_falls_back_to_wave(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = make_wav(str(tmp_path / "a5.wav"), seconds=0.6)

    duration = audio.get_audio_duration(path)
    assert duration == pytest.approx(0.6, abs=0.02)


def test_a6_no_ffprobe_non_wav_raises_runtimeerror(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    fake_mp3 = tmp_path / "a6.mp3"
    fake_mp3.write_bytes(b"\x00\x01\x02\x03not a real mp3")

    with pytest.raises(RuntimeError, match="ffprobe"):
        audio.get_audio_duration(str(fake_mp3))


def test_a7_file_not_found_raises(tmp_path):
    from karaokeforge.utils import audio

    with pytest.raises(FileNotFoundError):
        audio.get_audio_duration(str(tmp_path / "missing.wav"))


# ---------------------------------------------------------------------------
# rms_level
# ---------------------------------------------------------------------------


def test_a8_rms_silence_is_zero(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a8.wav"), seconds=0.3, amplitude=0.0, waveform="silence")
    assert audio.rms_level(path) == 0.0


def test_a9_rms_full_scale_square_close_to_one(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a9.wav"), seconds=0.3, amplitude=1.0, waveform="square")
    assert audio.rms_level(path) == pytest.approx(1.0, abs=1e-3)


def test_a10_rms_sine_half_amplitude(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a10.wav"), seconds=0.3, amplitude=0.5, waveform="sine")
    assert audio.rms_level(path) == pytest.approx(0.3536, abs=0.01)


def test_a11_rms_8bit_unsigned_silence_subtracts_offset(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a11.wav"), seconds=0.3, sampwidth=1, amplitude=0.0, waveform="silence")
    assert audio.rms_level(path) == pytest.approx(0.0, abs=1e-6)


def test_a12_rms_stereo_one_channel_silent(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(
        str(tmp_path / "a12.wav"),
        seconds=0.3,
        channels=2,
        waveform="square",
        channel_amplitudes=[1.0, 0.0],
    )
    assert audio.rms_level(path) == pytest.approx(0.707, abs=0.01)


def test_a13_rms_zero_frames_no_zerodivision(tmp_path):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a13.wav"), seconds=0.0)
    assert audio.rms_level(path) == 0.0


def test_a14_rms_chunked_matches_default(tmp_path, monkeypatch):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a14.wav"), seconds=0.3, amplitude=0.5, waveform="sine")
    baseline = audio.rms_level(path)

    monkeypatch.setattr(audio, "_CHUNK_FRAMES", 100)
    chunked = audio.rms_level(path)

    assert chunked == pytest.approx(baseline, abs=1e-9)


def test_a15_rms_file_not_found_raises(tmp_path):
    from karaokeforge.utils import audio

    with pytest.raises(FileNotFoundError):
        audio.rms_level(str(tmp_path / "missing.wav"))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(amplitude=0.0, waveform="silence"),
        dict(amplitude=1.0, waveform="square"),
        dict(amplitude=0.5, waveform="sine"),
        dict(sampwidth=1, amplitude=0.3, waveform="sine"),
        dict(sampwidth=4, amplitude=0.8, waveform="sine"),
    ],
)
def test_a16_rms_always_in_unit_range(tmp_path, kwargs):
    from karaokeforge.utils import audio

    path = make_wav(str(tmp_path / "a16.wav"), seconds=0.2, **kwargs)
    value = audio.rms_level(path)
    assert 0.0 <= value <= 1.0


def test_a17_import_succeeds_without_torch_or_soundfile(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.delitem(sys.modules, "karaokeforge.utils.audio", raising=False)
    import karaokeforge.utils.audio  # noqa: F401 — chỉ cần import không raise


def test_no_wave_no_ffmpeg_raises_runtimeerror(tmp_path, monkeypatch):
    """Edge case §5.4 #3: WAV float32 -> wave.Error -> nhánh ffmpeg -> không có ffmpeg -> RuntimeError rõ."""
    from karaokeforge.utils import audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    fake_float_wav = tmp_path / "float.wav"
    fake_float_wav.write_bytes(b"RIFF....WAVEfmt not a real pcm header")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        audio.rms_level(str(fake_float_wav))


# ---------------------------------------------------------------------------
# PR-PIPE-FIX Fix 2 — rms_level catch rộng hơn wave.Error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("sampwidth không hỗ trợ: 8"),
        EOFError("unexpected end of data"),
        OSError("read error"),
    ],
)
def test_rms_wave_errors_fall_back_to_ffmpeg(tmp_path, monkeypatch, exc):
    """`_rms_from_wave` có thể ném EOFError/OSError/RuntimeError (không chỉ wave.Error);
    rms_level phải bắt cả 3 và rơi xuống nhánh ffmpeg thay vì để lỗi thoát ra."""
    from karaokeforge.utils import audio

    def _boom(path):
        raise exc

    monkeypatch.setattr(audio, "_rms_from_wave", _boom)
    monkeypatch.setattr(audio, "_rms_from_ffmpeg", lambda path: 0.42)
    path = make_wav(str(tmp_path / "err.wav"), seconds=0.1)

    assert audio.rms_level(path) == 0.42
