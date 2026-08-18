"""Audio helpers. (PR-A1)

Không import torch/soundfile ở top-level — chỉ dùng stdlib (`wave`,
`subprocess`, `shutil`) + numpy (có trong requirements-dev.txt). Máy dev
Windows không có ffmpeg/ffprobe: mọi nhánh phụ thuộc binary ngoài phải
fallback an toàn về stdlib `wave`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave

import numpy as np

_CHUNK_FRAMES = 65536  # đọc theo chunk, không nạp cả file vào RAM (bẫy #11)

_PCM_INT_DTYPE = {1: np.dtype("u1"), 2: np.dtype("<i2"), 4: np.dtype("<i4")}
_PCM_MAX_VAL = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}


def _ffprobe_duration(path: str) -> float | None:
    """Chạy ffprobe để lấy duration. Trả None nếu ffprobe không có/không parse được
    (không bao giờ để lỗi thoát ra — dùng cho fallback)."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def _wave_duration(path: str) -> float | None:
    """Đọc duration bằng stdlib `wave`. Trả None nếu không mở được (không phải
    WAV PCM hợp lệ, ví dụ float WAV) — không để lỗi thoát ra."""
    try:
        with wave.open(path, "rb") as wf:
            framerate = wf.getframerate()
            if framerate == 0:
                return None
            return wf.getnframes() / framerate
    except (wave.Error, EOFError, OSError):
        return None


def get_audio_duration(path: str) -> float:
    """Duration (giây) của file audio — dùng ffprobe hoặc fallback stdlib `wave`."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    duration = _ffprobe_duration(path)
    if duration is not None:
        return duration

    duration = _wave_duration(path)
    if duration is not None:
        return duration

    raise RuntimeError(
        f"Không đọc được duration của {path}: cần ffprobe hoặc file WAV PCM"
    )


def _decode_pcm_chunk(raw: bytes, sampwidth: int) -> np.ndarray:
    """Giải mã 1 chunk PCM thô thành mảng int (chưa chuẩn hoá), flatten mọi kênh."""
    if sampwidth == 3:
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = arr[: (arr.size // 3) * 3].reshape(-1, 3)
        values = (
            arr[:, 0].astype(np.int32)
            | (arr[:, 1].astype(np.int32) << 8)
            | (arr[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        return np.where(values & sign_bit, values - (1 << 24), values)
    dtype = _PCM_INT_DTYPE[sampwidth]
    return np.frombuffer(raw, dtype=dtype)


def _rms_from_wave(path: str) -> float:
    with wave.open(path, "rb") as wf:
        nframes = wf.getnframes()
        if nframes == 0:
            return 0.0
        sampwidth = wf.getsampwidth()
        if sampwidth not in _PCM_MAX_VAL:
            raise RuntimeError(f"sampwidth không hỗ trợ: {sampwidth}")
        max_val = _PCM_MAX_VAL[sampwidth]
        offset = 128.0 if sampwidth == 1 else 0.0

        total_sq = 0.0
        total_samples = 0
        remaining = nframes
        while remaining > 0:
            read_frames = min(_CHUNK_FRAMES, remaining)
            raw = wf.readframes(read_frames)
            if not raw:
                break
            remaining -= read_frames
            samples = _decode_pcm_chunk(raw, sampwidth)
            if samples.size == 0:
                continue
            normalized = (samples.astype(np.float64) - offset) / max_val
            total_sq += float(np.sum(normalized * normalized))
            total_samples += normalized.size

        if total_samples == 0:
            return 0.0
        return float(np.sqrt(total_sq / total_samples))


def _rms_from_ffmpeg(path: str) -> float:
    """Fallback khi `wave` không đọc được file (mp3/m4a/float-WAV) — decode qua ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Không đọc được {path} bằng wave và không có ffmpeg để decode"
        )
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-"],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ffmpeg không decode được {path}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg không decode được {path}: {proc.stderr!r}")
    if not proc.stdout:
        return 0.0
    samples = np.frombuffer(proc.stdout, dtype="<i2")
    if samples.size == 0:
        return 0.0
    normalized = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))


def rms_level(path: str) -> float:
    """RMS trung bình chuẩn hoá về [0, 1] — dùng phát hiện track im lặng (S8)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return _rms_from_wave(path)
    except (wave.Error, EOFError, OSError, RuntimeError):
        # Fix 2: `_rms_from_wave` không chỉ ném wave.Error — file cụt (EOFError),
        # lỗi I/O (OSError) hay sampwidth lạ (RuntimeError) đều phải rơi xuống
        # nhánh ffmpeg thay vì để lỗi thoát ra ngoài.
        return _rms_from_ffmpeg(path)
