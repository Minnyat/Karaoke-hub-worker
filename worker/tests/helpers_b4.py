"""Helper dùng chung cho test PR-B4 (namespace riêng theo DECISIONS.md G1 —
không có worker/tests/conftest.py trong wave 1/2).

Import lại `make_job`/`validate`/`put_pending`/`read_json`/`iso` từ `helpers_b`
(DECISIONS.md G1: trùng lặp giữa 2 file helper được chấp nhận; import chéo giữa
2 helper cũng được phép vì `helpers_b.py` đã ở `main`). Phần mới: fake 3 pipeline
class + fixture Drive cho worker.

Không phải test, không được pytest tự thu thập (không có hàm bắt đầu bằng `test_`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import karaokeforge.worker as worker_module
from karaokeforge.config import Config
from karaokeforge.drive import DriveQueue, DriveStorage
from karaokeforge.worker import KaraokeWorker

from .helpers_b import (  # noqa: F401 — re-export cho test_worker.py
    ISO_FMT,
    iso,
    make_job,
    parse_iso,
    put_pending,
    put_processing,
    read_json,
    validate,
)

# ----------------------------------------------------------------------
# call log + control dùng chung giữa 3 fake pipeline class
# ----------------------------------------------------------------------

CALLS: list[tuple[str, dict]] = []
CONTROL: dict[str, Any] = {}


def reset_calls() -> None:
    CALLS.clear()
    CONTROL.clear()


def _record(name: str, **kwargs: Any) -> None:
    CALLS.append((name, kwargs))


def calls_of(name: str) -> list[dict]:
    return [kwargs for call_name, kwargs in CALLS if call_name == name]


def call_names() -> list[str]:
    return [name for name, _ in CALLS]


def record_call(name: str, **kwargs: Any) -> None:
    """API công khai cho test tự ghi thêm sự kiện vào CALLS dùng chung (ví dụ
    lệnh `ffmpeg` mix guide vocal mà test tự patch `subprocess.run`) — để so
    thứ tự với call log của 3 fake pipeline class bằng `call_names()`."""
    _record(name, **kwargs)


def _write_fake_file(path: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


# ----------------------------------------------------------------------
# fake pipeline
# ----------------------------------------------------------------------


class FakeSeparator:
    """Ghi 2 file WAV giả (vài byte) thay vì chạy Demucs thật."""

    def __init__(self) -> None:
        self.model_name: str | None = None

    def load_model(self, model_name: str = "htdemucs_ft") -> None:
        self.model_name = model_name
        _record("separator.load_model", model_name=model_name)

    def separate(self, audio_path: str, output_dir: str) -> dict[str, str]:
        _record("separator.separate", audio_path=audio_path, output_dir=output_dir)
        hook = CONTROL.get("separator_before_return")
        if hook is not None:
            hook()
        exc = CONTROL.get("separator_raise")
        if exc is not None:
            raise exc
        vocals_path = os.path.join(output_dir, "vocals.wav")
        inst_path = os.path.join(output_dir, "instrumental.wav")
        _write_fake_file(vocals_path, b"FAKEWAV-VOCALS")
        _write_fake_file(inst_path, b"FAKEWAV-INSTRUMENTAL")
        return {"vocals": vocals_path, "instrumental": inst_path}

    def unload(self) -> None:
        _record("separator.unload")


DEFAULT_FAKE_SEGMENTS: list[dict] = [
    {
        "start": 0.0,
        "end": 1.5,
        "text": "xin chào việt nam",
        "words": [
            {"start": 0.0, "end": 0.5, "word": "xin", "confidence": 0.9},
            {"start": 0.5, "end": 1.0, "word": "chào", "confidence": 0.9},
            {"start": 1.0, "end": 1.5, "word": "việt nam", "confidence": 0.9},
        ],
    },
]


class FakeTranscriber:
    """Trả segment giả thay vì chạy WhisperX thật."""

    def transcribe_and_align(
        self,
        audio_path: str,
        language: str = "vi",
        user_lyrics: str | None = None,
        whisper_model: str = "large-v3",
    ) -> list[dict]:
        _record(
            "transcriber.transcribe_and_align",
            audio_path=audio_path,
            language=language,
            user_lyrics=user_lyrics,
            whisper_model=whisper_model,
        )
        hook = CONTROL.get("transcriber_before_return")
        if hook is not None:
            hook()
        exc = CONTROL.get("transcriber_raise")
        if exc is not None:
            raise exc
        segments = CONTROL.get("transcriber_segments")
        if segments is None:
            segments = DEFAULT_FAKE_SEGMENTS
        return segments


class FakeRenderer:
    """Ghi 1 file MP4 giả thay vì chạy FFmpeg thật; hỗ trợ gọi `on_progress`."""

    def render(
        self,
        instrumental_path: str,
        lyrics: list[dict],
        output_path: str,
        config: dict,
        on_progress: Callable[[float], None] | None = None,
    ) -> str:
        _record(
            "renderer.render",
            instrumental_path=instrumental_path,
            lyrics=lyrics,
            output_path=output_path,
            config=dict(config),
        )
        progress_calls = CONTROL.get("renderer_progress_calls")
        if progress_calls and on_progress is not None:
            progress_hook = CONTROL.get("renderer_progress_hook")
            for i, pct in enumerate(progress_calls):
                on_progress(pct)
                if progress_hook is not None:
                    progress_hook(i, pct)
        hook = CONTROL.get("renderer_after_progress")
        if hook is not None:
            hook()
        exc = CONTROL.get("renderer_raise")
        if exc is not None:
            raise exc
        _write_fake_file(output_path, b"FAKEMP4")
        if on_progress is not None:
            on_progress(100.0)
        return output_path


# ----------------------------------------------------------------------
# fixture Drive + worker
# ----------------------------------------------------------------------


def setup_drive(
    tmp_path: Path,
    job: dict,
    *,
    with_audio: bool = True,
    lyrics: str | None = None,
) -> str:
    """Tạo `queue/pending/{id}.json`, `uploads/{id}/original.<ext>` (nếu
    `with_audio`), `uploads/{id}/lyrics_input.txt` (nếu `lyrics` truyền vào).
    Trả về `drive_root` (str)."""
    root = tmp_path / "drive"
    put_pending(str(root), job)

    if with_audio:
        upload_dir = root / "uploads" / job["id"]
        upload_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(job["input"]["audio_filename"])[1] or ".mp3"
        (upload_dir / f"original{ext}").write_bytes(b"FAKE-ORIGINAL-AUDIO")

    if lyrics is not None:
        upload_dir = root / "uploads" / job["id"]
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / "lyrics_input.txt").write_text(lyrics, encoding="utf-8")

    return str(root)


def seed_outputs(root: str | Path, job_id: str, files: dict[str, bytes]) -> Path:
    """Ghi thẳng file vào `outputs/{job_id}/` (giả lập output đã publish từ lần
    chạy trước — dùng cho test resume)."""
    out_dir = Path(root) / "outputs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_bytes(content)
    return out_dir


def fake_lyrics_bytes(segments: list[dict] | None = None) -> bytes:
    return json.dumps(segments or DEFAULT_FAKE_SEGMENTS, ensure_ascii=False).encode("utf-8")


def make_worker(
    tmp_path: Path,
    monkeypatch,
    *,
    root: str | None = None,
    poll_interval: float = 0.0,
    heartbeat_interval: float = 0.01,
    sleep_fn: Callable[[float], None] | None = None,
    **queue_kwargs: Any,
) -> KaraokeWorker:
    """Patch 3 pipeline class trong module `worker` + `Config.TEMP_DIR`, trả về
    `KaraokeWorker` với `DriveQueue` THẬT (claim_settle_s=0, sleep_fn giả) trên
    `tmp_path`. `worker.sleep_calls` (list) ghi lại mọi lần gọi `sleep_fn` khi
    không truyền `sleep_fn` riêng."""
    reset_calls()

    drive_root = root if root is not None else str(tmp_path / "drive")
    temp_dir = tmp_path / "temp"
    monkeypatch.setattr(Config, "TEMP_DIR", str(temp_dir))

    monkeypatch.setattr(worker_module, "AudioSeparator", FakeSeparator)
    monkeypatch.setattr(worker_module, "LyricsTranscriber", FakeTranscriber)
    monkeypatch.setattr(worker_module, "KaraokeRenderer", FakeRenderer)

    queue_kwargs.setdefault("claim_settle_s", 0)
    queue_kwargs.setdefault("sleep_fn", lambda s: None)
    queue = DriveQueue(drive_root, **queue_kwargs)
    storage = DriveStorage(drive_root)

    sleep_calls: list[float] = []
    if sleep_fn is None:

        def sleep_fn(seconds: float) -> None:  # noqa: ANN001
            sleep_calls.append(seconds)

    worker = KaraokeWorker(
        "worker_test",
        drive_root,
        queue=queue,
        storage=storage,
        poll_interval=poll_interval,
        heartbeat_interval=heartbeat_interval,
        sleep_fn=sleep_fn,
    )
    worker.sleep_calls = sleep_calls  # type: ignore[attr-defined]
    return worker


def wait_until(
    predicate: Callable[[], bool], timeout: float = 3.0, interval: float = 0.01
) -> bool:
    """Poll `predicate()` cho tới khi True hoặc hết `timeout` — tránh
    `time.sleep` cố định trong test heartbeat (bẫy #14 mục 8 plan)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def read_json_retry(path: str | Path, timeout: float = 1.0, interval: float = 0.005) -> dict:
    """`read_json` chịu được ghi dở dang (JSON rách trong lúc heartbeat/update
    đang ghi đè file) — poll tới khi đọc được hoặc hết `timeout`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return read_json(path)
        except (OSError, json.JSONDecodeError):
            time.sleep(interval)
    return read_json(path)


def processing_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / "queue" / "processing" / f"{job_id}.json"


def pending_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / "queue" / "pending" / f"{job_id}.json"


def completed_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / "queue" / "completed" / f"{job_id}.json"


def failed_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / "queue" / "failed" / f"{job_id}.json"


def output_path(root: str | Path, job_id: str, name: str) -> Path:
    return Path(root) / "outputs" / job_id / name
