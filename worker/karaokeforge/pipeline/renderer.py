"""Stage 3: render video karaoke. (PR-A4)

LƯU Ý contract (mục 6.2 contracts/README.md): render() nhận callback
on_progress, KHÔNG phải generator như pseudo-code trong PRD.

Guide vocal: theo docs/plans/DECISIONS.md (PR-A4) — renderer KHÔNG mix vocal.
`instrumental_path` là audio track duy nhất được dùng làm `-i` thứ 2 của
FFmpeg; PR-B4 chịu trách nhiệm mix vocals vào instrumental (nếu
`config.guide_vocal=true`) TRƯỚC khi gọi `render()`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable

from karaokeforge.utils.audio import get_audio_duration
from karaokeforge.video.frame_generator import iter_frame_bytes
from karaokeforge.video.templates import get_template

logger = logging.getLogger(__name__)

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
DEFAULT_RESOLUTION = "1080p"
DEFAULT_FPS = 30
DEFAULT_TEMPLATE = "modern"
DEFAULT_PRESET = "fast"
DEFAULT_CRF = 23
PROGRESS_INTERVAL_S = 5
FFMPEG_RETRIES = 1
STDERR_TAIL_LINES = 20


class RenderError(RuntimeError):
    """FFmpeg render thất bại sau khi retry, hoặc không tìm thấy binary ffmpeg."""


class _FfmpegCrash(RuntimeError):
    """Lỗi nội bộ: FFmpeg thoát với mã lỗi khác 0, hoặc stdin bị đóng bất ngờ
    (BrokenPipeError). Không lộ ra ngoài `render()` — bị bắt để retry/raise
    thành `RenderError` (REQ-A4-05)."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class _FfmpegNotFound(RuntimeError):
    """Lỗi nội bộ: không tìm thấy binary `ffmpeg` (Popen raise FileNotFoundError).
    Tách riêng để `render()` không bắt nhầm FileNotFoundError từ nguồn khác
    (vd tempfile.mkstemp) và báo sai "thiếu ffmpeg" (PR-A4-FIX)."""


class KaraokeRenderer:
    """Render video từ instrumental + lyrics timing, pipe frame vào FFmpeg."""

    def render(
        self,
        instrumental_path: str,
        lyrics: list[dict],
        output_path: str,
        config: dict,
        on_progress: Callable[[float], None] | None = None,
    ) -> str:
        """Render video hoàn chỉnh. Trả về output_path.

        on_progress được gọi với percent (0-100), tối đa ~1 lần/5s video.
        Phải đóng stdin của FFmpeg trước wait() (pitfall trong CLAUDE.md).
        """
        width, height = self._parse_resolution(config)
        fps = int(config.get("fps", DEFAULT_FPS)) or DEFAULT_FPS

        duration = get_audio_duration(instrumental_path)
        total_frames = max(1, int(duration * fps))

        render_config = {**config, "duration": duration, "fps": fps}
        template_name = config.get("video_template", DEFAULT_TEMPLATE)
        template = get_template(template_name, width, height, render_config)

        cmd = self._build_ffmpeg_cmd(width, height, fps, instrumental_path, output_path, config)

        attempts = FFMPEG_RETRIES + 1
        for attempt in range(attempts):
            try:
                self._run_once(cmd, template, lyrics, total_frames, fps, width, height, on_progress)
            except _FfmpegNotFound as exc:
                raise RenderError(
                    "Không tìm thấy binary 'ffmpeg' trong PATH. "
                    "Cài đặt FFmpeg trước khi render."
                ) from exc
            except _FfmpegCrash as exc:
                self._save_partial(output_path)
                if attempt < attempts - 1:
                    logger.warning(
                        "FFmpeg crash (lần %d/%d), thử lại: %s", attempt + 1, attempts, exc
                    )
                    continue
                raise RenderError(
                    f"FFmpeg render thất bại sau {attempts} lần thử: {exc}\n"
                    f"--- stderr (tail) ---\n{exc.stderr_tail}"
                ) from exc
            else:
                if on_progress is not None:
                    on_progress(100.0)
                return output_path

        # Không lý thuyết nào tới được đây (vòng lặp luôn return hoặc raise ở
        # lần thử cuối), nhưng giữ lại để mypy/đọc code rõ ràng.
        raise RenderError("FFmpeg render thất bại không rõ lý do.")

    # -- helpers -----------------------------------------------------------

    def _parse_resolution(self, config: dict) -> tuple[int, int]:
        """`video_resolution` -> `(width, height)` (REQ-A4-06). So khớp
        case-insensitive; thiếu key -> 1080p; giá trị lạ -> ValueError."""
        raw = config.get("video_resolution", DEFAULT_RESOLUTION)
        key = str(raw).strip().lower()
        if key not in RESOLUTIONS:
            valid = ", ".join(sorted(RESOLUTIONS))
            raise ValueError(
                f"video_resolution không hợp lệ: {raw!r}. Giá trị hợp lệ: {valid}"
            )
        return RESOLUTIONS[key]

    def _build_ffmpeg_cmd(
        self,
        width: int,
        height: int,
        fps: int,
        instrumental_path: str,
        output_path: str,
        config: dict,
    ) -> list[str]:
        """Lệnh FFmpeg pipe rawvideo qua stdin + audio = instrumental (REQ-A4-03)."""
        preset = config.get("ffmpeg_preset", DEFAULT_PRESET)
        crf = config.get("ffmpeg_crf", DEFAULT_CRF)
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            instrumental_path,
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]

    def _run_once(
        self,
        cmd: list[str],
        template,
        lyrics: list[dict],
        total_frames: int,
        fps: int,
        width: int,
        height: int,
        on_progress: Callable[[float], None] | None,
    ) -> None:
        """Chạy FFmpeg 1 lần: pipe toàn bộ frame vào stdin rồi chờ thoát.

        stdin LUÔN được đóng trong `finally` trước khi gọi `wait()` (REQ-A4-04,
        CLAUDE.md pitfall). stderr ghi ra file tạm (không dùng PIPE không đọc
        — sẽ treo khi buffer OS đầy, mục 4.2/8.2 plan).
        """
        stderr_path: str | None = None
        try:
            fd, stderr_path = tempfile.mkstemp(prefix="ffmpeg_stderr_", suffix=".log")
            os.close(fd)

            with open(stderr_path, "wb") as stderr_handle:
                # FileNotFoundError chỉ được bắt QUANH Popen (ffmpeg thiếu binary);
                # ENOENT từ nơi khác không bị báo nhầm "thiếu ffmpeg" (PR-A4-FIX).
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_handle)
                except FileNotFoundError as exc:
                    raise _FfmpegNotFound(str(exc)) from exc

                returncode: int | None = None
                crash_exc: Exception | None = None
                try:
                    interval = max(1, fps * PROGRESS_INTERVAL_S)
                    for frame_idx, chunk in iter_frame_bytes(
                        template, lyrics, total_frames, fps, width, height
                    ):
                        try:
                            proc.stdin.write(chunk)
                        except BrokenPipeError as exc:
                            crash_exc = exc
                            break
                        if on_progress is not None and frame_idx % interval == 0:
                            on_progress(min(100.0, frame_idx / total_frames * 100))
                finally:
                    # stdin PHẢI đóng trước wait() (REQ-A4-04). close() có thể tự
                    # raise (BrokenPipe/OSError) khi ffmpeg đã chết -> gom vào
                    # crash_exc thay vì để lọt thô, mất retry (PR-A4-FIX a).
                    if proc.stdin is not None:
                        try:
                            proc.stdin.close()
                        except OSError as exc:
                            if crash_exc is None:
                                crash_exc = exc
                    # wait() PHẢI chạy kể cả khi iter_frame_bytes raise, nếu không
                    # ffmpeg mồ côi (PR-A4-FIX b). Nếu đang unwind exception (không
                    # phải BrokenPipe đã bắt) mà ffmpeg còn sống -> kill trước để
                    # wait() không treo chờ EOF.
                    if (
                        crash_exc is None
                        and sys.exc_info()[0] is not None
                        and proc.poll() is None
                    ):
                        proc.kill()
                    returncode = proc.wait()

            if crash_exc is not None:
                raise _FfmpegCrash(
                    f"FFmpeg đóng stdin bất ngờ (BrokenPipeError): {crash_exc}",
                    self._read_stderr_tail(stderr_path),
                ) from crash_exc
            if returncode != 0:
                raise _FfmpegCrash(
                    f"FFmpeg thoát với mã lỗi {returncode}",
                    self._read_stderr_tail(stderr_path),
                )
        finally:
            if stderr_path is not None:
                try:
                    os.unlink(stderr_path)
                except OSError:
                    pass

    def _save_partial(self, output_path: str) -> None:
        """Giữ video dở lại thành `<output_path>.partial` nếu FFmpeg đã ghi
        được gì đó trước khi crash (PRD S8: "Save partial video if possible")."""
        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                os.replace(output_path, f"{output_path}.partial")
        except OSError as exc:
            logger.warning("Không thể lưu video dở dang: %s", exc)

    def _read_stderr_tail(self, stderr_path: str, lines: int = STDERR_TAIL_LINES) -> str:
        try:
            with open(stderr_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        tail = content.splitlines()[-lines:]
        return "\n".join(tail)
