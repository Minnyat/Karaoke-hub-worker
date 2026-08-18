"""Helpers/fixtures riêng cho test PR-A4 (renderer + video templates).

Theo quy ước G1 (docs/plans/DECISIONS.md): mỗi PR có file helper riêng, không
dùng conftest.py chung trong wave 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageFont


def lyrics_fixture() -> list[dict[str, Any]]:
    """3 segment liền kề, mỗi segment có `words` — khớp `contracts/lyrics.schema.json`.

    Text dùng ASCII để không phụ thuộc font hỗ trợ dấu tiếng Việt trên máy test
    (Be Vietnam Pro không có sẵn trên máy dev, xem PR-A4 plan mục 4.7).
    """
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world",
            "words": [
                {"start": 0.0, "end": 1.0, "word": "Hello", "confidence": 0.9},
                {"start": 1.0, "end": 2.0, "word": "world", "confidence": 0.9},
            ],
        },
        {
            "start": 2.0,
            "end": 4.0,
            "text": "Second line here",
            "words": [
                {"start": 2.0, "end": 2.6, "word": "Second", "confidence": 0.9},
                {"start": 2.6, "end": 3.3, "word": "line", "confidence": 0.9},
                {"start": 3.3, "end": 4.0, "word": "here", "confidence": 0.9},
            ],
        },
        {
            "start": 4.0,
            "end": 6.0,
            "text": "Third line now",
            "words": [
                {"start": 4.0, "end": 4.6, "word": "Third", "confidence": 0.9},
                {"start": 4.6, "end": 5.3, "word": "line", "confidence": 0.9},
                {"start": 5.3, "end": 6.0, "word": "now", "confidence": 0.9},
            ],
        },
    ]


def gapped_lyrics_fixture() -> list[dict[str, Any]]:
    """5 segment với 1 khoảng trống (gap) giữa segment idx1 và idx2, dùng cho
    test `_get_active_lines` (T-L07)."""
    return [
        {"start": 0.0, "end": 2.0, "text": "one", "words": []},
        {"start": 2.0, "end": 4.0, "text": "two", "words": []},
        {"start": 4.5, "end": 6.5, "text": "three", "words": []},
        {"start": 6.5, "end": 8.5, "text": "four", "words": []},
        {"start": 8.5, "end": 10.5, "text": "five", "words": []},
    ]


def template_config(**overrides: Any) -> dict[str, Any]:
    """Config tối thiểu cho template test — mặc định trỏ font_dir không tồn tại
    (buộc fallback, nhanh, không cần font thật trên máy dev)."""
    config: dict[str, Any] = {"font_dir": "/nonexistent/fonts/dir"}
    config.update(overrides)
    return config


def renderer_config(**overrides: Any) -> dict[str, Any]:
    """Config tối thiểu cho test `KaraokeRenderer.render`."""
    config: dict[str, Any] = {"video_resolution": "720p", "video_template": "modern"}
    config.update(overrides)
    return config


def extract_real_font(tmp_dir: Path) -> str:
    """Trích xuất font TTF thật (Aileron, nhúng sẵn trong Pillow >=10.1 cho
    `ImageFont.load_default(size=...)`) ra `tmp_dir`, dùng làm "font tồn tại"
    cho test không phụ thuộc font hệ thống/Be Vietnam Pro (không có trên máy dev).
    """
    font_obj = ImageFont.load_default(size=40)
    font_path = Path(tmp_dir) / "test_real_font.ttf"
    font_path.write_bytes(font_obj.font_bytes)
    return str(font_path)


class FakeTemplate:
    """Template giả cho test renderer — không phụ thuộc Pillow, ghi lại lời gọi."""

    def __init__(self, width: int, height: int, config: dict | None = None) -> None:
        self.width = width
        self.height = height
        self.config = config or {}
        self.calls: list[tuple[float, int]] = []

    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        self.calls.append((timestamp, frame_idx))
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class RaisingTemplate(FakeTemplate):
    """Template ném lỗi từ frame `raise_after` trở đi — test propagate exception
    giữa chừng render (T-R10)."""

    def __init__(
        self, width: int, height: int, config: dict | None = None, raise_after: int = 2
    ) -> None:
        super().__init__(width, height, config)
        self.raise_after = raise_after

    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        if frame_idx >= self.raise_after:
            raise RuntimeError("boom trong template (test)")
        return super().render_frame(timestamp, lyrics, frame_idx)


def make_fake_process(returncode: int = 0, write_side_effect=None):
    """Fake `subprocess.Popen` return value: `.stdin` là MagicMock, `.wait()`
    trả `returncode`."""
    from unittest.mock import MagicMock

    proc = MagicMock(name="FakeFFmpegProcess")
    proc.stdin = MagicMock(name="stdin")
    if write_side_effect is not None:
        proc.stdin.write.side_effect = write_side_effect
    proc.wait.return_value = returncode
    return proc
