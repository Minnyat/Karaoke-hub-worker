"""Sinh frame video + convert sang bytes rawvideo rgb24 cho FFmpeg stdin. (PR-A4)

`frame_to_bytes` validate shape/dtype nghiêm ngặt (REQ-A4-08): sai shape/dtype
FFmpeg không báo lỗi, chỉ xuất video nhiễu sọc — bug rất tốn thời gian nếu lọt
qua (CLAUDE.md/PR-A4 plan mục 8, bẫy #7).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def frame_to_bytes(frame: np.ndarray, width: int, height: int) -> bytes:
    """Validate `frame` là `ndarray` shape `(height, width, 3)` dtype `uint8`,
    rồi trả bytes rawvideo rgb24 liền kề bộ nhớ. Sai -> `ValueError` nêu rõ
    shape/dtype thực tế."""
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame phải là numpy.ndarray, nhận {type(frame)!r}")
    expected_shape = (height, width, 3)
    if frame.shape != expected_shape:
        raise ValueError(f"frame shape sai: kỳ vọng {expected_shape}, nhận {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype sai: kỳ vọng uint8, nhận {frame.dtype}")
    return np.ascontiguousarray(frame).tobytes()


def iter_frame_bytes(
    template, lyrics: list[dict], total_frames: int, fps: int, width: int, height: int
) -> Iterator[tuple[int, bytes]]:
    """Yield `(frame_idx, bytes)` cho từng frame; `timestamp = frame_idx / fps`.

    Generator nội bộ (được phép — contract chỉ cấm `KaraokeRenderer.render()`
    là generator, xem REQ-A4-02).
    """
    for frame_idx in range(total_frames):
        timestamp = frame_idx / fps
        frame = template.render_frame(timestamp, lyrics, frame_idx)
        yield frame_idx, frame_to_bytes(frame, width, height)
