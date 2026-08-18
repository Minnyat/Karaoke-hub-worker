"""Checkpoint recovery logic. (PR-B2B3)"""

from __future__ import annotations

_STAGE_ORDER = (
    ("audio_separated", "audio_separation"),
    ("lyrics_aligned", "lyrics_alignment"),
    ("video_rendered", "video_render"),
)


def resume_stage(job: dict) -> str | None:
    """Từ job['checkpoints'] suy ra stage cần chạy tiếp.

    Trả về một trong: "audio_separation" | "lyrics_alignment" | "video_render"
    hoặc None nếu đã xong cả 3.

    Robust: thiếu key `checkpoints` hoặc thiếu key con → coi như False, không
    `KeyError` (REQ-C03) — job JSON cũ/hỏng vẫn resume được từ đầu.
    Checkpoint bất thường (ví dụ `lyrics_aligned=True` nhưng `audio_separated=False`)
    vẫn trả về stage sớm nhất chưa xong, không nhảy cóc (REQ-C02) — pipeline chạy
    tuần tự, stage sau cần output của stage trước.
    """
    checkpoints = job.get("checkpoints") or {}
    for checkpoint_name, stage in _STAGE_ORDER:
        if not checkpoints.get(checkpoint_name):
            return stage
    return None
