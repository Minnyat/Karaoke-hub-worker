"""Helper dùng chung cho test PR-A2A3 (namespace riêng theo DECISIONS.md G1).

Không import từ karaokeforge/utils/ (PR-A1) và không tạo conftest.py (G1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "lyrics.schema.json"
)


def load_lyrics_schema() -> dict[str, Any]:
    """Đọc contracts/lyrics.schema.json — nguồn sự thật cho LyricsSegment."""
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def assert_valid_lyrics(segments: list[dict]) -> None:
    """Validate `segments` theo contracts/lyrics.schema.json, ném lỗi nếu sai."""
    jsonschema.validate(instance=segments, schema=load_lyrics_schema())


def make_word(
    start: float = 0.0,
    end: float = 1.0,
    word: str = "w",
    confidence: float = 1.0,
) -> dict:
    """Dựng 1 WordTiming hợp lệ theo contract, dùng làm input test."""
    return {"start": start, "end": end, "word": word, "confidence": confidence}


def make_segment(
    start: float = 0.0,
    end: float = 1.0,
    text: str = "",
    words: list[dict] | None = None,
) -> dict:
    """Dựng 1 LyricsSegment hợp lệ theo contract, dùng làm input test."""
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": words if words is not None else [],
    }
