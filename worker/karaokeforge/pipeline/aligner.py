"""Stage 2b: forced alignment với lời user cung cấp. (PR-A2A3)"""
from __future__ import annotations

import copy
import logging
import unicodedata

logger = logging.getLogger(__name__)


def _split_lyrics_lines(user_lyrics: str) -> list[str]:
    """Chuẩn hoá `user_lyrics` (paste tự do) thành danh sách dòng sạch.

    Chuẩn hoá `\\r\\n`/`\\r` → `\\n`, NFC, strip từng dòng, gộp khoảng trắng
    liên tiếp trong dòng thành 1 space, bỏ dòng rỗng (khổ ngăn cách).
    """
    if not user_lyrics:
        return []

    normalized = unicodedata.normalize("NFC", user_lyrics)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
    return lines


def apply_user_lyrics(segments: list[dict], user_lyrics: str) -> list[dict]:
    """Thay text Whisper nhận dạng bằng lời user paste, giữ timing để re-align.

    - `len(lines) == len(segments)`: map 1-1.
    - `len(lines) > len(segments)`: `len(segments)-1` segment đầu map 1-1;
      mọi dòng còn lại nối vào segment cuối bằng 1 space.
    - `len(lines) < len(segments)`: map 1-1 `len(lines)` segment đầu, bỏ các
      segment thừa; độ dài output == len(lines).
    - `words` của mọi segment được thay bị đổi thành `[]` (sẽ được dựng lại
      bởi whisperx.align hoặc interpolate_missing_word_timings).
    - Không mutate `segments` truyền vào.
    """
    result = copy.deepcopy(segments)
    lines = _split_lyrics_lines(user_lyrics)

    if not lines:
        return result

    n_seg = len(result)
    if n_seg == 0:
        return result

    n_lines = len(lines)
    if n_lines != n_seg:
        logger.warning(
            "apply_user_lyrics: số dòng lời (%d) khác số segment (%d)",
            n_lines, n_seg,
        )

    if n_lines == n_seg:
        mapped_lines = lines
    elif n_lines > n_seg:
        mapped_lines = lines[: n_seg - 1] + [" ".join(lines[n_seg - 1:])]
    else:
        result = result[:n_lines]
        mapped_lines = lines

    for seg, line in zip(result, mapped_lines):
        seg["text"] = line
        seg["words"] = []

    return result


def _clamp_confidence(value) -> float:
    """Ép `value` về `[0.0, 1.0]`, thiếu (`None`) → `0.0` (REQ-02)."""
    if value is None:
        return 0.0
    value = float(value)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _words_from_text(text: str) -> list[dict]:
    """Sinh danh sách word thô (chưa có timing) từ text tách theo khoảng trắng."""
    return [{"word": tok} for tok in (text or "").split()]


def interpolate_missing_word_timings(segments: list[dict]) -> list[dict]:
    """Fallback khi WhisperX align fail từng từ: nội suy đều từ start/end segment.

    - Segment không có `words`/`words` rỗng nhưng `text` có token → sinh word
      chia đều `[segment.start, segment.end]`, `confidence = 0.0`.
    - Từ thiếu `start`/`end` nằm giữa 2 mốc đã biết → nội suy chia đều giữa
      mốc trước và mốc sau; thiếu ở đầu/cuối → neo `segment.start`/`segment.end`.
    - `segment.end <= segment.start` (dữ liệu bẩn) → mọi word start=end=segment.start.
    - Kết quả luôn đơn điệu không giảm, mọi word có đúng 4 key
      (`start`, `end`, `word`, `confidence`), `start`/`end` round 3 chữ số.
    - Không mutate `segments` truyền vào.
    """
    result = copy.deepcopy(segments)

    for seg in result:
        s = float(seg.get("start") or 0.0)
        e = float(seg.get("end") or 0.0)

        words = seg.get("words") or []
        if not words:
            words = _words_from_text(seg.get("text", ""))

        if not words:
            seg["words"] = []
            seg["start"] = round(s, 3)
            seg["end"] = round(e, 3)
            continue

        if e <= s:
            seg["words"] = [
                {
                    "start": round(s, 3),
                    "end": round(s, 3),
                    "word": w.get("word", ""),
                    "confidence": _clamp_confidence(w.get("confidence")),
                }
                for w in words
            ]
            seg["start"] = round(s, 3)
            seg["end"] = round(e, 3)
            continue

        n = len(words)
        starts: list[float | None] = [None] * n
        ends: list[float | None] = [None] * n
        for i, w in enumerate(words):
            w_start = w.get("start")
            w_end = w.get("end")
            if w_start is not None and w_end is not None:
                starts[i] = float(w_start)
                ends[i] = float(w_end)

        i = 0
        while i < n:
            if starts[i] is not None:
                i += 1
                continue
            j = i
            while j < n and starts[j] is None:
                j += 1
            # run thiếu timing: [i, j-1]
            lo = ends[i - 1] if i > 0 else s
            hi = starts[j] if j < n else e
            if hi < lo:
                hi = lo
            run_len = j - i
            step = (hi - lo) / run_len
            for k in range(run_len):
                starts[i + k] = lo + k * step
                ends[i + k] = lo + (k + 1) * step
            i = j

        new_words = []
        prev_end = s
        for idx, w in enumerate(words):
            ws = starts[idx]
            we = ends[idx]
            ws = max(ws, prev_end)
            ws = min(ws, e)
            we = max(we, ws)
            we = min(we, e)
            ws = round(ws, 3)
            we = round(we, 3)
            new_words.append({
                "start": ws,
                "end": we,
                "word": w.get("word", ""),
                "confidence": _clamp_confidence(w.get("confidence")),
            })
            prev_end = we

        seg["words"] = new_words
        seg["start"] = round(s, 3)
        seg["end"] = round(e, 3)

    return result
