"""Stage 2: WhisperX transcribe + word-level align. (PR-A2A3)

Import whisperx/torch BÊN TRONG method — máy dev không cài được lib GPU.
Kết quả segment theo format LyricsSegment trong contracts/lyrics.schema.json:
  {"start": float, "end": float, "text": str,
   "words": [{"start", "end", "word", "confidence"}]}
"""
from __future__ import annotations

import logging

from karaokeforge.config import Config
from karaokeforge.utils.gpu import detect_gpu

from .aligner import (
    _split_lyrics_lines,
    apply_user_lyrics,
    interpolate_missing_word_timings,
)
from .vietnamese import fix_vietnamese

logger = logging.getLogger(__name__)

WHISPER_BATCH_SIZE = 16  # phải khớp Config.WHISPER_BATCH_SIZE (config.py, PR-B4)
SAMPLE_RATE = 16000  # whisperx.audio.SAMPLE_RATE
CPU_COMPUTE_TYPE = "int8"  # float16 không chạy được trên CPU (faster-whisper)

NO_SPEECH_ERROR = "Không nhận dạng được giọng hát. Hãy paste lời."


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


def _to_contract_segments(raw: list[dict]) -> list[dict]:
    """Chuẩn hoá segment thô (whisperx hoặc pre-align) về format LyricsSegment.

    Map `score → confidence`, giữ đúng 4 key mỗi segment/word (REQ-01), round
    3 chữ số (REQ-03). `start`/`end` của word thiếu (whisperx bỏ từ không
    align được) giữ nguyên `None` — sẽ được `interpolate_missing_word_timings`
    dựng lại (REQ-12b).
    """
    out = []
    for seg in raw:
        start = seg.get("start")
        end = seg.get("end")

        words_out = []
        for w in seg.get("words") or []:
            w_start = w.get("start")
            w_end = w.get("end")
            words_out.append({
                "start": round(float(w_start), 3) if w_start is not None else None,
                "end": round(float(w_end), 3) if w_end is not None else None,
                "word": str(w.get("word", "")),
                "confidence": _clamp_confidence(
                    w.get("score", w.get("confidence"))
                ),
            })

        out.append({
            "start": round(float(start), 3) if start is not None else 0.0,
            "end": round(float(end), 3) if end is not None else 0.0,
            "text": str(seg.get("text", "") or ""),
            "words": words_out,
        })
    return out


def _synthesize_segments(lines: list[str], duration: float) -> list[dict]:
    """Dựng segment tổng hợp chia đều `duration` cho số dòng lời user (REQ-09)."""
    n = len(lines)
    if n == 0 or duration <= 0:
        return []

    step = duration / n
    return [
        {
            "start": round(i * step, 3),
            "end": round((i + 1) * step, 3),
            "text": line,
            "words": [],
        }
        for i, line in enumerate(lines)
    ]


class LyricsTranscriber:
    """Whisper + WhisperX wrapper."""

    def transcribe_and_align(
        self,
        audio_path: str,
        language: str = "vi",
        user_lyrics: str | None = None,
        whisper_model: str = "large-v3",
    ) -> list[dict]:
        """Transcribe (hoặc forced-align theo user_lyrics) rồi align word-level.

        Thứ tự: load_model → load_audio → transcribe → (nếu user_lyrics:
        apply_user_lyrics) → load_align_model → align → cleanup VRAM →
        (nếu language == "vi": fix_vietnamese) → return segments.

        Raises:
            RuntimeError: Whisper không nhận dạng được giọng hát và không có
                `user_lyrics` để forced-align.
        """
        import gc
        import torch
        import whisperx

        # Fix 4: chọn device/compute_type theo GPU thực tế + Config, thay vì
        # hard-code "cuda"/float16 (Colab không cấp GPU → whisperx chết khó hiểu;
        # auto_select_models hạ int8 bị vô hiệu). Không GPU → CPU + int8
        # (float16 không chạy trên CPU).
        gpu = detect_gpu()
        if gpu["available"]:
            device = "cuda"
            compute_type = Config.WHISPER_COMPUTE_TYPE
        else:
            device = "cpu"
            compute_type = CPU_COMPUTE_TYPE

        model = None
        align_model = None
        try:
            model = whisperx.load_model(
                whisper_model,
                device,
                compute_type=compute_type,
                language=language,
            )
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

            detected = result.get("language")
            if detected and detected != language:
                logger.warning(
                    "Ngôn ngữ nhận dạng được (%s) khác ngôn ngữ yêu cầu (%s), "
                    "tiếp tục xử lý bằng %s",
                    detected, language, language,
                )

            raw_segments = [
                s for s in (result.get("segments") or [])
                if (s.get("text") or "").strip()
            ]
            lines = _split_lyrics_lines(user_lyrics or "")

            if not raw_segments:
                if not lines:
                    raise RuntimeError(NO_SPEECH_ERROR)
                duration = len(audio) / SAMPLE_RATE
                segments = _synthesize_segments(lines, duration)
            elif lines:
                segments = apply_user_lyrics(raw_segments, user_lyrics or "")
            else:
                segments = raw_segments

            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=language, device=device,
                )
                aligned = whisperx.align(
                    segments, align_model, metadata, audio, device,
                )
                out = _to_contract_segments(aligned.get("segments") or [])
            except Exception as exc:
                logger.error(
                    "WhisperX align thất bại, fallback nội suy timing: %s", exc,
                )
                out = _to_contract_segments(segments)

            out = interpolate_missing_word_timings(out)

            if language == "vi":
                out = fix_vietnamese(out)

            return out
        finally:
            del model
            del align_model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.warning("torch.cuda.empty_cache thất bại", exc_info=True)
