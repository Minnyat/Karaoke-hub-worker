"""Post-processing tiếng Việt: sửa dấu, chuẩn hoá text. (PR-A2A3)

Chỉ dùng từ điển tĩnh curated (v1). Không đưa vào dict những từ mà dạng
không dấu trùng với một từ tiếng Việt hợp lệ khác (REQ-27) — ví dụ
`nho` (nhỏ/nhớ/nhờ/nho quả), `cho` (cho/chờ), `ma` (ma/mà/má)... đều bị loại.
"""
from __future__ import annotations

import copy
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# key: dạng không dấu, casefold, NFC. value: dạng đúng dấu, NFC, chữ thường.
# Mỗi entry đã tự soi theo REQ-27 (không phải một từ tiếng Việt hợp lệ khác).
VIETNAMESE_CORRECTIONS: dict[str, str] = {
    "nguoi": "người",
    "khong": "không",
    "yeu": "yêu",
    "duoc": "được",
    "thuong": "thương",
    "buon": "buồn",
    "nguoc": "ngược",
    "nhung": "những",
}

_PUNCT = ",.!?…\"'()[]{}-–—:;«»"


def _fix_token(token: str) -> tuple[str, bool]:
    """Sửa 1 token, giữ dấu câu đầu/cuối + kiểu chữ hoa. Trả (token_mới, đã_sửa)."""
    token = unicodedata.normalize("NFC", token)
    left_stripped = token.lstrip(_PUNCT)
    prefix = token[: len(token) - len(left_stripped)]
    core = left_stripped.rstrip(_PUNCT)
    suffix = left_stripped[len(core):]

    if not core:
        return token, False

    repl = VIETNAMESE_CORRECTIONS.get(core.casefold())
    if repl is None:
        return token, False

    if core.isupper():
        new_core = repl.upper()
    elif core[:1].isupper():
        new_core = repl.capitalize()
    else:
        new_core = repl

    return prefix + new_core + suffix, True


def _fix_text(text: str) -> str:
    """Sửa từng token trong `text`, giữ nguyên format khoảng trắng gốc (REQ-28)."""

    def _apply(match: re.Match) -> str:
        old = match.group()
        new, changed = _fix_token(old)
        if changed:
            logger.info("fix_vietnamese: %s → %s", old, new)
        return new

    return re.sub(r"\S+", _apply, text)


def fix_vietnamese(segments: list[dict]) -> list[dict]:
    """Áp dụng từ điển sửa lỗi dấu phổ biến (nguoi→người, khong→không, ...).

    Không đổi `start`/`end`/`confidence` hay số lượng segment/word (REQ-23).
    Không mutate `segments` truyền vào (REQ-18). Mọi text/word đầu ra là NFC
    (REQ-25).
    """
    result = copy.deepcopy(segments)
    for seg in result:
        text = seg.get("text")
        if text is not None:
            seg["text"] = _fix_text(text)

        words = seg.get("words")
        if words:
            for w in words:
                word = w.get("word")
                if word is not None:
                    new_word, changed = _fix_token(word)
                    if changed:
                        logger.info("fix_vietnamese: %s → %s", word, new_word)
                    w["word"] = new_word
    return result
