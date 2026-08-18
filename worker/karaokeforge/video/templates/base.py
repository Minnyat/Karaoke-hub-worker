"""BaseTemplate — interface chung cho mọi template. Spec: PRD S3.5. (PR-A4)

Chứa helper logic thuần được test kỹ, độc lập I/O ngoài việc nạp font:
`_load_fonts` (REQ-A4-09/10), `_get_active_lines` (REQ-A4-11),
`_get_word_highlight_progress` (REQ-A4-12, đã sửa bug chia 0 của PRD S3.5).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from karaokeforge.video import effects

logger = logging.getLogger(__name__)

DEFAULT_FONT_DIR = "/usr/share/fonts/custom"
DEFAULT_FONT = "BeVietnamPro-Bold.ttf"
FALLBACK_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
BASE_SIZES = {"current": 48, "other": 36, "small": 24}
FONT_EXTENSIONS = (".ttf", ".otf")
# Dòng hiện tại dùng Bold, dòng phụ (other/small) dùng Regular (DECISIONS PR-A4).
FONT_WEIGHT_BY_KEY = {"current": "Bold", "other": "Regular", "small": "Regular"}


class BaseTemplate(ABC):
    """Render từng frame karaoke thành numpy array RGB (H, W, 3) uint8."""

    def __init__(self, width: int, height: int, config: dict) -> None:
        self.width = width
        self.height = height
        self.config = config
        self.scale = height / 1080
        self.font_fallback_used: bool = False
        self.fonts = self._load_fonts()
        self._bg_cache = self._build_background()

    @abstractmethod
    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        """Render 1 frame tại timestamp (giây). Trả về np.ndarray RGB."""
        raise NotImplementedError

    # -- nền -----------------------------------------------------------

    def _build_background(self) -> Image.Image:
        """Nền mặc định: đen tuyền. Subclass override cho gradient/overlay riêng."""
        bg = self._resolve_color("background_color", "#000000")
        return Image.new("RGB", (self.width, self.height), bg)

    def _resolve_color(self, key: str, default_hex: str) -> tuple[int, int, int]:
        """Đọc màu từ `config[key]` (ghi đè) hoặc dùng mặc định của template
        (REQ-A4-15). Hex sai định dạng -> ValueError (fail fast)."""
        value = self.config.get(key)
        if value:
            return effects.hex_to_rgb(value)
        return effects.hex_to_rgb(default_hex)

    def _new_canvas(self) -> Image.Image:
        """Canvas mới cho 1 frame, dùng lại nền đã cache (tối ưu tốc độ, mục 4.5 plan)."""
        return self._bg_cache.copy()

    def _to_array(self, img: Image.Image) -> np.ndarray:
        """Chuyển `Image` (bất kỳ mode) -> numpy RGB uint8, liền kề bộ nhớ."""
        return np.ascontiguousarray(np.array(img.convert("RGB"), dtype=np.uint8))

    def _duration(self, lyrics: list[dict]) -> float:
        """Tổng thời lượng để vẽ progress bar: `config["duration"]` hoặc
        `lyrics[-1]["end"]`; không có gì -> 0.0 (template ẩn progress bar)."""
        value = self.config.get("duration")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if lyrics:
            return float(lyrics[-1].get("end", 0.0))
        return 0.0

    # -- font ------------------------------------------------------------

    def _load_fonts(self) -> dict[str, ImageFont.FreeTypeFont]:
        """Trả `{"current", "other", "small"}`, size scale theo resolution
        (REQ-A4-09). Thiếu font -> fallback DejaVuSans/NotoSansCJK/load_default,
        log warning, KHÔNG raise (REQ-A4-10)."""
        font_dir = self.config.get("font_dir", DEFAULT_FONT_DIR)
        font_name = self.config.get("font", DEFAULT_FONT)
        return {
            key: self._load_single_font(
                str(Path(font_dir) / self._font_filename(font_name, key)),
                max(8, int(base_size * self.scale)),
            )
            for key, base_size in BASE_SIZES.items()
        }

    @staticmethod
    def _font_filename(font_name: str, key: str) -> str:
        """Map `config["font"]` -> tên file thật cho từng dòng (PR-A4-FIX).

        Nếu `font_name` đã là tên file (`*.ttf`/`*.otf`) -> dùng nguyên. Nếu là
        tên family (vd "Be Vietnam Pro", không đuôi) -> bỏ khoảng trắng + hậu tố
        weight theo dòng: current=Bold, other/small=Regular. Trước đây ghép thẳng
        family thành path -> luôn fallback DejaVu -> mọi video sai font tiếng Việt.
        """
        if font_name.lower().endswith(FONT_EXTENSIONS):
            return font_name
        base = font_name.replace(" ", "")
        weight = FONT_WEIGHT_BY_KEY.get(key, "Regular")
        return f"{base}-{weight}.ttf"

    def _load_single_font(self, primary_path: str, size: int) -> ImageFont.FreeTypeFont:
        candidates = [primary_path, *FALLBACK_FONT_PATHS]
        for index, path in enumerate(candidates):
            try:
                font = ImageFont.truetype(path, size)
            except OSError:
                continue
            if index > 0:
                self.font_fallback_used = True
                logger.warning(
                    "Font '%s' không dùng được, fallback sang '%s'", primary_path, path
                )
            return font
        self.font_fallback_used = True
        logger.warning(
            "Không tìm được font hệ thống nào trong danh sách fallback, "
            "dùng ImageFont.load_default(size=%d)",
            size,
        )
        return ImageFont.load_default(size=size)

    # -- lyrics timing (logic thuần, test kỹ) -----------------------------

    def _get_active_lines(
        self, timestamp: float, lyrics: list[dict], count: int = 3
    ) -> tuple[list[dict], int]:
        """Trả `(window, active_in_window)`.

        `lyrics == []` -> `([], -1)` (REQ-A4-11, quy ước DECISIONS.md PR-A4).
        `window` là slice thật của `lyrics` (cùng object, không copy).
        """
        if not lyrics:
            return [], -1

        n = len(lyrics)
        current_idx = n - 1
        for i, segment in enumerate(lyrics):
            # lyrics_edited.json (WebUI, D4) có thể thiếu key start/end -> .get
            # tránh KeyError (PR-A4-FIX).
            seg_start = segment.get("start", 0.0)
            seg_end = segment.get("end", 0.0)
            if seg_start <= timestamp <= seg_end:
                current_idx = i
                break
            if timestamp < seg_start:
                current_idx = max(0, i - 1)
                break

        half = count // 2
        start = current_idx - half
        end = start + count
        if start < 0:
            end += -start
            start = 0
        if end > n:
            start -= end - n
            end = n
        start = max(0, start)

        window = lyrics[start:end]
        active_in_window = current_idx - start
        return window, active_in_window

    def _get_word_highlight_progress(self, timestamp: float, segment: dict) -> float:
        """Tiến độ hát trong dòng hiện tại, luôn ∈ [0.0, 1.0] (REQ-A4-12).

        Sửa bug PRD S3.5: `word["end"] - word["start"]` có thể = 0 (WhisperX
        hoàn toàn có thể trả timing như vậy) -> không chia 0, coi như từ đó
        "hát xong ngay lập tức".
        """
        raw_words = segment.get("words") or []
        valid_words: list[tuple[float, float]] = []
        words_usable = bool(raw_words)
        for word in raw_words:
            start = word.get("start")
            end = word.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                words_usable = False
                break
            valid_words.append((float(start), float(end)))

        if not words_usable or not valid_words:
            seg_start = segment.get("start", 0.0)
            seg_end = segment.get("end", 0.0)
            if seg_end - seg_start <= 0:
                return 0.0
            return effects.clamp01((timestamp - seg_start) / (seg_end - seg_start))

        n = len(valid_words)
        if timestamp < valid_words[0][0]:
            return 0.0
        if timestamp >= valid_words[-1][1]:
            return 1.0

        for i, (w_start, w_end) in enumerate(valid_words):
            duration = w_end - w_start
            if duration <= 0:
                if timestamp >= w_start:
                    continue
                return effects.clamp01(i / n)
            if timestamp < w_start:
                return effects.clamp01(i / n)
            if timestamp < w_end:
                return effects.clamp01((i + (timestamp - w_start) / duration) / n)
        return 1.0

    # -- vẽ text dùng chung (highlight fill trái -> phải, mục 4.4 plan) --

    def _draw_plain_line(
        self, img: Image.Image, xy: tuple[int, int], text: str, font, rgb: tuple[int, int, int]
    ) -> tuple[int, int]:
        """Vẽ 1 dòng text 1 màu duy nhất (dòng đã hát/chưa hát, không highlight)."""
        if not text:
            return (0, 0)
        ImageDraw.Draw(img).text(xy, text, font=font, fill=rgb)
        return effects.text_size(font, text)

    def _draw_highlighted_line(
        self,
        img: Image.Image,
        xy: tuple[int, int],
        text: str,
        font,
        unsung_rgb: tuple[int, int, int],
        sung_rgb: tuple[int, int, int],
        progress: float,
        words: list[str] | None = None,
    ) -> tuple[int, int]:
        """Vẽ dòng hiện tại với hiệu ứng highlight trái -> phải (kỹ thuật mask,
        mục 4.4 plan): vẽ nguyên dòng màu `unsung_rgb`, sau đó phủ phần đã hát
        (`sung_rgb`) qua mask hình chữ nhật rộng dần theo `progress`."""
        if not text:
            return (0, 0)
        x0, _y0 = xy
        line_w, line_h = effects.text_size(font, text)
        draw = ImageDraw.Draw(img)
        draw.text(xy, text, font=font, fill=unsung_rgb)

        if words:
            space_w = effects.text_size(font, " ")[0] or max(1, int(6 * self.scale))
            fill_w = effects.highlight_width(font, words, progress, space_w)
        else:
            fill_w = line_w * effects.clamp01(progress)

        if fill_w > 0:
            sung_layer = img.copy()
            ImageDraw.Draw(sung_layer).text(xy, text, font=font, fill=sung_rgb)
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).rectangle(
                [x0, 0, x0 + fill_w, img.size[1]], fill=255
            )
            img.paste(sung_layer, (0, 0), mask)
        return (line_w, line_h)
