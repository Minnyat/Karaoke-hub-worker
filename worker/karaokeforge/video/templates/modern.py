"""Template Modern — spec PRD S3.3. (PR-A4)

Nền `#0f0f23` + overlay radial `#1a1a3e`; 2 dòng = hiện tại (lớn) + kế tiếp
(nhỏ, mờ); highlight đổi màu + underline sweep.

Lưu ý: dùng `_get_active_lines(count=3)` rồi cắt `window[active:active+2]` để
lấy (dòng hiện tại, dòng kế tiếp). KHÔNG gọi `count=2` trực tiếp — với
`count=2`, thuật toán trả về (dòng trước, dòng hiện tại) khi `current_idx > 0`,
sai spec S3.3 (xem PR-A4 plan mục 4.6 / bẫy #11).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from karaokeforge.video import effects
from karaokeforge.video.templates.base import BaseTemplate

BG_BASE = "#0f0f23"
BG_OVERLAY = "#1a1a3e"
OVERLAY_STRENGTH = 0.6
COLOR_UNSUNG = "#E0E0E0"
COLOR_SINGING = "#00D4FF"
COLOR_SUNG = "#4A4A6A"
NEXT_LINE_DIM = 0.5
LOOKAHEAD_COUNT = 3
PADDING_PX_1080 = 100
AREA_TOP_RATIO = 0.6
AREA_HEIGHT_RATIO = 0.4


class ModernTemplate(BaseTemplate):
    """2 dòng: hiện tại (lớn) + kế tiếp (nhỏ, mờ). Highlight đổi màu + underline."""

    def _build_background(self) -> Image.Image:
        bg = self.config.get("background_color")
        if bg:
            return Image.new("RGB", (self.width, self.height), effects.hex_to_rgb(bg))
        base = Image.new("RGB", (self.width, self.height), effects.hex_to_rgb(BG_BASE))
        return effects.radial_overlay(base, effects.hex_to_rgb(BG_OVERLAY), strength=OVERLAY_STRENGTH)

    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        img = self._new_canvas()
        window, active = self._get_active_lines(timestamp, lyrics, LOOKAHEAD_COUNT)
        lines = window[active : active + 2] if active >= 0 else []
        self._draw_lyrics(img, timestamp, lines)
        self._draw_progress_bar(img, timestamp, lyrics)
        return self._to_array(img)

    def _bg_sample_color(self) -> tuple[int, int, int]:
        bg = self.config.get("background_color")
        if bg:
            return effects.hex_to_rgb(bg)
        return effects.hex_to_rgb(BG_BASE)

    def _draw_lyrics(self, img: Image.Image, timestamp: float, lines: list[dict]) -> None:
        if not lines:
            return
        padding = int(PADDING_PX_1080 * self.scale)
        area_top = int(self.height * AREA_TOP_RATIO)
        area_height = int(self.height * AREA_HEIGHT_RATIO)

        singing_rgb = self._resolve_color("highlight_color", COLOR_SINGING)
        unsung_rgb = effects.hex_to_rgb(COLOR_UNSUNG)

        current = lines[0]
        current_y = area_top + int(area_height * 0.2)
        words = [w.get("word", "") for w in current.get("words") or []]
        progress = self._get_word_highlight_progress(timestamp, current)
        line_w, line_h = self._draw_highlighted_line(
            img,
            (padding, current_y),
            current.get("text", ""),
            self.fonts["current"],
            unsung_rgb,
            singing_rgb,
            progress,
            words=words or None,
        )
        self._draw_underline_sweep(img, (padding, current_y + line_h), line_w, progress, singing_rgb)

        if len(lines) > 1:
            next_seg = lines[1]
            next_y = area_top + int(area_height * 0.65)
            next_rgb = effects.lerp_color(self._bg_sample_color(), unsung_rgb, NEXT_LINE_DIM)
            self._draw_plain_line(img, (padding, next_y), next_seg.get("text", ""), self.fonts["small"], next_rgb)

    def _draw_underline_sweep(
        self, img: Image.Image, xy: tuple[int, int], line_w: float, progress: float, rgb: tuple[int, int, int]
    ) -> None:
        x0, y0 = xy
        y0 = min(self.height - 1, y0 + int(4 * self.scale))
        x1 = x0 + line_w * effects.clamp01(progress)
        if x1 > x0:
            draw = ImageDraw.Draw(img)
            draw.line([(x0, y0), (x1, y0)], fill=rgb, width=max(1, int(2 * self.scale)))

    def _draw_progress_bar(self, img: Image.Image, timestamp: float, lyrics: list[dict]) -> None:
        duration = self._duration(lyrics)
        progress = effects.clamp01(timestamp / duration) if duration > 0 else 0.0
        bar_h = max(1, int(2 * self.scale))
        y1 = self.height
        y0 = y1 - bar_h
        draw = ImageDraw.Draw(img)
        effects.draw_progress_bar(
            draw,
            (0, y0, self.width, y1),
            progress,
            effects.hex_to_rgb(COLOR_SINGING),
            effects.hex_to_rgb(COLOR_SUNG),
        )
