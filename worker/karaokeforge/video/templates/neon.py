"""Template Neon — spec PRD S3.4. (PR-A4)

Nền `#0a0a0a`; 3 dòng, viền neon quanh lyrics area; dòng hiện tại UPPERCASE;
animation theo `frame_idx` (glow, hue viền, flicker).

Animation PHẢI deterministic theo `frame_idx` (REQ-A4-14): dùng
`random.Random(frame_idx)`, KHÔNG dùng module `random` global — cùng
`(timestamp, lyrics, frame_idx)` phải luôn ra cùng byte output.
"""

from __future__ import annotations

import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageDraw

from karaokeforge.video import effects
from karaokeforge.video.templates.base import BaseTemplate

BG_COLOR = "#0a0a0a"
COLOR_UNSUNG = "#FF00FF"
COLOR_SINGING = "#00FF41"
COLOR_SUNG = "#333333"
LINE_COUNT = 3
PADDING_PX_1080 = 60
AREA_TOP_RATIO = 0.15
AREA_BOTTOM_RATIO = 0.85
GLOW_RADIUS_PX_1080 = 10
GLOW_PASSES = 3
BORDER_WIDTH_PX_1080 = 4
FLICKER_AMPLITUDE = 0.02


class NeonTemplate(BaseTemplate):
    """3 dòng, viền neon quanh lyrics area, dòng hiện tại UPPERCASE, animation
    theo frame_idx (deterministic)."""

    def _build_background(self) -> Image.Image:
        bg = self.config.get("background_color")
        color = effects.hex_to_rgb(bg) if bg else effects.hex_to_rgb(BG_COLOR)
        return Image.new("RGB", (self.width, self.height), color)

    def _display_text(self, text: str) -> str:
        """Chuẩn hoá text hiển thị cho dòng hiện tại: UPPERCASE (Python xử lý
        đúng dấu tiếng Việt, ví dụ `"Đường"` -> `"ĐƯỜNG"`)."""
        return text.upper()

    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        img = self._new_canvas()
        padding = int(PADDING_PX_1080 * self.scale)
        area_box = (
            padding,
            int(self.height * AREA_TOP_RATIO),
            self.width - padding,
            int(self.height * AREA_BOTTOM_RATIO),
        )
        self._draw_border(img, area_box, frame_idx)
        window, active = self._get_active_lines(timestamp, lyrics, LINE_COUNT)
        self._draw_lyrics(img, timestamp, window, active, area_box, frame_idx)
        self._draw_progress_bar(img, timestamp, lyrics)
        return self._to_array(img)

    def _draw_border(self, img: Image.Image, box: tuple[int, int, int, int], frame_idx: int) -> None:
        hue = (frame_idx * 0.5) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        rng = random.Random(frame_idx)
        flicker = 1.0 + (rng.random() * 2 - 1) * FLICKER_AMPLITUDE
        width = max(1, int(BORDER_WIDTH_PX_1080 * self.scale * flicker))
        x0, y0, x1, y1 = box
        if x1 > x0 and y1 > y0:
            ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), outline=rgb, width=width)

    def _draw_lyrics(
        self,
        img: Image.Image,
        timestamp: float,
        window: list[dict],
        active: int,
        area_box: tuple[int, int, int, int],
        frame_idx: int,
    ) -> None:
        if not window:
            return
        x0, y0, x1, y1 = area_box
        line_gap = (y1 - y0) / max(1, LINE_COUNT)

        singing_rgb = self._resolve_color("highlight_color", COLOR_SINGING)
        unsung_rgb = effects.hex_to_rgb(COLOR_UNSUNG)
        sung_rgb = effects.hex_to_rgb(COLOR_SUNG)
        glow = math.sin(frame_idx * 0.05) * 0.1 + 0.9

        for i, segment in enumerate(window):
            y = y0 + int(line_gap * i) + int(line_gap * 0.3)
            text = segment.get("text", "")
            if i == active:
                text = self._display_text(text)
                font = self.fonts["current"]
                words = [w.get("word", "").upper() for w in segment.get("words") or []]
                progress = self._get_word_highlight_progress(timestamp, segment)
                glow_rgb = effects.lerp_color((0, 0, 0), singing_rgb, effects.clamp01(glow))
                effects.draw_glow_text(
                    img,
                    (x0, y),
                    text,
                    font,
                    glow_rgb,
                    radius=int(GLOW_RADIUS_PX_1080 * self.scale),
                    passes=GLOW_PASSES,
                )
                self._draw_highlighted_line(
                    img, (x0, y), text, font, unsung_rgb, singing_rgb, progress, words=words or None
                )
            elif i < active:
                self._draw_plain_line(img, (x0, y), text, self.fonts["other"], sung_rgb)
            else:
                self._draw_plain_line(img, (x0, y), text, self.fonts["other"], unsung_rgb)

    def _draw_progress_bar(self, img: Image.Image, timestamp: float, lyrics: list[dict]) -> None:
        duration = self._duration(lyrics)
        progress = effects.clamp01(timestamp / duration) if duration > 0 else 0.0
        bar_h = max(2, int(4 * self.scale))
        y1 = max(bar_h, self.height - int(10 * self.scale))
        y0 = y1 - bar_h
        margin = int(self.width * 0.05)
        draw = ImageDraw.Draw(img)
        effects.draw_progress_bar(
            draw,
            (margin, y0, self.width - margin, y1),
            progress,
            effects.hex_to_rgb(COLOR_SINGING),
            effects.hex_to_rgb(COLOR_SUNG),
        )
