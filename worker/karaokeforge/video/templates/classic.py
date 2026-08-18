"""Template Classic — spec PRD S3.2. (PR-A4)

Gradient nền `#000428 -> #004e92`; 3 dòng lyrics canh giữa (60% chiều cao);
highlight từ trái sang phải theo từ.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from karaokeforge.video import effects
from karaokeforge.video.templates.base import BaseTemplate

BG_TOP = "#000428"
BG_BOTTOM = "#004e92"
COLOR_UNSUNG = "#FFFFFF"
COLOR_SINGING = "#FFD700"
COLOR_SUNG = "#888888"
PROGRESS_FG = "#FFD700"
PROGRESS_BG = "#333333"
LINE_COUNT = 3
PADDING_RATIO = 80 / 1080
AREA_TOP_RATIO = 0.2
AREA_HEIGHT_RATIO = 0.6


class ClassicTemplate(BaseTemplate):
    """3 dòng lời canh giữa trên gradient nền, highlight trái -> phải theo từ."""

    def _build_background(self) -> Image.Image:
        bg = self.config.get("background_color")
        if bg:
            return Image.new("RGB", (self.width, self.height), effects.hex_to_rgb(bg))
        top = effects.hex_to_rgb(BG_TOP)
        bottom = effects.hex_to_rgb(BG_BOTTOM)
        return effects.linear_gradient(self.width, self.height, top, bottom)

    def render_frame(self, timestamp: float, lyrics: list[dict], frame_idx: int) -> np.ndarray:
        img = self._new_canvas()
        window, active = self._get_active_lines(timestamp, lyrics, LINE_COUNT)
        self._draw_lyrics(img, timestamp, window, active)
        self._draw_progress_bar(img, timestamp, lyrics)
        return self._to_array(img)

    def _draw_lyrics(self, img: Image.Image, timestamp: float, window: list[dict], active: int) -> None:
        if not window:
            return
        padding = int(PADDING_RATIO * self.height)
        area_top = int(self.height * AREA_TOP_RATIO)
        area_height = int(self.height * AREA_HEIGHT_RATIO)
        line_gap = area_height / max(1, LINE_COUNT)

        singing_rgb = self._resolve_color("highlight_color", COLOR_SINGING)
        unsung_rgb = effects.hex_to_rgb(COLOR_UNSUNG)
        sung_rgb = effects.hex_to_rgb(COLOR_SUNG)

        for i, segment in enumerate(window):
            y = area_top + int(line_gap * i) + int(line_gap * 0.3)
            text = segment.get("text", "")
            x = padding
            if i == active:
                words = [w.get("word", "") for w in segment.get("words") or []]
                progress = self._get_word_highlight_progress(timestamp, segment)
                self._draw_highlighted_line(
                    img,
                    (x, y),
                    text,
                    self.fonts["current"],
                    unsung_rgb,
                    singing_rgb,
                    progress,
                    words=words or None,
                )
            elif i < active:
                self._draw_plain_line(img, (x, y), text, self.fonts["other"], sung_rgb)
            else:
                self._draw_plain_line(img, (x, y), text, self.fonts["other"], unsung_rgb)

    def _draw_progress_bar(self, img: Image.Image, timestamp: float, lyrics: list[dict]) -> None:
        duration = self._duration(lyrics)
        progress = effects.clamp01(timestamp / duration) if duration > 0 else 0.0
        bar_h = max(2, int(self.height * 0.008))
        y1 = int(self.height * 0.9)
        y0 = y1 - bar_h
        margin = int(self.width * 0.05)
        draw = ImageDraw.Draw(img)
        effects.draw_progress_bar(
            draw,
            (margin, y0, self.width - margin, y1),
            progress,
            effects.hex_to_rgb(PROGRESS_FG),
            effects.hex_to_rgb(PROGRESS_BG),
        )
