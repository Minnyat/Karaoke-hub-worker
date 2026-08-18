"""Hàm thuần dùng để vẽ template: màu sắc, gradient, glow, progress bar. (PR-A4)

Không I/O ngoài Pillow/numpy; không phụ thuộc font Be Vietnam Pro thật (font
được nạp ở `templates/base.py`, hàm ở đây chỉ nhận `ImageFont` đã load sẵn).
"""

from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def clamp01(x: float) -> float:
    """Ép `x` về khoảng [0.0, 1.0]."""
    return max(0.0, min(1.0, float(x)))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Chuyển `'#RRGGBB'` -> `(r, g, b)` 0-255.

    Sai định dạng -> `ValueError` (fail fast; WebUI đã validate bằng
    `contracts/job.schema.json` pattern `^#[0-9A-Fa-f]{6}$`, xem REQ-A4-15).
    """
    if not isinstance(value, str) or not _HEX_RE.match(value):
        raise ValueError(f"Màu hex không hợp lệ: {value!r} (kỳ vọng dạng '#RRGGBB')")
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def with_alpha(rgb: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    """Thêm kênh alpha (0.0-1.0) vào màu RGB -> RGBA."""
    a = int(round(clamp01(alpha) * 255))
    return (rgb[0], rgb[1], rgb[2], a)


def lerp_color(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Nội suy tuyến tính giữa 2 màu RGB theo `t` (0-1, tự động clamp)."""
    t = clamp01(t)
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))  # type: ignore[return-value]


def linear_gradient(
    width: int, height: int, top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]
) -> Image.Image:
    """Gradient dọc: hàng 0 = `top_rgb`, hàng cuối = `bottom_rgb` (dùng cho classic).

    Vector hoá bằng numpy — 1 frame 4K vẫn tính trong vài chục ms, chỉ chạy 1
    lần rồi cache (xem `templates/base.py::_build_background`).
    """
    width = max(1, int(width))
    height = max(1, int(height))
    top = np.array(top_rgb, dtype=np.float64).reshape(1, 1, 3)
    bottom = np.array(bottom_rgb, dtype=np.float64).reshape(1, 1, 3)
    denom = max(1, height - 1)
    t = (np.arange(height, dtype=np.float64) / denom).reshape(height, 1, 1)
    row_colors = top + (bottom - top) * t
    arr = np.repeat(row_colors, width, axis=1)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def radial_overlay(
    img: Image.Image, center_rgb: tuple[int, int, int], strength: float = 0.5
) -> Image.Image:
    """Phủ overlay radial (đậm ở tâm, mờ dần ra biên) màu `center_rgb` lên `img`.

    Dùng cho nền template modern (base color + overlay). Trả `Image` mới,
    không mutate `img`.
    """
    width, height = img.size
    cx, cy = width / 2.0, height / 2.0
    max_dist = math.hypot(cx, cy) or 1.0
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.hypot(xx - cx, yy - cy) / max_dist
    alpha = np.clip((1.0 - dist) * clamp01(strength), 0.0, 1.0)[..., None]

    base = np.array(img.convert("RGB"), dtype=np.float64)
    overlay = np.array(center_rgb, dtype=np.float64).reshape(1, 1, 3)
    blended = base * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def text_size(font, text: str) -> tuple[int, int]:
    """Đo (width, height) của `text` với `font` — dùng `getbbox` (Pillow 10 đã
    bỏ `font.getsize()`)."""
    if not text:
        return (0, 0)
    bbox = font.getbbox(text)
    if bbox is None:
        return (0, 0)
    left, top, right, bottom = bbox
    return (max(0, right - left), max(0, bottom - top))


def highlight_width(font, words: list[str], progress: float, space_w: int) -> float:
    """Bề rộng phần "đã hát" của 1 dòng, tính theo mốc từng từ.

    `progress * len(words)` -> phần nguyên = số từ đã hát trọn vẹn, phần dư =
    tỉ lệ đã hát của từ đang hát. `progress=0` -> 0.0; `progress=1` -> tổng bề
    rộng cả dòng (kể cả khoảng trắng giữa từ). Đơn điệu không giảm theo
    `progress` — dùng cho hiệu ứng fill trái->phải (mục 4.4 plan).
    """
    if not words:
        return 0.0
    n = len(words)
    progress = clamp01(progress)

    widths = [text_size(font, w)[0] for w in words]
    cumulative: list[float] = []
    acc = 0.0
    for i, w in enumerate(widths):
        if i > 0:
            acc += space_w
        acc += w
        cumulative.append(acc)
    total_full = cumulative[-1] if cumulative else 0.0

    pos = progress * n
    idx = int(pos)
    if idx >= n:
        return total_full
    frac = pos - idx
    prev_cum = cumulative[idx - 1] if idx > 0 else 0.0
    space_before = space_w if idx > 0 else 0.0
    return prev_cum + space_before + widths[idx] * frac


def draw_glow_text(
    img: Image.Image, xy: tuple[int, int], text: str, font, rgb, radius: int, passes: int
) -> None:
    """Vẽ hiệu ứng glow (không vẽ chữ nét sắc) quanh `text` lên `img` (mutate
    in-place). Dùng nhiều lớp blur bán kính tăng dần, alpha giảm dần (neon).

    Chỉ blur vùng crop quanh text (không phải toàn canvas) — GaussianBlur trên
    canvas 1080p/4K đầy đủ mỗi frame sẽ rất chậm (mục 8 plan, tối ưu tốc độ).
    """
    if not text or radius <= 0:
        return
    x0, y0 = xy
    text_w, text_h = text_size(font, text)
    margin = int(radius * max(1, passes)) + 4
    box_x0 = max(0, int(x0) - margin)
    box_y0 = max(0, int(y0) - margin)
    box_x1 = min(img.width, int(x0) + text_w + margin)
    box_y1 = min(img.height, int(y0) + text_h + margin)
    box_w = max(1, box_x1 - box_x0)
    box_h = max(1, box_y1 - box_y0)

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (x0 - box_x0, y0 - box_y0), text, font=font, fill=with_alpha(rgb, 1.0)
    )

    result = img.crop((box_x0, box_y0, box_x1, box_y1)).convert("RGBA")
    passes = max(1, passes)
    for i in range(passes):
        blur_radius = radius * (i + 1) / passes
        alpha_factor = 1.0 - (i / passes) * 0.6
        blurred = layer.filter(ImageFilter.GaussianBlur(blur_radius))
        r, g, b, a = blurred.split()
        a = a.point(lambda v: int(v * alpha_factor))
        blurred = Image.merge("RGBA", (r, g, b, a))
        result = Image.alpha_composite(result, blurred)
    img.paste(result.convert("RGB"), (box_x0, box_y0))


def draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    progress: float,
    fg_rgb: tuple[int, int, int],
    bg_rgb: tuple[int, int, int],
    radius: int = 0,
) -> None:
    """Vẽ thanh tiến độ trong `box=(x0,y0,x1,y1)`: nền `bg_rgb`, phần đã trôi
    qua tô `fg_rgb` (trái -> phải theo `progress` 0-1)."""
    x0, y0, x1, y1 = box
    progress = clamp01(progress)
    fill_x1 = x0 + (x1 - x0) * progress
    if radius > 0:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_rgb)
        if fill_x1 > x0:
            draw.rounded_rectangle([x0, y0, fill_x1, y1], radius=radius, fill=fg_rgb)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=bg_rgb)
        if fill_x1 > x0:
            draw.rectangle([x0, y0, fill_x1, y1], fill=fg_rgb)


def format_timecode(seconds: float) -> str:
    """`83.4` -> `"1:23"`, `0` -> `"0:00"`, `245` -> `"4:05"` (M:SS, làm tròn xuống)."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"
