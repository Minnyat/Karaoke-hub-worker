"""Test video/templates: base helpers (logic thuần) + effects + 3 template + factory.

PR-A4. Test-first (CLAUDE.md Code Conventions): viết đỏ trước, không sửa assertion
để xanh. Tham chiếu ID test theo docs/plans/PR-A4.md mục 5.1.
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
import pytest
from PIL import ImageDraw

from karaokeforge.video import effects
from karaokeforge.video.templates.base import BaseTemplate

from .helpers_a4 import (
    extract_real_font,
    gapped_lyrics_fixture,
    lyrics_fixture,
    template_config,
)


class _DummyTemplate(BaseTemplate):
    """Subclass cụ thể tối thiểu — dùng để test helper của BaseTemplate độc lập
    với classic/modern/neon."""

    def render_frame(self, timestamp, lyrics, frame_idx):
        return self._to_array(self._new_canvas())


# ---------------------------------------------------------------------------
# A. _get_word_highlight_progress (REQ-A4-12)
# ---------------------------------------------------------------------------


def _contiguous_words():
    return [
        {"start": 0.0, "end": 1.0, "word": "a", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "word": "b", "confidence": 0.9},
        {"start": 2.0, "end": 3.0, "word": "c", "confidence": 0.9},
        {"start": 3.0, "end": 4.0, "word": "d", "confidence": 0.9},
    ]


@pytest.fixture
def tpl():
    return _DummyTemplate(320, 180, template_config())


def test_w01_no_words_key_linear(tpl):
    segment = {"start": 0.0, "end": 10.0, "text": "x"}
    assert tpl._get_word_highlight_progress(5.0, segment) == pytest.approx(0.5)


def test_w02_empty_words_linear(tpl):
    segment = {"start": 0.0, "end": 10.0, "text": "x", "words": []}
    assert tpl._get_word_highlight_progress(5.0, segment) == pytest.approx(0.5)


def test_w03_linear_end_equals_start(tpl):
    segment = {"start": 5.0, "end": 5.0, "text": "x", "words": []}
    assert tpl._get_word_highlight_progress(5.0, segment) == 0.0


def test_w04_linear_end_less_than_start(tpl):
    segment = {"start": 5.0, "end": 2.0, "text": "x", "words": []}
    assert tpl._get_word_highlight_progress(3.0, segment) == 0.0


def test_w05_before_first_word(tpl):
    segment = {"start": 0.0, "end": 4.0, "text": "x", "words": _contiguous_words()}
    assert tpl._get_word_highlight_progress(-0.5, segment) == 0.0


def test_w06_mid_second_of_four_words(tpl):
    segment = {"start": 0.0, "end": 4.0, "text": "x", "words": _contiguous_words()}
    assert tpl._get_word_highlight_progress(1.5, segment) == pytest.approx(0.375)


def test_w07_exact_word_start_boundary(tpl):
    segment = {"start": 0.0, "end": 4.0, "text": "x", "words": _contiguous_words()}
    assert tpl._get_word_highlight_progress(2.0, segment) == pytest.approx(0.5)


def test_w08_gap_between_word_2_and_3(tpl):
    words = [
        {"start": 0.0, "end": 1.0, "word": "a", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "word": "b", "confidence": 0.9},
        {"start": 2.5, "end": 3.5, "word": "c", "confidence": 0.9},
        {"start": 3.5, "end": 4.5, "word": "d", "confidence": 0.9},
    ]
    segment = {"start": 0.0, "end": 4.5, "text": "x", "words": words}
    assert tpl._get_word_highlight_progress(2.2, segment) == pytest.approx(0.5)


def test_w09_after_last_word(tpl):
    segment = {"start": 0.0, "end": 4.0, "text": "x", "words": _contiguous_words()}
    assert tpl._get_word_highlight_progress(10.0, segment) == 1.0


def test_w10_zero_duration_word_no_zero_division(tpl):
    words = [
        {"start": 0.0, "end": 1.0, "word": "a", "confidence": 0.9},
        {"start": 1.0, "end": 1.0, "word": "b", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "word": "c", "confidence": 0.9},
        {"start": 2.0, "end": 3.0, "word": "d", "confidence": 0.9},
    ]
    segment = {"start": 0.0, "end": 3.0, "text": "x", "words": words}
    result = tpl._get_word_highlight_progress(1.0, segment)
    assert result == pytest.approx(0.5)


def test_w11_all_zero_duration_words(tpl):
    words = [
        {"start": 0.0, "end": 0.0, "word": "a", "confidence": 0.9},
        {"start": 1.0, "end": 1.0, "word": "b", "confidence": 0.9},
        {"start": 2.0, "end": 2.0, "word": "c", "confidence": 0.9},
        {"start": 3.0, "end": 3.0, "word": "d", "confidence": 0.9},
    ]
    segment = {"start": 0.0, "end": 3.0, "text": "x", "words": words}
    assert tpl._get_word_highlight_progress(3.0, segment) == 1.0
    assert tpl._get_word_highlight_progress(100.0, segment) == 1.0


def test_w12_word_missing_end_key_fallback_linear(tpl):
    words = [{"start": 0.0, "word": "a", "confidence": 0.9}]
    segment = {"start": 0.0, "end": 2.0, "text": "x", "words": words}
    assert tpl._get_word_highlight_progress(1.0, segment) == pytest.approx(0.5)


def test_w13_clamp_extreme_timestamps(tpl):
    segment = {"start": 0.0, "end": 4.0, "text": "x", "words": _contiguous_words()}
    assert tpl._get_word_highlight_progress(-1000.0, segment) == 0.0
    assert tpl._get_word_highlight_progress(1000.0, segment) == 1.0


# ---------------------------------------------------------------------------
# B. _get_active_lines (REQ-A4-11)
# ---------------------------------------------------------------------------


def _five_lines():
    return [
        {"start": 0.0, "end": 2.0, "text": "s0", "words": []},
        {"start": 2.0, "end": 4.0, "text": "s1", "words": []},
        {"start": 4.5, "end": 6.5, "text": "s2", "words": []},
        {"start": 6.5, "end": 8.5, "text": "s3", "words": []},
        {"start": 8.5, "end": 10.5, "text": "s4", "words": []},
    ]


def test_l01_empty_lyrics(tpl):
    assert tpl._get_active_lines(0.0, [], count=3) == ([], -1)


def test_l02_before_first_segment(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(-1.0, lyrics, count=3)
    assert active == 0
    assert window[active]["text"] == "s0"


def test_l03_inside_first_segment(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(1.0, lyrics, count=3)
    assert [s["text"] for s in window] == ["s0", "s1", "s2"]
    assert active == 0


def test_l04_inside_middle_segment(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(5.0, lyrics, count=3)
    assert [s["text"] for s in window] == ["s1", "s2", "s3"]
    assert active == 1


def test_l05_inside_last_segment(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(9.5, lyrics, count=3)
    assert [s["text"] for s in window] == ["s2", "s3", "s4"]
    assert active == 2


def test_l06_after_last_segment(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(100.0, lyrics, count=3)
    assert len(window) == 3
    assert window[active]["text"] == "s4"


def test_l07_in_gap_between_segments(tpl):
    lyrics = gapped_lyrics_fixture()
    window, active = tpl._get_active_lines(4.2, lyrics, count=3)
    assert window[active]["text"] == "two"


def test_l08_single_line(tpl):
    lyrics = [{"start": 0.0, "end": 2.0, "text": "only", "words": []}]
    window, active = tpl._get_active_lines(1.0, lyrics, count=3)
    assert len(window) == 1
    assert active == 0


def test_l09_two_lines(tpl):
    lyrics = _five_lines()[:2]
    window, active = tpl._get_active_lines(3.0, lyrics, count=3)
    assert len(window) == 2
    assert 0 <= active < len(window)


def test_l10_count_one(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(5.0, lyrics, count=1)
    assert len(window) == 1
    assert active == 0
    assert window[0]["text"] == "s2"


def test_l11_invariant_window_bounds(tpl):
    lyrics = _five_lines()
    for count in (1, 2, 3, 4, 5, 6):
        for t in (-5.0, 0.5, 3.0, 4.2, 5.0, 9.5, 50.0):
            window, active = tpl._get_active_lines(t, lyrics, count=count)
            assert len(window) <= count
            assert 0 <= active < len(window)


def test_l12_window_identity_not_copy(tpl):
    lyrics = _five_lines()
    window, active = tpl._get_active_lines(5.0, lyrics, count=3)
    assert window[active] is lyrics[2]


def test_l13_segment_missing_start_key_no_keyerror(tpl):
    """PR-A4-FIX (Low, D4): file lyrics_edited.json từ WebUI có thể thiếu key
    `start`/`end` -> `_get_active_lines` không được KeyError, dùng .get(k, 0.0)."""
    lyrics = [
        {"end": 2.0, "text": "a", "words": []},  # thiếu "start"
        {"start": 2.0, "end": 4.0, "text": "b", "words": []},
    ]
    window, active = tpl._get_active_lines(1.0, lyrics, count=2)
    assert 0 <= active < len(window)


# ---------------------------------------------------------------------------
# C. _load_fonts (REQ-A4-09/10)
# ---------------------------------------------------------------------------


def test_f01_missing_font_dir_no_raise(tmp_path):
    config = template_config(font_dir=str(tmp_path / "empty"))
    t = _DummyTemplate(320, 180, config)
    assert set(t.fonts) == {"current", "other", "small"}
    assert t.font_fallback_used is True


def test_f02_fallback_font_usable(tmp_path):
    config = template_config(font_dir=str(tmp_path / "empty"))
    t = _DummyTemplate(320, 180, config)
    img = t._new_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "Hello", font=t.fonts["current"])


def test_f03_scale_by_resolution(tmp_path):
    config = template_config(font_dir=str(tmp_path / "empty"))
    t_1080 = _DummyTemplate(1920, 1080, config)
    t_2160 = _DummyTemplate(3840, 2160, config)
    assert t_2160.fonts["current"].size == pytest.approx(2 * t_1080.fonts["current"].size)


def test_f04_small_resolution_min_size(tmp_path):
    config = template_config(font_dir=str(tmp_path / "empty"))
    t = _DummyTemplate(320, 180, config)
    for font in t.fonts.values():
        assert font.size >= 8


def test_f05_warning_logged_on_fallback(tmp_path, caplog):
    config = template_config(font_dir=str(tmp_path / "empty"))
    with caplog.at_level(logging.WARNING):
        _DummyTemplate(320, 180, config)
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_f06_real_font_no_fallback(tmp_path):
    font_path = extract_real_font(tmp_path)
    config = template_config(font_dir=str(tmp_path), font="test_real_font.ttf")
    t = _DummyTemplate(320, 180, config)
    assert t.font_fallback_used is False
    assert t.fonts["current"].path == font_path


def test_f07_family_name_maps_to_weight_files_no_fallback(tmp_path):
    """PR-A4-FIX (Critical): `font="Be Vietnam Pro"` (tên family, không có đuôi)
    phải được map sang file thật `BeVietnamPro-Bold.ttf` (dòng hiện tại) /
    `BeVietnamPro-Regular.ttf` (dòng phụ), KHÔNG ghép thẳng "Be Vietnam Pro"
    thành path -> KHÔNG fallback DejaVu khi file tồn tại."""
    from PIL import ImageFont

    font_bytes = ImageFont.load_default(size=40).font_bytes
    (tmp_path / "BeVietnamPro-Bold.ttf").write_bytes(font_bytes)
    (tmp_path / "BeVietnamPro-Regular.ttf").write_bytes(font_bytes)

    config = template_config(font_dir=str(tmp_path), font="Be Vietnam Pro")
    t = _DummyTemplate(320, 180, config)

    assert t.font_fallback_used is False
    assert t.fonts["current"].path.replace("\\", "/").endswith("BeVietnamPro-Bold.ttf")
    assert t.fonts["other"].path.replace("\\", "/").endswith("BeVietnamPro-Regular.ttf")
    assert t.fonts["small"].path.replace("\\", "/").endswith("BeVietnamPro-Regular.ttf")
    assert "Be Vietnam Pro" not in t.fonts["current"].path


def test_f08_family_name_path_never_contains_raw_family(tmp_path, monkeypatch):
    """PR-A4-FIX: đường dẫn primary truyền vào `ImageFont.truetype` cho font
    family phải kết thúc bằng `-Bold.ttf`/`-Regular.ttf`, tuyệt đối không phải
    tên family thô 'Be Vietnam Pro'. Chặn OSError để độc lập font hệ thống."""
    import os

    import karaokeforge.video.templates.base as base_mod

    captured: list[str] = []
    real_truetype = base_mod.ImageFont.truetype

    def spy_truetype(path, size, *args, **kwargs):
        # Chỉ chặn load từ đường dẫn file (primary + fallback DejaVu); để
        # load_default (truyền BytesIO) chạy thật để không raise ra ngoài.
        if isinstance(path, (str, os.PathLike)):
            captured.append(str(path))
            raise OSError("skip real file load in test")
        return real_truetype(path, size, *args, **kwargs)

    monkeypatch.setattr(base_mod.ImageFont, "truetype", spy_truetype)

    font_dir = str(tmp_path / "custom_fonts")
    config = template_config(font_dir=font_dir, font="Be Vietnam Pro")
    _DummyTemplate(320, 180, config)

    primary = [p for p in captured if font_dir.replace("\\", "/") in p.replace("\\", "/")]
    assert primary, "phải thử load font primary trong font_dir"
    for p in primary:
        norm = p.replace("\\", "/")
        assert norm.endswith("-Bold.ttf") or norm.endswith("-Regular.ttf")
        assert "Be Vietnam Pro" not in p


# ---------------------------------------------------------------------------
# D. effects.py
# ---------------------------------------------------------------------------


def test_e01_hex_to_rgb():
    assert effects.hex_to_rgb("#FFD700") == (255, 215, 0)


def test_e02_hex_to_rgb_invalid():
    for bad in ("FFD700", "#GGG", "", "#12345"):
        with pytest.raises(ValueError):
            effects.hex_to_rgb(bad)


def test_e03_format_timecode():
    assert effects.format_timecode(83.4) == "1:23"
    assert effects.format_timecode(0) == "0:00"
    assert effects.format_timecode(245) == "4:05"


def test_e04_linear_gradient_top_bottom():
    img = effects.linear_gradient(10, 20, (0, 0, 0), (255, 255, 255))
    arr = np.array(img)
    assert np.allclose(arr[0, 0], [0, 0, 0], atol=8)
    assert np.allclose(arr[-1, 0], [255, 255, 255], atol=8)


def test_e05_draw_progress_bar_pixel():
    from PIL import Image

    img = Image.new("RGB", (100, 10))
    draw = ImageDraw.Draw(img)
    fg = (255, 215, 0)
    bg = (51, 51, 51)
    effects.draw_progress_bar(draw, (0, 0, 100, 10), 0.5, fg, bg)
    arr = np.array(img)
    assert tuple(arr[5, 25]) == fg
    assert tuple(arr[5, 75]) == bg


def test_e06_highlight_width_bounds(tmp_path):
    from PIL import ImageFont

    font_path = extract_real_font(tmp_path)
    font = ImageFont.truetype(font_path, 20)
    words = ["hello", "world", "foo"]
    space_w = 5
    assert effects.highlight_width(font, words, 0.0, space_w) == 0.0
    full = effects.highlight_width(font, words, 1.0, space_w)
    expected_total = sum(effects.text_size(font, w)[0] for w in words) + space_w * (len(words) - 1)
    assert full == pytest.approx(expected_total)


def test_e07_highlight_width_monotonic(tmp_path):
    from PIL import ImageFont

    font_path = extract_real_font(tmp_path)
    font = ImageFont.truetype(font_path, 20)
    words = ["hello", "world", "foo", "bar"]
    prev = -1.0
    for i in range(0, 101, 5):
        progress = i / 100
        width = effects.highlight_width(font, words, progress, 5)
        assert width >= prev - 1e-9
        prev = width


def test_e08_clamp01():
    assert effects.clamp01(-1) == 0.0
    assert effects.clamp01(2) == 1.0
    assert effects.clamp01(0.5) == 0.5


# ---------------------------------------------------------------------------
# E. 3 template + factory (REQ-A4-13/14/15/16)
# ---------------------------------------------------------------------------

from karaokeforge.video.templates import get_template  # noqa: E402
from karaokeforge.video.templates.classic import ClassicTemplate  # noqa: E402
from karaokeforge.video.templates.modern import ModernTemplate  # noqa: E402
from karaokeforge.video.templates.neon import NeonTemplate  # noqa: E402

WIDTH, HEIGHT = 640, 360


def _closest_pixel_distance(arr: np.ndarray, target_rgb, y_slice, x_slice=None):
    region = arr[y_slice, x_slice] if x_slice is not None else arr[y_slice]
    diff = region.astype(np.int32) - np.array(target_rgb, dtype=np.int32)
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist.min()


@pytest.mark.parametrize("name,cls", [("classic", ClassicTemplate), ("modern", ModernTemplate), ("neon", NeonTemplate)])
def test_t01_get_template_returns_correct_class(name, cls):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    assert isinstance(t, cls)
    assert isinstance(t, BaseTemplate)


def test_t02_get_template_case_insensitive():
    t = get_template("CLASSIC", WIDTH, HEIGHT, template_config())
    assert isinstance(t, ClassicTemplate)


def test_t03_get_template_invalid_name():
    with pytest.raises(ValueError, match="classic"):
        get_template("banana", WIDTH, HEIGHT, template_config())


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t04_render_frame_shape_dtype(name):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    frame = t.render_frame(1.0, lyrics_fixture(), 30)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t05_empty_lyrics_no_raise(name):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    frame = t.render_frame(1.0, [], 0)
    assert frame.shape == (HEIGHT, WIDTH, 3)


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t06_timestamp_after_last_line_no_raise(name):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    frame = t.render_frame(9999.0, lyrics_fixture(), 30)
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_t07_classic_gradient_background():
    t = ClassicTemplate(WIDTH, HEIGHT, template_config())
    frame = t.render_frame(0.0, [], 0)
    assert np.allclose(frame[0, 0], effects.hex_to_rgb("#000428"), atol=8)
    assert np.allclose(frame[-1, 0], effects.hex_to_rgb("#004e92"), atol=8)


def test_t08_modern_background():
    t = ModernTemplate(WIDTH, HEIGHT, template_config())
    frame = t.render_frame(0.0, [], 0)
    assert np.allclose(frame[2, 2], effects.hex_to_rgb("#0f0f23"), atol=8)


def test_t09_neon_background():
    t = NeonTemplate(WIDTH, HEIGHT, template_config())
    frame = t.render_frame(0.0, [], 0)
    assert np.allclose(frame[2, 2], effects.hex_to_rgb("#0a0a0a"), atol=8)


@pytest.mark.parametrize(
    "name,highlight_hex",
    [("classic", "#FFD700"), ("modern", "#00D4FF"), ("neon", "#00FF41")],
)
def test_t10_active_line_has_highlight_and_unsung_pixels(name, highlight_hex):
    tmp_dir = "/nonexistent/fonts/dir"
    t = get_template(name, WIDTH, HEIGHT, template_config(font_dir=tmp_dir))
    lyrics = lyrics_fixture()
    frame = t.render_frame(0.5, lyrics, 15)  # giữa dòng 0 ("Hello world")
    highlight_rgb = effects.hex_to_rgb(highlight_hex)
    dist = _closest_pixel_distance(frame, highlight_rgb, slice(0, HEIGHT))
    assert dist < 40, f"{name}: không tìm thấy pixel gần màu highlight {highlight_hex}"


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t11_highlight_moves_left_to_right(name):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    lyrics = lyrics_fixture()
    highlight_hex = {"classic": "#FFD700", "modern": "#00D4FF", "neon": "#00FF41"}[name]
    highlight_rgb = np.array(effects.hex_to_rgb(highlight_hex))

    def mean_x(ts):
        frame = t.render_frame(ts, lyrics, int(ts * 30))
        diff = frame.astype(np.int32) - highlight_rgb.astype(np.int32)
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        ys, xs = np.where(dist < 30)
        if len(xs) == 0:
            return None
        return xs.mean()

    x_early = mean_x(0.1)
    x_late = mean_x(1.8)
    assert x_early is not None and x_late is not None
    assert x_late > x_early


def test_t12_highlight_color_override():
    config = template_config(highlight_color="#FF0000")
    t = ClassicTemplate(WIDTH, HEIGHT, config)
    lyrics = lyrics_fixture()
    frame = t.render_frame(0.5, lyrics, 15)
    dist_red = _closest_pixel_distance(frame, (255, 0, 0), slice(0, HEIGHT))
    assert dist_red < 40


def test_t13_background_color_override():
    config = template_config(background_color="#123456")
    t = ClassicTemplate(WIDTH, HEIGHT, config)
    frame = t.render_frame(0.0, [], 0)
    assert np.allclose(frame[0, 0], effects.hex_to_rgb("#123456"), atol=4)


def test_t14_neon_determinism():
    t = NeonTemplate(WIDTH, HEIGHT, template_config())
    lyrics = lyrics_fixture()
    frame1 = t.render_frame(1.0, lyrics, 42)
    frame2 = t.render_frame(1.0, lyrics, 42)
    assert np.array_equal(frame1, frame2)


def test_t15_neon_frame_idx_changes_output():
    t = NeonTemplate(WIDTH, HEIGHT, template_config())
    lyrics = lyrics_fixture()
    frame0 = t.render_frame(1.0, lyrics, 0)
    frame37 = t.render_frame(1.0, lyrics, 37)
    assert not np.array_equal(frame0, frame37)


def test_t16_neon_uppercase_vietnamese():
    t = NeonTemplate(WIDTH, HEIGHT, template_config())
    assert t._display_text("Đường xa") == "ĐƯỜNG XA"


def test_t17_modern_draws_only_two_lines():
    t = ModernTemplate(WIDTH, HEIGHT, template_config())
    lyrics = lyrics_fixture()
    frame = t.render_frame(0.5, lyrics, 15)
    bg_top_left = frame[2, 2].astype(np.int32)
    top_region = frame[: int(HEIGHT * 0.5)]
    diff = top_region.astype(np.int32) - bg_top_left
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    assert dist.max() < 40


def test_t18_classic_neon_draw_three_lines():
    for cls in (ClassicTemplate, NeonTemplate):
        t = cls(WIDTH, HEIGHT, template_config())
        window, active = t._get_active_lines(0.5, lyrics_fixture(), 3)
        assert len(window) == 3


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t19_missing_font_still_renders(name, tmp_path):
    config = template_config(font_dir=str(tmp_path / "no_fonts_here"))
    t = get_template(name, WIDTH, HEIGHT, config)
    frame = t.render_frame(1.0, lyrics_fixture(), 30)
    assert frame.shape == (HEIGHT, WIDTH, 3)


@pytest.mark.parametrize("name", ["classic", "modern", "neon"])
def test_t20_render_frame_does_not_mutate_lyrics(name):
    t = get_template(name, WIDTH, HEIGHT, template_config())
    lyrics = lyrics_fixture()
    before = copy.deepcopy(lyrics)
    t.render_frame(1.5, lyrics, 45)
    assert lyrics == before
