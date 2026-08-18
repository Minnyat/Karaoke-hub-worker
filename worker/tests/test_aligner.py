"""Test thuần cho karaokeforge.pipeline.aligner — KHÔNG mock.

Bảng REQ: xem docs/plans/PR-A2A3.md §5.1 (A-01..A-12) và §5.2 (A-20..A-37).
"""
from __future__ import annotations

import copy
import logging
import unicodedata

from tests.helpers_a2a3 import assert_valid_lyrics, make_segment, make_word

from karaokeforge.pipeline.aligner import (
    apply_user_lyrics,
    interpolate_missing_word_timings,
)


# ---------------------------------------------------------------------------
# apply_user_lyrics (A-01..A-12)
# ---------------------------------------------------------------------------


def test_a01_replace_text_keep_timing():
    segments = [
        make_segment(0.0, 1.0, "old1", [make_word(0.0, 1.0, "old1")]),
        make_segment(1.0, 2.0, "old2", [make_word(1.0, 2.0, "old2")]),
        make_segment(2.0, 3.0, "old3", [make_word(2.0, 3.0, "old3")]),
    ]
    out = apply_user_lyrics(segments, "line1\nline2\nline3")
    assert [s["text"] for s in out] == ["line1", "line2", "line3"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0
    assert out[1]["start"] == 1.0 and out[1]["end"] == 2.0
    assert out[2]["start"] == 2.0 and out[2]["end"] == 3.0


def test_a02_does_not_mutate_input():
    segments = [make_segment(0.0, 1.0, "old", [make_word(0.0, 1.0, "old")])]
    snapshot = copy.deepcopy(segments)
    apply_user_lyrics(segments, "new line")
    assert segments == snapshot


def test_a03_more_lines_than_segments_glue_to_last():
    segments = [
        make_segment(0.0, 1.0, "a"),
        make_segment(1.0, 2.0, "b"),
    ]
    out = apply_user_lyrics(segments, "dong1\ndong2\ndong3\ndong4\ndong5")
    assert len(out) == 2
    assert out[0]["text"] == "dong1"
    assert out[1]["text"] == "dong2 dong3 dong4 dong5"


def test_a04_fewer_lines_than_segments_truncate():
    segments = [
        make_segment(0.0, 1.0, "a"),
        make_segment(1.0, 2.0, "b"),
        make_segment(2.0, 3.0, "c"),
        make_segment(3.0, 4.0, "d"),
        make_segment(4.0, 5.0, "e"),
    ]
    out = apply_user_lyrics(segments, "dong1\ndong2")
    assert len(out) == 2
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0
    assert out[1]["start"] == 1.0 and out[1]["end"] == 2.0
    assert out[0]["text"] == "dong1"
    assert out[1]["text"] == "dong2"


def test_a05_mismatch_logs_warning_with_both_counts(caplog):
    segments = [make_segment(0.0, 1.0, "a"), make_segment(1.0, 2.0, "b")]
    with caplog.at_level(logging.WARNING):
        apply_user_lyrics(segments, "dong1\ndong2\ndong3")
    assert "3" in caplog.text
    assert "2" in caplog.text


def test_a06_blank_or_empty_user_lyrics_returns_copy_unchanged():
    segments = [make_segment(0.0, 1.0, "original")]
    out1 = apply_user_lyrics(segments, "")
    out2 = apply_user_lyrics(segments, "   \n\n ")
    assert out1[0]["text"] == "original"
    assert out2[0]["text"] == "original"


def test_a07_empty_segments_returns_empty():
    assert apply_user_lyrics([], "dong1\ndong2") == []
    assert apply_user_lyrics([], "") == []


def test_a08_normalizes_crlf_blank_lines_and_padding():
    segments = [make_segment(0.0, 1.0, "a"), make_segment(1.0, 2.0, "b")]
    user_lyrics = "  dong1  \r\n\r\n  dong2  \r\n"
    out = apply_user_lyrics(segments, user_lyrics)
    assert out[0]["text"] == "dong1"
    assert out[1]["text"] == "dong2"


def test_a09_collapses_internal_whitespace():
    segments = [make_segment(0.0, 1.0, "a")]
    out = apply_user_lyrics(segments, "anh    yêu")
    assert out[0]["text"] == "anh yêu"


def test_a10_nfd_input_becomes_nfc_output():
    nfd_line = unicodedata.normalize("NFD", "người yêu")
    segments = [make_segment(0.0, 1.0, "a")]
    out = apply_user_lyrics(segments, nfd_line)
    assert unicodedata.normalize("NFC", out[0]["text"]) == out[0]["text"]
    assert out[0]["text"] == "người yêu"


def test_a11_clears_old_words_after_text_replace():
    segments = [
        make_segment(0.0, 1.0, "old", [make_word(0.0, 0.5, "old"), make_word(0.5, 1.0, "x")]),
    ]
    out = apply_user_lyrics(segments, "new line")
    assert out[0]["words"] == []


def test_a12_punctuation_only_line_is_valid():
    segments = [make_segment(0.0, 1.0, "a"), make_segment(1.0, 2.0, "b")]
    out = apply_user_lyrics(segments, "dong1\n...")
    assert len(out) == 2
    assert out[1]["text"] == "..."


# ---------------------------------------------------------------------------
# interpolate_missing_word_timings (A-20..A-37)
# ---------------------------------------------------------------------------


def _assert_monotonic(words):
    for i in range(len(words) - 1):
        assert words[i]["end"] <= words[i + 1]["start"]


def test_a20_no_words_key_generates_evenly_from_text():
    segments = [{"start": 0.0, "end": 4.0, "text": "a b c d"}]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert len(words) == 4
    bounds = [words[0]["start"]] + [w["end"] for w in words]
    assert bounds == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(w["confidence"] == 0.0 for w in words)


def test_a21_empty_words_list_with_text_generates_evenly():
    segments = [make_segment(0.0, 2.0, "a b", [])]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert len(words) == 2
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == 1.0
    assert words[1]["start"] == 1.0
    assert words[1]["end"] == 2.0


def test_a22_middle_words_missing_interpolated_between_known_neighbors():
    segments = [
        make_segment(
            0.0,
            4.0,
            "a b c d",
            [
                make_word(0.0, 1.0, "a"),
                {"word": "b"},
                {"word": "c"},
                make_word(3.0, 4.0, "d"),
            ],
        ),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[0]["start"] == 0.0 and words[0]["end"] == 1.0
    assert words[1]["start"] == 1.0 and words[1]["end"] == 2.0
    assert words[2]["start"] == 2.0 and words[2]["end"] == 3.0
    assert words[3]["start"] == 3.0 and words[3]["end"] == 4.0
    _assert_monotonic(words)


def test_a23_first_word_missing_anchors_segment_start():
    segments = [
        make_segment(0.0, 2.0, "a b", [{"word": "a"}, make_word(1.0, 2.0, "b")]),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == 1.0


def test_a24_last_word_missing_anchors_segment_end():
    segments = [
        make_segment(0.0, 2.0, "a b", [make_word(0.0, 1.0, "a"), {"word": "b"}]),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[1]["start"] == 1.0
    assert words[1]["end"] == 2.0


def test_a25_all_words_missing_evenly_split():
    segments = [
        make_segment(0.0, 4.0, "a b c d", [{"word": "a"}, {"word": "b"}, {"word": "c"}, {"word": "d"}]),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    bounds = [words[0]["start"]] + [w["end"] for w in words]
    assert bounds == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a26_fully_known_words_unchanged_except_round():
    segments = [
        make_segment(
            0.0,
            2.0,
            "a b",
            [make_word(0.0, 0.9999999, "a", 0.5), make_word(1.0, 2.0, "b", 0.6)],
        ),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == round(0.9999999, 3)
    assert words[0]["word"] == "a"
    assert words[1]["start"] == 1.0
    assert words[1]["end"] == 2.0


def test_a27_start_present_end_missing_treated_as_missing():
    segments = [
        make_segment(
            0.0,
            2.0,
            "a b",
            [{"start": 0.0, "word": "a"}, make_word(1.5, 2.0, "b")],
        ),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == 1.5


def test_a28_monotonic_across_all_cases():
    cases = [
        [{"start": 0.0, "end": 4.0, "text": "a b c d"}],
        [make_segment(0.0, 2.0, "a b", [])],
        [
            make_segment(
                0.0, 4.0, "a b c d",
                [make_word(0.0, 1.0, "a"), {"word": "b"}, {"word": "c"}, make_word(3.0, 4.0, "d")],
            ),
        ],
        [make_segment(0.0, 2.0, "a b", [{"word": "a"}, make_word(1.0, 2.0, "b")])],
        [make_segment(0.0, 2.0, "a b", [make_word(0.0, 1.0, "a"), {"word": "b"}])],
    ]
    for segments in cases:
        out = interpolate_missing_word_timings(segments)
        _assert_monotonic(out[0]["words"])


def test_a29_zero_duration_segment_no_zero_division():
    segments = [
        make_segment(1.0, 1.0, "a b", [{"word": "a"}, {"word": "b"}]),
    ]
    out = interpolate_missing_word_timings(segments)
    for w in out[0]["words"]:
        assert w["start"] == 1.0
        assert w["end"] == 1.0


def test_a30_negative_duration_segment_treated_like_zero():
    segments = [
        make_segment(2.0, 1.0, "a b", [{"word": "a"}, {"word": "b"}]),
    ]
    out = interpolate_missing_word_timings(segments)
    for w in out[0]["words"]:
        assert w["start"] == 2.0
        assert w["end"] == 2.0


def test_a31_empty_segments_list():
    assert interpolate_missing_word_timings([]) == []


def test_a32_empty_text_and_words_no_crash():
    segments = [make_segment(0.0, 1.0, "", [])]
    out = interpolate_missing_word_timings(segments)
    assert out[0]["words"] == []


def test_a33_does_not_mutate_input():
    segments = [
        make_segment(0.0, 2.0, "a b", [{"word": "a"}, make_word(1.0, 2.0, "b")]),
    ]
    snapshot = copy.deepcopy(segments)
    interpolate_missing_word_timings(segments)
    assert segments == snapshot


def test_a34_word_keys_exact_and_types():
    segments = [{"start": 0.0, "end": 2.0, "text": "a b"}]
    out = interpolate_missing_word_timings(segments)
    for w in out[0]["words"]:
        assert set(w.keys()) == {"start", "end", "word", "confidence"}
        assert isinstance(w["start"], float)
        assert isinstance(w["end"], float)


def test_a35_confidence_clamped():
    segments = [
        make_segment(
            0.0,
            3.0,
            "a b c",
            [
                {"word": "a", "start": 0.0, "end": 1.0},
                {"word": "b", "start": 1.0, "end": 2.0, "confidence": 1.7},
                {"word": "c", "start": 2.0, "end": 3.0, "confidence": -0.2},
            ],
        ),
    ]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert words[0]["confidence"] == 0.0
    assert words[1]["confidence"] == 1.0
    assert words[2]["confidence"] == 0.0


def test_a36_rounded_to_3_decimals():
    segments = [make_segment(0.0, 1.0, "a", [{"word": "a", "start": 0.0, "end": 0.3333333}])]
    out = interpolate_missing_word_timings(segments)
    assert out[0]["words"][0]["end"] == 0.333


def test_a37_single_word_missing_spans_full_segment():
    segments = [make_segment(0.5, 3.5, "a", [{"word": "a"}])]
    out = interpolate_missing_word_timings(segments)
    words = out[0]["words"]
    assert len(words) == 1
    assert words[0]["start"] == 0.5
    assert words[0]["end"] == 3.5


def test_output_validates_against_lyrics_schema():
    segments = [
        make_segment(0.0, 4.0, "a b c d", [
            make_word(0.0, 1.0, "a"),
            {"word": "b"},
            {"word": "c"},
            make_word(3.0, 4.0, "d"),
        ]),
    ]
    out = interpolate_missing_word_timings(segments)
    assert_valid_lyrics(out)
