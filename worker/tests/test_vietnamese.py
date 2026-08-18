"""Test thuần cho karaokeforge.pipeline.vietnamese.fix_vietnamese — KHÔNG mock.

Bảng REQ: xem docs/plans/PR-A2A3.md §5.3 (V-01..V-17).
"""
from __future__ import annotations

import copy
import logging
import unicodedata

import pytest

from tests.helpers_a2a3 import assert_valid_lyrics, make_segment, make_word
from karaokeforge.pipeline.vietnamese import VIETNAMESE_CORRECTIONS, fix_vietnamese


def test_v01_basic_word_and_text_corrected():
    segments = [
        make_segment(0.0, 1.0, "nguoi", [make_word(0.0, 1.0, "nguoi")]),
    ]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "người"
    assert out[0]["words"][0]["word"] == "người"


def test_v02_timing_invariant_and_counts_unchanged():
    segments = [
        make_segment(
            0.0,
            1.0,
            "nguoi yeu em",
            [
                make_word(0.0, 0.3, "nguoi", 0.9),
                make_word(0.3, 0.6, "yeu", 0.8),
                make_word(0.6, 1.0, "em", 0.7),
            ],
        ),
    ]
    before = copy.deepcopy(segments)
    out = fix_vietnamese(segments)

    assert len(out) == len(before)
    for seg_out, seg_before in zip(out, before):
        assert len(seg_out["words"]) == len(seg_before["words"])
        assert seg_out["start"] == seg_before["start"]
        assert seg_out["end"] == seg_before["end"]
        for w_out, w_before in zip(seg_out["words"], seg_before["words"]):
            assert w_out["start"] == w_before["start"]
            assert w_out["end"] == w_before["end"]
            assert w_out["confidence"] == w_before["confidence"]


def test_v03_does_not_mutate_input():
    segments = [
        make_segment(0.0, 1.0, "nguoi", [make_word(0.0, 1.0, "nguoi")]),
    ]
    snapshot = copy.deepcopy(segments)
    fix_vietnamese(segments)
    assert segments == snapshot


def test_v04_case_preserved():
    segments = [
        make_segment(0.0, 1.0, "Nguoi", [make_word(0.0, 1.0, "Nguoi")]),
        make_segment(0.0, 1.0, "NGUOI", [make_word(0.0, 1.0, "NGUOI")]),
        make_segment(0.0, 1.0, "nguoi", [make_word(0.0, 1.0, "nguoi")]),
    ]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "Người"
    assert out[1]["text"] == "NGƯỜI"
    assert out[2]["text"] == "người"


def test_v05_punctuation_preserved():
    segments = [
        make_segment(0.0, 1.0, "khong,", [make_word(0.0, 1.0, "khong,")]),
        make_segment(0.0, 1.0, "(nguoi)", [make_word(0.0, 1.0, "(nguoi)")]),
        make_segment(0.0, 1.0, "yeu...", [make_word(0.0, 1.0, "yeu...")]),
    ]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "không,"
    assert out[1]["text"] == "(người)"
    assert out[2]["text"] == "yêu..."


def test_v06_words_outside_dict_unchanged():
    segments = [
        make_segment(0.0, 1.0, "xyzzy guitar", [make_word(0.0, 0.5, "xyzzy"), make_word(0.5, 1.0, "guitar")]),
    ]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "xyzzy guitar"
    assert out[0]["words"][0]["word"] == "xyzzy"
    assert out[0]["words"][1]["word"] == "guitar"


def test_v07_already_correct_nfd_becomes_nfc():
    nfd_word = unicodedata.normalize("NFD", "người")
    assert nfd_word != "người"  # confirm NFD form khác NFC (dấu tổ hợp)
    segments = [
        make_segment(0.0, 1.0, nfd_word, [make_word(0.0, 1.0, nfd_word)]),
    ]
    out = fix_vietnamese(segments)
    assert unicodedata.normalize("NFC", nfd_word) == out[0]["text"]
    assert out[0]["text"] == "người"
    assert out[0]["words"][0]["word"] == "người"


def test_v08_lookup_succeeds_inside_nfd_mixed_string():
    mixed_text = "tôi " + unicodedata.normalize("NFD", "yêu") + " nguoi"
    segments = [make_segment(0.0, 1.0, mixed_text, [])]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "tôi yêu người"


def test_v09_output_always_nfc():
    nfd_word = unicodedata.normalize("NFD", "thương")
    segments = [
        make_segment(0.0, 1.0, f"{nfd_word} nguoi xyzzy", [
            make_word(0.0, 0.3, nfd_word),
            make_word(0.3, 0.6, "nguoi"),
            make_word(0.6, 1.0, "xyzzy"),
        ]),
    ]
    out = fix_vietnamese(segments)
    assert unicodedata.is_normalized("NFC", out[0]["text"])
    for w in out[0]["words"]:
        assert unicodedata.is_normalized("NFC", w["word"])


def test_v10_empty_segments():
    assert fix_vietnamese([]) == []


def test_v11_segment_without_words_key():
    segments = [{"start": 0.0, "end": 1.0, "text": "nguoi"}]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "người"
    assert "words" not in out[0]


def test_v12_whitespace_format_preserved():
    segments = [make_segment(0.0, 1.0, "nguoi   yeu\tem", [])]
    out = fix_vietnamese(segments)
    assert out[0]["text"] == "người   yêu\tem"


def test_v13_join_equality_preserved():
    text = "nguoi yeu em"
    words = [make_word(0.0, 0.3, "nguoi"), make_word(0.3, 0.6, "yeu"), make_word(0.6, 1.0, "em")]
    segments = [make_segment(0.0, 1.0, text, words)]
    assert " ".join(w["word"] for w in segments[0]["words"]) == segments[0]["text"]

    out = fix_vietnamese(segments)
    assert " ".join(w["word"] for w in out[0]["words"]) == out[0]["text"]


def test_v14_correction_logged(caplog):
    segments = [make_segment(0.0, 1.0, "nguoi", [make_word(0.0, 1.0, "nguoi")])]
    with caplog.at_level(logging.INFO):
        fix_vietnamese(segments)
    assert "nguoi" in caplog.text
    assert "người" in caplog.text


def test_v15_no_correction_no_log(caplog):
    segments = [make_segment(0.0, 1.0, "xyzzy guitar", [make_word(0.0, 1.0, "xyzzy")])]
    with caplog.at_level(logging.INFO):
        fix_vietnamese(segments)
    assert caplog.text == ""


FORBIDDEN_WORDS = [
    "cho", "co", "ma", "ban", "con", "tai", "tay", "ta", "noi", "nho", "dam", "canh", "cai",
]


@pytest.mark.parametrize("word", FORBIDDEN_WORDS)
def test_v16_forbidden_ambiguous_words_locked(word):
    assert word not in VIETNAMESE_CORRECTIONS


def test_v17_dict_shape_locked():
    assert len(VIETNAMESE_CORRECTIONS) > 0
    for key, value in VIETNAMESE_CORRECTIONS.items():
        assert key.isascii()
        assert key == key.casefold()
        assert unicodedata.is_normalized("NFC", value)
        assert value != key


def test_output_validates_against_lyrics_schema():
    segments = [
        make_segment(0.0, 1.0, "nguoi yeu em", [
            make_word(0.0, 0.3, "nguoi", 0.9),
            make_word(0.3, 0.6, "yeu", 0.8),
            make_word(0.6, 1.0, "em", 0.7),
        ]),
    ]
    out = fix_vietnamese(segments)
    assert_valid_lyrics(out)
