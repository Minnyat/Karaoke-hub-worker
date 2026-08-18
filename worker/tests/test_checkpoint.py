"""RED-first: karaokeforge.drive.checkpoint.resume_stage (PR-B2B3, TC-C01..C04).

Đọc trước: docs/plans/PR-B2B3.md §5.3, contracts/README.md §3 (bảng stage/checkpoint).
"""

from __future__ import annotations

import pytest

from karaokeforge.drive.checkpoint import resume_stage

_CONTRACT_STAGES = {"audio_separation", "lyrics_alignment", "video_render"}


@pytest.mark.parametrize(
    "audio_separated,lyrics_aligned,video_rendered,expected",
    [
        (False, False, False, "audio_separation"),
        (True, False, False, "lyrics_alignment"),
        (True, True, False, "video_render"),
        (True, True, True, None),
    ],
)
def test_resume_stage_all_four_combinations(
    audio_separated: bool, lyrics_aligned: bool, video_rendered: bool, expected: str | None
) -> None:
    job = {
        "checkpoints": {
            "audio_separated": audio_separated,
            "lyrics_aligned": lyrics_aligned,
            "video_rendered": video_rendered,
        }
    }
    assert resume_stage(job) == expected


@pytest.mark.parametrize(
    "checkpoints,expected",
    [
        (
            {"audio_separated": False, "lyrics_aligned": True, "video_rendered": False},
            "audio_separation",
        ),
        (
            {"audio_separated": False, "lyrics_aligned": False, "video_rendered": True},
            "audio_separation",
        ),
        (
            {"audio_separated": True, "lyrics_aligned": False, "video_rendered": True},
            "lyrics_alignment",
        ),
    ],
)
def test_resume_stage_out_of_order_checkpoints(checkpoints: dict, expected: str) -> None:
    # Checkpoint bất thường (stage sau xong nhưng stage trước chưa) vẫn phải trả
    # về stage sớm nhất chưa xong, không nhảy cóc (REQ-C02, pipeline tuần tự).
    assert resume_stage({"checkpoints": checkpoints}) == expected


@pytest.mark.parametrize(
    "job",
    [
        {},
        {"checkpoints": {}},
        {"checkpoints": None},
    ],
)
def test_resume_stage_missing_keys(job: dict) -> None:
    # Thiếu checkpoints hoặc thiếu key con -> coi như False, không KeyError (REQ-C03).
    assert resume_stage(job) == "audio_separation"


def test_stage_names_match_contract() -> None:
    combos = [
        (False, False, False),
        (True, False, False),
        (True, True, False),
    ]
    for audio_separated, lyrics_aligned, video_rendered in combos:
        job = {
            "checkpoints": {
                "audio_separated": audio_separated,
                "lyrics_aligned": lyrics_aligned,
                "video_rendered": video_rendered,
            }
        }
        result = resume_stage(job)
        assert result in _CONTRACT_STAGES
