"""RED-first: karaokeforge.drive.storage.DriveStorage (PR-B2B3, TC-S01..S08).

Đọc trước: contracts/README.md §2 (folder layout, tên file output), §6.6
(cleanup separated tracks bắt buộc); PRD dòng 1013 (giữ instrumental.wav).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import karaokeforge.drive.storage as storage_mod
from karaokeforge.drive.storage import DriveStorage

_CONTRACT_OUTPUT_FILENAMES = (
    "vocals.wav",
    "instrumental.wav",
    "lyrics_raw.json",
    "lyrics_aligned.json",
    "lyrics_edited.json",
    "preview.mp4",
    "karaoke_final.mp4",
)


def test_path_helpers(tmp_path: Path) -> None:
    root = str(tmp_path)
    storage = DriveStorage(root)
    job_id = "job_a1b2c3d4"

    upload_dir = storage.upload_dir(job_id)
    assert os.path.normpath(upload_dir) == os.path.normpath(
        os.path.join(root, "uploads", job_id)
    )

    output_dir = storage.output_dir(job_id)
    assert os.path.normpath(output_dir) == os.path.normpath(
        os.path.join(root, "outputs", job_id)
    )
    assert os.path.isdir(output_dir)


def test_publish_outputs_copies_and_returns_map(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    vocals_src = local_dir / "v.wav"
    vocals_src.write_bytes(b"vocals-data")
    instrumental_src = local_dir / "i.wav"
    instrumental_src.write_bytes(b"instrumental-data")

    result = storage.publish_outputs(
        "job_x",
        {"vocals.wav": str(vocals_src), "instrumental.wav": str(instrumental_src)},
    )

    assert set(result.keys()) == {"vocals.wav", "instrumental.wav"}
    for path in result.values():
        assert os.path.isfile(path)
    out_dir = tmp_path / "outputs" / "job_x"
    assert (out_dir / "vocals.wav").read_bytes() == b"vocals-data"
    assert (out_dir / "instrumental.wav").read_bytes() == b"instrumental-data"


def test_publish_outputs_leaves_no_temp_file(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    src = local_dir / "a.wav"
    src.write_bytes(b"data")

    storage.publish_outputs("job_x", {"vocals.wav": str(src)})

    out_dir = tmp_path / "outputs" / "job_x"
    names = os.listdir(out_dir)
    assert names == ["vocals.wav"]
    assert not any(n.startswith(".") or n.endswith(".part") for n in names)


def test_publish_outputs_overwrites_existing(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    src = local_dir / "a.wav"

    src.write_bytes(b"v1")
    storage.publish_outputs("job_x", {"vocals.wav": str(src)})
    src.write_bytes(b"v2")
    storage.publish_outputs("job_x", {"vocals.wav": str(src)})

    out_dir = tmp_path / "outputs" / "job_x"
    assert os.listdir(out_dir) == ["vocals.wav"]
    assert (out_dir / "vocals.wav").read_bytes() == b"v2"


def test_publish_outputs_missing_source_raises(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    ok_src = local_dir / "ok.wav"
    ok_src.write_bytes(b"ok-data")
    missing_src = local_dir / "missing.wav"

    with pytest.raises(FileNotFoundError):
        storage.publish_outputs(
            "job_x",
            {"vocals.wav": str(ok_src), "instrumental.wav": str(missing_src)},
        )

    out_dir = tmp_path / "outputs" / "job_x"
    names = os.listdir(out_dir)
    assert "vocals.wav" in names
    assert (out_dir / "vocals.wav").read_bytes() == b"ok-data"
    assert not any(n.endswith(".part") for n in names)


@pytest.mark.parametrize("filename", _CONTRACT_OUTPUT_FILENAMES)
def test_publish_outputs_accepts_contract_filenames(tmp_path: Path, filename: str) -> None:
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir(exist_ok=True)
    src = local_dir / "src.bin"
    src.write_bytes(b"data")

    result = storage.publish_outputs("job_x", {filename: str(src)})

    assert filename in result
    assert os.path.isfile(tmp_path / "outputs" / "job_x" / filename)


def test_cleanup_removes_stems_keeps_instrumental(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    out_dir = Path(storage.output_dir("job_x"))
    all_files = (
        "vocals.wav",
        "drums.wav",
        "bass.wav",
        "other.wav",
        "instrumental.wav",
        "karaoke_final.mp4",
        "lyrics_aligned.json",
        "lyrics_edited.json",
    )
    for name in all_files:
        (out_dir / name).write_bytes(b"data")

    storage.cleanup_intermediate("job_x")

    remaining = set(os.listdir(out_dir))
    assert remaining == {
        "instrumental.wav",
        "karaoke_final.mp4",
        "lyrics_aligned.json",
        "lyrics_edited.json",
    }


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    out_dir = Path(storage.output_dir("job_x"))
    (out_dir / "vocals.wav").write_bytes(b"data")

    storage.cleanup_intermediate("job_x")
    storage.cleanup_intermediate("job_x")  # gọi lần 2 trên folder đã dọn -> không lỗi
    storage.cleanup_intermediate("job_never_existed")  # folder chưa từng tồn tại -> không lỗi


# ===========================================================================
# PR-B2B3-FIX — hardening storage (TC-S09, TC-S10)
# ===========================================================================


@pytest.mark.parametrize(
    "unsafe_id",
    ["../evil", "..", "a/b", "a\\b", "/abs/path", "", "."],
)
def test_output_dir_rejects_unsafe_id(tmp_path: Path, unsafe_id: str) -> None:
    """TC-S09: output_dir/upload_dir reject job_id không an toàn (traversal, tách
    path, absolute) → ValueError, không tạo folder ngoài outputs/."""
    storage = DriveStorage(str(tmp_path))
    with pytest.raises(ValueError):
        storage.output_dir(unsafe_id)
    with pytest.raises(ValueError):
        storage.upload_dir(unsafe_id)
    # không có folder rác nào bị tạo ngoài phạm vi
    assert not (tmp_path.parent / "evil").exists()


def test_output_dir_still_accepts_normal_id(tmp_path: Path) -> None:
    storage = DriveStorage(str(tmp_path))
    out = storage.output_dir("job_a1b2c3d4")
    assert os.path.isdir(out)


def test_publish_cleans_orphan_part_files(tmp_path: Path) -> None:
    """TC-S10a: .part mồ côi (từ lần crash trước) bị dọn đầu publish_outputs."""
    storage = DriveStorage(str(tmp_path))
    out_dir = Path(storage.output_dir("job_x"))
    # orphan tên KHÁC file sắp publish -> chỉ bị dọn nếu publish thật sự quét .part
    orphan = out_dir / ".instrumental.wav.part"
    orphan.write_bytes(b"stale-junk")

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    src = local_dir / "v.wav"
    src.write_bytes(b"good")

    storage.publish_outputs("job_x", {"vocals.wav": str(src)})

    names = os.listdir(out_dir)
    assert not any(n.endswith(".part") for n in names)
    assert (out_dir / "vocals.wav").read_bytes() == b"good"


def test_publish_leaves_no_part_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TC-S10b: copy2 ném giữa chừng → không để lại .part rác (try/finally)."""
    storage = DriveStorage(str(tmp_path))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    src = local_dir / "v.wav"
    src.write_bytes(b"data")

    def boom(_src, dst, *_a, **_k):
        # giả lập copy dở dang: tạo .part rồi mới ném -> đúng kịch bản rác thật
        with open(dst, "wb") as f:
            f.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(storage_mod.shutil, "copy2", boom)

    with pytest.raises(OSError):
        storage.publish_outputs("job_x", {"vocals.wav": str(src)})

    out_dir = tmp_path / "outputs" / "job_x"
    assert not any(n.endswith(".part") for n in os.listdir(out_dir))
