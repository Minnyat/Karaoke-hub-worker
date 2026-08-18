"""RED-first: karaokeforge.drive.queue.DriveQueue (PR-B2B3, TC-Q01..Q29).

Đọc trước: contracts/README.md D3/D4/§2/§3, contracts/job.schema.json,
docs/plans/DECISIONS.md (PR-B2B3: Q1 keyword-only params DUYỆT, Q3 vocals_file_id
null, Q4 heartbeat null trong processing = stale ngay, Q5 thua claim trả None,
Q6 requeue giữ current_stage + message "Requeued (stale worker)").

Mọi test dùng tmp_path làm drive_root, claim_settle_s=0 (hoặc sleep_fn giả) —
KHÔNG test nào được sleep thật > 0.05s.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import zlib
from pathlib import Path

import pytest

import karaokeforge.drive.queue as queue_mod
from karaokeforge.drive.queue import DriveQueue, LostClaimError

from .helpers_b import (
    EXAMPLE_TOP_LEVEL_KEYS,
    iso,
    make_job,
    parse_iso,
    put_pending,
    put_processing,
    read_json,
    validate,
)

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _pending_dir(tmp_path: Path) -> Path:
    return tmp_path / "queue" / "pending"


def _processing_dir(tmp_path: Path) -> Path:
    return tmp_path / "queue" / "processing"


def _completed_dir(tmp_path: Path) -> Path:
    return tmp_path / "queue" / "completed"


def _failed_dir(tmp_path: Path) -> Path:
    return tmp_path / "queue" / "failed"


def _job_ids_for_buckets(total: int) -> list[str]:
    """Sinh job_id hợp lệ (job_[a-z0-9]{8}) sao cho crc32(id) % total lần lượt
    bằng 0, 1, ..., total-1 — test không được hardcode id "ma thuật" (bẫy plan)."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    found: dict[int, str] = {}
    for combo in itertools.product(chars, repeat=8):
        candidate = "job_" + "".join(combo)
        bucket = zlib.crc32(candidate.encode("utf-8")) % total
        if bucket not in found:
            found[bucket] = candidate
        if len(found) == total:
            break
    assert len(found) == total, "không tìm đủ id cho mỗi bucket"
    return [found[i] for i in range(total)]


# ---------------------------------------------------------------------------
# TC-Q01..Q07 — claim protocol
# ---------------------------------------------------------------------------


def test_claim_moves_job_and_stamps_owner(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert not (_pending_dir(tmp_path) / f"{job['id']}.json").exists()
    on_disk = read_json(_processing_dir(tmp_path) / f"{job['id']}.json")
    assert on_disk["progress"]["worker_id"] == "worker_a"
    assert on_disk["progress"]["started_at"] is not None
    assert on_disk["progress"]["heartbeat_at"] is not None
    assert on_disk["status"] == "processing"
    assert claimed == on_disk
    validate(claimed)


def test_claim_returns_none_when_pending_empty(tmp_path: Path) -> None:
    root = str(tmp_path)
    q = DriveQueue(root, claim_settle_s=0)

    assert q.poll_and_claim("worker_a") is None
    assert list(_processing_dir(tmp_path).iterdir()) == []


def test_lost_claim_returns_none_and_does_not_touch_file(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)
    processing_path = _processing_dir(tmp_path) / f"{job['id']}.json"
    winner_raw: dict[str, str] = {}

    def fake_sleep(_seconds: float) -> None:
        # Giả lập worker_b thắng claim trong khoảng settle window.
        current = read_json(processing_path)
        current["progress"]["worker_id"] = "worker_b"
        current["progress"]["started_at"] = iso()
        current["progress"]["heartbeat_at"] = iso()
        with open(processing_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        winner_raw["content"] = processing_path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=5, sleep_fn=fake_sleep)
    result = q.poll_and_claim("worker_a")

    assert result is None
    assert processing_path.exists()
    assert processing_path.read_text(encoding="utf-8") == winner_raw["content"]
    assert list(_pending_dir(tmp_path).iterdir()) == []


def test_settle_duration_is_injectable(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    calls: list[float] = []

    def spy(seconds: float) -> None:
        calls.append(seconds)

    q = DriveQueue(root, claim_settle_s=0, sleep_fn=spy)
    start = time.monotonic()
    q.poll_and_claim("worker_a")
    elapsed = time.monotonic() - start

    assert calls == [0]
    assert elapsed < 1.0


def test_two_sequential_claims_return_different_jobs_fifo(tmp_path: Path) -> None:
    root = str(tmp_path)
    now = time.time()
    older = make_job(job_id="job_11111111")
    older["created_at"] = iso(-60, base=now)
    older["updated_at"] = older["created_at"]
    newer = make_job(job_id="job_22222222")
    newer["created_at"] = iso(0, base=now)
    newer["updated_at"] = newer["created_at"]
    # Ghi newer trước để loại trừ khả năng implementation FIFO nhầm theo mtime.
    put_pending(root, newer)
    put_pending(root, older)

    q = DriveQueue(root, claim_settle_s=0)
    first = q.poll_and_claim("worker_a")
    second = q.poll_and_claim("worker_a")

    assert first is not None and second is not None
    assert first["id"] == older["id"]
    assert second["id"] == newer["id"]
    assert first["id"] != second["id"]
    assert len(list(_processing_dir(tmp_path).iterdir())) == 2


def test_partition_only_claims_own_bucket(tmp_path: Path) -> None:
    root = str(tmp_path)
    bucket0_id, bucket1_id = _job_ids_for_buckets(2)
    job0 = make_job(job_id=bucket0_id)
    job1 = make_job(job_id=bucket1_id)
    # job1 tạo sớm hơn để nếu partition KHÔNG hoạt động, nó sẽ được chọn trước (bug).
    job1["created_at"] = iso(-60)
    job1["updated_at"] = job1["created_at"]
    put_pending(root, job0)
    put_pending(root, job1)

    q = DriveQueue(root, claim_settle_s=0, partition=(0, 2))
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == bucket0_id
    assert (_pending_dir(tmp_path) / f"{bucket1_id}.json").exists()

    assert q.poll_and_claim("worker_a") is None


def test_partition_disabled_by_default(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == job["id"]


# ---------------------------------------------------------------------------
# TC-Q08..Q13 — heartbeat / progress / checkpoint
# ---------------------------------------------------------------------------


def test_heartbeat_updates_only_heartbeat_at(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    old_heartbeat = iso(-120)
    claimed["progress"]["heartbeat_at"] = old_heartbeat
    started_before = claimed["progress"]["started_at"]

    q.heartbeat(claimed)

    on_disk = read_json(_processing_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["progress"]["started_at"] == started_before
    assert parse_iso(on_disk["progress"]["heartbeat_at"]) > parse_iso(old_heartbeat)


def test_update_progress_writes_stage_and_heartbeat(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.update_progress(claimed, "audio_separation", 42.5, "Đang tách nhạc")

    on_disk = read_json(_processing_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["progress"]["current_stage"] == "audio_separation"
    assert on_disk["progress"]["stages"]["audio_separation"]["status"] == "running"
    assert on_disk["progress"]["stages"]["audio_separation"]["progress"] == 42.5
    assert on_disk["progress"]["message"] == "Đang tách nhạc"
    assert on_disk["updated_at"] is not None
    assert on_disk["progress"]["heartbeat_at"] is not None
    validate(on_disk)


def test_update_progress_rejects_unknown_stage(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    path = _processing_dir(tmp_path) / f"{claimed['id']}.json"
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        q.update_progress(claimed, "separation", 10, "x")

    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("raw,expected", [(150, 100), (-5, 0)])
def test_update_progress_clamps_range(tmp_path: Path, raw: float, expected: float) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.update_progress(claimed, "audio_separation", raw, "msg")

    on_disk = read_json(_processing_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["progress"]["stages"]["audio_separation"]["progress"] == expected
    validate(on_disk)


@pytest.mark.parametrize(
    "checkpoint_name,stage",
    [
        ("audio_separated", "audio_separation"),
        ("lyrics_aligned", "lyrics_alignment"),
        ("video_rendered", "video_render"),
    ],
)
def test_save_checkpoint_sets_flag_and_completes_stage(
    tmp_path: Path, checkpoint_name: str, stage: str
) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.save_checkpoint(claimed, checkpoint_name)

    on_disk = read_json(_processing_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["checkpoints"][checkpoint_name] is True
    assert on_disk["progress"]["stages"][stage]["status"] == "completed"
    assert on_disk["progress"]["stages"][stage]["progress"] == 100
    validate(on_disk)


def test_save_checkpoint_rejects_unknown_name(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    with pytest.raises(ValueError):
        q.save_checkpoint(claimed, "audio_done")


# ---------------------------------------------------------------------------
# TC-Q14..Q15 — kết thúc job
# ---------------------------------------------------------------------------


def test_mark_completed_moves_to_completed(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.mark_completed(claimed)

    assert not (_processing_dir(tmp_path) / f"{claimed['id']}.json").exists()
    on_disk = read_json(_completed_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["status"] == "completed"
    assert on_disk["progress"]["current_stage"] == "done"
    assert on_disk["updated_at"] is not None
    validate(on_disk)


def test_mark_failed_moves_to_failed_with_error(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.mark_failed(claimed, "demucs OOM")

    assert not (_processing_dir(tmp_path) / f"{claimed['id']}.json").exists()
    on_disk = read_json(_failed_dir(tmp_path) / f"{claimed['id']}.json")
    assert on_disk["status"] == "failed"
    assert on_disk["error"] == "demucs OOM"
    validate(on_disk)


# ---------------------------------------------------------------------------
# TC-Q16..Q21 — stale recovery
# ---------------------------------------------------------------------------


def _stale_processing_job(root: str, *, heartbeat_offset: float | None, attempts: int = 0) -> dict:
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["started_at"] = iso(-700)
    job["progress"]["heartbeat_at"] = None if heartbeat_offset is None else iso(heartbeat_offset)
    job["attempts"] = attempts
    put_processing(root, job)
    return job


def test_recover_stale_requeues_after_heartbeat_expiry(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-660)  # 11 phút trước

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [job["id"]]
    assert not (_processing_dir(tmp_path) / f"{job['id']}.json").exists()
    on_disk = read_json(_pending_dir(tmp_path) / f"{job['id']}.json")
    assert on_disk["attempts"] == 1
    assert on_disk["progress"]["worker_id"] is None
    assert on_disk["progress"]["started_at"] is None
    assert on_disk["progress"]["heartbeat_at"] is None
    assert on_disk["status"] == "pending"
    assert on_disk["progress"]["message"] == "Requeued (stale worker)"
    validate(on_disk)


def test_recover_stale_preserves_checkpoints(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["heartbeat_at"] = iso(-660)
    job["checkpoints"]["audio_separated"] = True
    put_processing(root, job)

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    on_disk = read_json(_pending_dir(tmp_path) / f"{job['id']}.json")
    assert on_disk["checkpoints"]["audio_separated"] is True
    assert on_disk["checkpoints"]["lyrics_aligned"] is False


def test_recover_stale_fails_after_max_attempts(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-660, attempts=3)

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [job["id"]]
    assert not (_pending_dir(tmp_path) / f"{job['id']}.json").exists()
    on_disk = read_json(_failed_dir(tmp_path) / f"{job['id']}.json")
    assert on_disk["attempts"] == 4
    assert on_disk["status"] == "failed"
    assert on_disk["error"] == "max retries exceeded"
    validate(on_disk)


def test_recover_stale_ignores_fresh_heartbeat(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-60)  # 1 phút trước, còn sống
    path = _processing_dir(tmp_path) / f"{job['id']}.json"
    before = path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == []
    assert path.exists()
    assert path.read_text(encoding="utf-8") == before


def test_long_running_job_with_fresh_heartbeat_is_not_recovered(tmp_path: Path) -> None:
    """Regression chống lỗi PRD §5.3 (contracts/README.md D3.5 + §6.3): job render
    dài dùng started_at cũ (90 phút, vượt xa mốc 30 phút của PRD cũ) nhưng
    heartbeat_at đều (30s trước) — KHÔNG được coi là stale. Nếu implementation lỡ
    dùng started_at thay vì heartbeat_at, test này FAIL."""
    root = str(tmp_path)
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["started_at"] = iso(-5400)  # 90 phút trước
    job["progress"]["heartbeat_at"] = iso(-30)
    job["attempts"] = 0
    put_processing(root, job)
    path = _processing_dir(tmp_path) / f"{job['id']}.json"
    before = path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == []
    assert path.exists()
    assert path.read_text(encoding="utf-8") == before
    assert read_json(path)["attempts"] == 0


def test_recover_stale_treats_null_heartbeat_as_stale(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=None)  # crash trước khi kịp stamp

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [job["id"]]


# ---------------------------------------------------------------------------
# TC-Q22..Q25 — robustness (file hỏng / thiếu key / không phải json)
# ---------------------------------------------------------------------------


def test_corrupt_json_in_pending_is_skipped(tmp_path: Path) -> None:
    root = str(tmp_path)
    good = make_job(job_id="job_gggggggg")
    put_pending(root, good)
    bad_path = _pending_dir(tmp_path) / "job_bad0000.json"
    bad_path.write_text("{not json", encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == good["id"]
    assert bad_path.exists()
    assert bad_path.read_text(encoding="utf-8") == "{not json"


def test_schema_invalid_json_in_pending_is_skipped(tmp_path: Path) -> None:
    root = str(tmp_path)
    good = make_job(job_id="job_gggggggg")
    put_pending(root, good)

    incomplete_path = _pending_dir(tmp_path) / "job_incompl0.json"
    incomplete_path.parent.mkdir(parents=True, exist_ok=True)
    with open(incomplete_path, "w", encoding="utf-8") as f:
        json.dump({"id": "job_incompl0"}, f)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == good["id"]
    assert incomplete_path.exists()


def test_corrupt_json_in_processing_does_not_break_recovery(tmp_path: Path) -> None:
    root = str(tmp_path)
    stale = _stale_processing_job(root, heartbeat_offset=-660)
    bad_path = _processing_dir(tmp_path) / "job_bad00001.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not json", encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [stale["id"]]
    assert bad_path.exists()


def test_non_json_files_ignored(tmp_path: Path) -> None:
    root = str(tmp_path)
    good = make_job()
    put_pending(root, good)
    pending_dir = _pending_dir(tmp_path)
    (pending_dir / "README.txt").write_text("hello", encoding="utf-8")
    (pending_dir / f"{good['id']}.json.tmp").write_text("junk", encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == good["id"]


# ---------------------------------------------------------------------------
# TC-Q26..Q28 — schema validity qua toàn bộ chuỗi thao tác + timestamp format
# ---------------------------------------------------------------------------


def test_job_json_valid_against_schema_after_every_operation(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)
    q = DriveQueue(root, claim_settle_s=0)

    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    validate(claimed)

    proc_path = _processing_dir(tmp_path) / f"{job['id']}.json"

    q.update_progress(claimed, "audio_separation", 10, "a")
    validate(read_json(proc_path))
    q.update_progress(claimed, "lyrics_alignment", 20, "b")
    validate(read_json(proc_path))
    q.update_progress(claimed, "video_render", 30, "c")
    validate(read_json(proc_path))

    q.save_checkpoint(claimed, "audio_separated")
    validate(read_json(proc_path))
    q.save_checkpoint(claimed, "lyrics_aligned")
    validate(read_json(proc_path))
    q.save_checkpoint(claimed, "video_rendered")
    validate(read_json(proc_path))

    q.heartbeat(claimed)
    validate(read_json(proc_path))

    q.mark_completed(claimed)
    validate(read_json(_completed_dir(tmp_path) / f"{job['id']}.json"))


def test_job_json_valid_after_mark_failed(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None

    q.mark_failed(claimed, "demucs OOM")

    validate(read_json(_failed_dir(tmp_path) / f"{job['id']}.json"))


def test_job_json_valid_after_recover_stale_requeue(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-660)

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    validate(read_json(_pending_dir(tmp_path) / f"{job['id']}.json"))


def test_job_json_valid_after_recover_stale_failed(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-660, attempts=3)

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    validate(read_json(_failed_dir(tmp_path) / f"{job['id']}.json"))


def test_no_extra_fields_written(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    q.update_progress(claimed, "audio_separation", 10, "a")
    q.save_checkpoint(claimed, "audio_separated")
    q.mark_completed(claimed)

    on_disk = read_json(_completed_dir(tmp_path) / f"{job['id']}.json")
    assert set(on_disk.keys()) == EXAMPLE_TOP_LEVEL_KEYS
    assert "_jobFileId" not in on_disk


def test_timestamps_are_utc_iso(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    q.update_progress(claimed, "audio_separation", 10, "a")
    q.mark_completed(claimed)

    on_disk = read_json(_completed_dir(tmp_path) / f"{job['id']}.json")
    for value in (
        on_disk["created_at"],
        on_disk["updated_at"],
        on_disk["progress"]["started_at"],
        on_disk["progress"]["heartbeat_at"],
    ):
        assert _TIMESTAMP_RE.match(value), value
        time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# TC-Q29 — init tạo folder
# ---------------------------------------------------------------------------


def test_init_creates_queue_folders(tmp_path: Path) -> None:
    root = str(tmp_path)
    DriveQueue(root, claim_settle_s=0)

    for name in ("pending", "processing", "completed", "failed"):
        assert (tmp_path / "queue" / name).is_dir()
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "outputs").is_dir()

    DriveQueue(root, claim_settle_s=0)  # gọi lần 2 không lỗi


# ===========================================================================
# PR-B2B3-FIX — hardening claim protocol (TC-Q30..Q46)
# ===========================================================================


def _files_in(root: Path) -> list[str]:
    """Snapshot mọi file (path tương đối) dưới `root` — dùng để chốt không có file
    nào bị tạo/di chuyển ngoài phạm vi mong đợi."""
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


# --- FIX-2: _read_job validate shape, quarantine processing hỏng ------------


def test_nested_poison_in_processing_quarantined_valid_still_recovered(
    tmp_path: Path,
) -> None:
    """TC-Q30: nested type độc trong processing/ không giết recover; job hợp lệ vẫn
    được cứu, file độc bị quarantine sang failed/."""
    root = str(tmp_path)
    good = _stale_processing_job(root, heartbeat_offset=-660)
    poison = make_job(job_id="job_poison01")
    poison["progress"]["stages"] = ["not", "a", "dict"]  # shape độc, vẫn parse được
    put_processing(root, poison)

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [good["id"]]
    assert read_json(_pending_dir(tmp_path) / f"{good['id']}.json")["status"] == "pending"
    assert not (_processing_dir(tmp_path) / "job_poison01.json").exists()
    quarantined = read_json(_failed_dir(tmp_path) / "job_poison01.json")
    assert quarantined["error"] == "invalid/corrupt job json"


def test_nested_poison_in_pending_is_skipped_not_moved(tmp_path: Path) -> None:
    """TC-Q31: shape độc trong pending → skip + không move; job tốt vẫn claim được."""
    root = str(tmp_path)
    good = make_job(job_id="job_gggggggg")
    good["created_at"] = iso(-10)
    put_pending(root, good)
    poison = make_job(job_id="job_poison01")
    poison["created_at"] = iso(-9999)  # cũ hơn -> sẽ được FIFO chọn TRƯỚC nếu không reject
    poison["progress"] = "not-a-dict"  # progress không phải dict
    put_pending(root, poison)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == good["id"]
    # poison còn nguyên trong pending, không bị move sang processing
    assert (_pending_dir(tmp_path) / "job_poison01.json").exists()
    assert not (_processing_dir(tmp_path) / "job_poison01.json").exists()


def test_non_string_created_at_does_not_break_fifo(tmp_path: Path) -> None:
    """TC-Q32: created_at không phải string (shape độc) không làm sort FIFO ném
    TypeError; job hợp lệ vẫn được claim đúng thứ tự."""
    root = str(tmp_path)
    good = make_job(job_id="job_gggggggg")
    good["created_at"] = iso(-30)
    put_pending(root, good)
    weird = make_job(job_id="job_weird001")
    weird["created_at"] = 1234567890  # int, không phải str -> invalid shape
    put_pending(root, weird)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is not None
    assert claimed["id"] == good["id"]


def test_corrupt_shape_in_processing_eventually_quarantined(tmp_path: Path) -> None:
    """TC-Q40: file processing/ parse được nhưng shape hỏng → quarantine sang failed/
    (giải phóng slot, không kẹt vĩnh viễn)."""
    root = str(tmp_path)
    corrupt = make_job(job_id="job_corrupt1")
    corrupt["output"] = "not-a-dict"
    put_processing(root, corrupt)

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    assert not (_processing_dir(tmp_path) / "job_corrupt1.json").exists()
    on_disk = read_json(_failed_dir(tmp_path) / "job_corrupt1.json")
    assert on_disk["error"] == "invalid/corrupt job json"


# --- FIX-2: id an toàn ------------------------------------------------------


def test_malicious_id_in_content_is_rejected_no_path_escape(tmp_path: Path) -> None:
    """TC-Q34: job["id"] chứa traversal/absolute/khác filename → reject, không tạo
    file ngoài drive_root, không claim."""
    root = str(tmp_path)
    pending = _pending_dir(tmp_path)
    pending.mkdir(parents=True, exist_ok=True)
    # filename hợp lệ nhưng content id độc (khác stem + có traversal)
    evil = make_job(job_id="../../../../evil")
    (pending / "job_aaaaaaaa.json").write_text(
        json.dumps(evil), encoding="utf-8"
    )
    # id absolute
    evil2 = make_job(job_id="/tmp/evil2")
    (pending / "job_bbbbbbbb.json").write_text(
        json.dumps(evil2), encoding="utf-8"
    )
    before = _files_in(tmp_path)

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is None
    assert _files_in(tmp_path) == before  # không có file nào bị tạo/di chuyển
    # không có file "evil" ở bất cứ đâu trong cây
    assert not any("evil" in name and name.endswith("evil") for name in _files_in(tmp_path))


# --- FIX-3: một job = một file = một chủ ------------------------------------


def test_recover_skips_requeue_when_completed_exists(tmp_path: Path) -> None:
    """TC-Q35: job vừa ở completed/ vừa ở processing/ (stale) → chỉ xoá bản
    processing/, KHÔNG requeue về pending."""
    root = str(tmp_path)
    stale = _stale_processing_job(root, heartbeat_offset=-660)
    done = make_job(job_id=stale["id"])
    done["status"] = "completed"
    completed_path = _completed_dir(tmp_path) / f"{stale['id']}.json"
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    completed_path.write_text(json.dumps(done), encoding="utf-8")
    completed_before = completed_path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == []
    assert not (_processing_dir(tmp_path) / f"{stale['id']}.json").exists()
    assert not (_pending_dir(tmp_path) / f"{stale['id']}.json").exists()
    assert completed_path.read_text(encoding="utf-8") == completed_before


def test_claim_skips_candidate_already_in_processing(tmp_path: Path) -> None:
    """TC-Q36: candidate ở pending/ nhưng đã có processing/{id}.json (job đang có
    chủ) → skip, KHÔNG move đè; file processing không đổi byte."""
    root = str(tmp_path)
    job = make_job(job_id="job_dup00001")
    put_pending(root, job)
    owned = make_job(job_id="job_dup00001")
    owned["status"] = "processing"
    owned["progress"]["worker_id"] = "worker_owner"
    proc_path = _processing_dir(tmp_path) / "job_dup00001.json"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.write_text(json.dumps(owned), encoding="utf-8")
    proc_before = proc_path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")

    assert claimed is None
    assert proc_path.read_text(encoding="utf-8") == proc_before
    assert (_pending_dir(tmp_path) / "job_dup00001.json").exists()


# --- FIX-4: xác thực quyền sở hữu (LostClaimError) --------------------------


def _claim_one(root: str, tmp_path: Path, worker: str = "worker_a") -> tuple[DriveQueue, dict]:
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim(worker)
    assert claimed is not None
    return q, claimed


def _steal_on_disk(tmp_path: Path, job_id: str, new_worker: str) -> str:
    path = _processing_dir(tmp_path) / f"{job_id}.json"
    current = read_json(path)
    current["progress"]["worker_id"] = new_worker
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_heartbeat_raises_lost_claim_when_owner_changed(tmp_path: Path) -> None:
    """TC-Q37: worker_id trên đĩa khác in-memory → heartbeat raise LostClaimError,
    KHÔNG ghi đè bản của chủ mới."""
    root = str(tmp_path)
    q, claimed = _claim_one(root, tmp_path)
    stolen = _steal_on_disk(tmp_path, claimed["id"], "worker_b")

    with pytest.raises(LostClaimError):
        q.heartbeat(claimed)

    path = _processing_dir(tmp_path) / f"{claimed['id']}.json"
    assert path.read_text(encoding="utf-8") == stolen


def test_update_progress_raises_lost_claim_when_owner_changed(tmp_path: Path) -> None:
    root = str(tmp_path)
    q, claimed = _claim_one(root, tmp_path)
    stolen = _steal_on_disk(tmp_path, claimed["id"], "worker_b")

    with pytest.raises(LostClaimError):
        q.update_progress(claimed, "audio_separation", 10, "x")

    path = _processing_dir(tmp_path) / f"{claimed['id']}.json"
    assert path.read_text(encoding="utf-8") == stolen


def test_save_checkpoint_raises_lost_claim_when_owner_changed(tmp_path: Path) -> None:
    root = str(tmp_path)
    q, claimed = _claim_one(root, tmp_path)
    stolen = _steal_on_disk(tmp_path, claimed["id"], "worker_b")

    with pytest.raises(LostClaimError):
        q.save_checkpoint(claimed, "audio_separated")

    path = _processing_dir(tmp_path) / f"{claimed['id']}.json"
    assert path.read_text(encoding="utf-8") == stolen


def test_heartbeat_raises_lost_claim_when_file_gone(tmp_path: Path) -> None:
    root = str(tmp_path)
    q, claimed = _claim_one(root, tmp_path)
    (_processing_dir(tmp_path) / f"{claimed['id']}.json").unlink()

    with pytest.raises(LostClaimError):
        q.heartbeat(claimed)


# --- FIX-5: heartbeat tương lai = stale, clock skew tolerance ---------------


def test_future_heartbeat_beyond_skew_is_stale(tmp_path: Path) -> None:
    """TC-Q38: heartbeat_at ở tương lai xa (vượt CLOCK_SKEW_TOLERANCE_S) → stale."""
    root = str(tmp_path)
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["heartbeat_at"] = iso(+100000)  # xa trong tương lai
    put_processing(root, job)

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == [job["id"]]
    assert (_pending_dir(tmp_path) / f"{job['id']}.json").exists()


def test_small_future_heartbeat_within_skew_not_stale(tmp_path: Path) -> None:
    """TC-Q39: heartbeat_at hơi lệch tương lai (clock skew 2 worker, trong ngưỡng
    dung sai) → KHÔNG requeue oan."""
    root = str(tmp_path)
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["heartbeat_at"] = iso(+60)  # trong ngưỡng 120s
    put_processing(root, job)
    path = _processing_dir(tmp_path) / f"{job['id']}.json"
    before = path.read_text(encoding="utf-8")

    q = DriveQueue(root, claim_settle_s=0)
    recovered = q.recover_stale_jobs()

    assert recovered == []
    assert path.read_text(encoding="utf-8") == before


# --- FIX-6: sanitize attempts / created_at ----------------------------------


def test_negative_attempts_does_not_bypass_max(tmp_path: Path) -> None:
    """TC-Q41: attempts âm (attacker) → sanitize về 0 rồi +1 = 1, không loop vĩnh
    viễn (nếu không sanitize thì -100+1 = -99 < MAX, requeue mãi mãi)."""
    root = str(tmp_path)
    job = make_job()
    job["status"] = "processing"
    job["progress"]["worker_id"] = "worker_a"
    job["progress"]["heartbeat_at"] = iso(-660)
    job["attempts"] = -100
    put_processing(root, job)

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    on_disk = read_json(_pending_dir(tmp_path) / f"{job['id']}.json")
    assert on_disk["attempts"] == 1


def test_empty_created_at_sorts_last_in_fifo(tmp_path: Path) -> None:
    """TC-Q42: created_at rỗng (không parse được) xếp CUỐI hàng FIFO, không phải
    đầu (nếu để đầu, job mới nhất/thời gian rỗng chiếm chỗ job cũ thật)."""
    root = str(tmp_path)
    real = make_job(job_id="job_realtime")
    real["created_at"] = iso(-30)
    empty = make_job(job_id="job_empty001")
    empty["created_at"] = ""
    put_pending(root, empty)
    put_pending(root, real)

    q = DriveQueue(root, claim_settle_s=0)
    first = q.poll_and_claim("worker_a")
    second = q.poll_and_claim("worker_a")

    assert first is not None and second is not None
    assert first["id"] == "job_realtime"
    assert second["id"] == "job_empty001"


# --- FIX-9: partition validation --------------------------------------------


@pytest.mark.parametrize("partition", [(0, 0), (2, 2), (-1, 2), (0, -3)])
def test_invalid_partition_raises_value_error(tmp_path: Path, partition: tuple[int, int]) -> None:
    """TC-Q45: partition total<=0 hoặc index ngoài [0,total) → ValueError lúc init
    (không để ZeroDivisionError lúc poll)."""
    with pytest.raises(ValueError):
        DriveQueue(str(tmp_path), claim_settle_s=0, partition=partition)


# --- FIX-7: I/O error không giết loop ---------------------------------------


def test_move_oserror_does_not_kill_poll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TC-Q46: shutil.move ném OSError (DriveFS lỗi) → poll trả None, không crash;
    job vẫn ở pending."""
    root = str(tmp_path)
    job = make_job()
    put_pending(root, job)

    def boom(*_a, **_k):
        raise OSError("drivefs sync error")

    monkeypatch.setattr(queue_mod.shutil, "move", boom)
    q = DriveQueue(root, claim_settle_s=0)

    assert q.poll_and_claim("worker_a") is None
    assert (_pending_dir(tmp_path) / f"{job['id']}.json").exists()


# --- FIX-1: fileId (st_ino) không đổi qua transition ------------------------


def test_fileid_preserved_across_mark_completed(tmp_path: Path) -> None:
    """TC-Q33: st_ino không đổi khi mark_completed (D6 poll theo fileId)."""
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    ino_before = (_processing_dir(tmp_path) / f"{claimed['id']}.json").stat().st_ino

    q.mark_completed(claimed)

    ino_after = (_completed_dir(tmp_path) / f"{claimed['id']}.json").stat().st_ino
    assert ino_after == ino_before


def test_fileid_preserved_across_mark_failed(tmp_path: Path) -> None:
    root = str(tmp_path)
    put_pending(root, make_job())
    q = DriveQueue(root, claim_settle_s=0)
    claimed = q.poll_and_claim("worker_a")
    assert claimed is not None
    ino_before = (_processing_dir(tmp_path) / f"{claimed['id']}.json").stat().st_ino

    q.mark_failed(claimed, "boom")

    ino_after = (_failed_dir(tmp_path) / f"{claimed['id']}.json").stat().st_ino
    assert ino_after == ino_before


def test_fileid_preserved_across_requeue(tmp_path: Path) -> None:
    root = str(tmp_path)
    job = _stale_processing_job(root, heartbeat_offset=-660)
    ino_before = (_processing_dir(tmp_path) / f"{job['id']}.json").stat().st_ino

    q = DriveQueue(root, claim_settle_s=0)
    q.recover_stale_jobs()

    ino_after = (_pending_dir(tmp_path) / f"{job['id']}.json").stat().st_ino
    assert ino_after == ino_before


# --- FIX-10: _parse_iso chấp nhận mili-giây ---------------------------------


def test_parse_iso_accepts_milliseconds(tmp_path: Path) -> None:
    """TC (FIX-10): web ghi ISO có phần mili-giây → _parse_iso vẫn ra epoch đúng."""
    from karaokeforge.drive.queue import _parse_iso

    assert _parse_iso("2026-08-18T10:30:00Z") == parse_iso("2026-08-18T10:30:00Z")
    with_ms = _parse_iso("2026-08-18T10:30:00.123Z")
    assert with_ms is not None
    # phần thập phân < 1s: cùng giây với dạng không mili-giây
    assert int(with_ms) == parse_iso("2026-08-18T10:30:00Z")
