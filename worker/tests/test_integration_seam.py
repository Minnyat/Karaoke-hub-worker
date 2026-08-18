"""RED-first: seam WebUI (buildJobData) <-> worker (DriveQueue), TC-I01..TC-I13.

Đọc trước: contracts/README.md D3/D4/§2/§3, contracts/job.schema.json,
docs/plans/DECISIONS.md (PR-B2B3 Q4/Q6, PR-INT Q2/Q4), docs/plans/PR-INT.md.

Fixture do web sinh: worker/tests/fixtures/webui_job_*.json — KHÔNG sửa tay,
regenerate bằng: cd web && UPDATE_FIXTURES=1 npx vitest run scripts

Mục đích: chứng minh job JSON do `buildJobData` (web/lib/job-service.ts) sinh ra
đi trọn vòng đời DriveQueue (worker/karaokeforge/drive/queue.py) THẬT trên một
temp dir đóng vai Drive mount — không cần Drive/Colab thật (docs/plans/PR-INT.md
§1). Không tạo `helpers_int.py` (DECISIONS.md §PR-INT Q4 DUYỆT) — helper để
inline trong chính file này.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from karaokeforge.drive.checkpoint import resume_stage
from karaokeforge.drive.queue import DriveQueue

from .helpers_b import read_json, validate

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"  # REQ-I36: không cwd
FIXTURE_NAMES = ("webui_job_default.json", "webui_job_edge.json", "webui_job_full.json")

JOB_ID_RE = re.compile(r"^job_[a-z0-9]{8}$")
WORKER_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")  # REQ-I31
WEB_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")  # REQ-I30

# Field bị đổi bởi poll_and_claim (worker/karaokeforge/drive/queue.py:224-228,
# xem PR-INT.md §4.3).
CLAIM_MUTATED = {
    "status",
    "updated_at",
    "progress.worker_id",
    "progress.started_at",
    "progress.heartbeat_at",
}

# Map stage -> checkpoint theo contracts/README.md §3 (không import tên private
# _CHECKPOINT_TO_STAGE của queue.py — giữ test độc lập với chi tiết implementation).
_STAGE_TO_CHECKPOINT = {
    "audio_separation": "audio_separated",
    "lyrics_alignment": "lyrics_aligned",
    "video_render": "video_rendered",
}


def load_fixture(name: str) -> dict:
    """Nạp fixture JSON; thiếu file -> pytest.fail có hướng dẫn regenerate (RED
    sạch, không phải collection error)."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.fail(f"Thiếu fixture {name}. Sinh bằng: cd web && UPDATE_FIXTURES=1 npx vitest run scripts")
    return read_json(path)


def copy_to_pending(root: Path, name: str) -> Path:
    """Copy NGUYÊN BYTES fixture vào queue/pending/{id}.json (REQ-I10) — chứng
    minh worker đọc đúng byte stream mà WebUI sẽ upload, không re-serialize."""
    job = load_fixture(name)
    dst = root / "queue" / "pending" / f"{job['id']}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES_DIR / name, dst)
    return dst


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Làm phẳng dict lồng bất kỳ cấp -> {"progress.stages.video_render.status":
    "pending", ...}. Giá trị không phải dict (kể cả None/bool/số) là lá."""
    if not isinstance(node, dict):
        return {prefix: node}
    result: dict[str, Any] = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, dotted))
        else:
            result[dotted] = value
    return result


_MISSING = object()


def diff_keys(before: dict, after: dict) -> set[str]:
    """Tập dotted-key có giá trị khác nhau giữa `before` và `after` (dùng cho
    REQ-I13/I17/I34) — bao gồm cả key chỉ xuất hiện ở một bên."""
    flat_before = flatten(before)
    flat_after = flatten(after)
    keys = set(flat_before) | set(flat_after)
    return {
        key
        for key in keys
        if flat_before.get(key, _MISSING) != flat_after.get(key, _MISSING)
    }


def make_queue(tmp_path: Path, *, now_fn=None) -> tuple[DriveQueue, list[float]]:
    """DriveQueue thật + sleep spy; claim_settle_s=0 (REQ-I11, REQ-I35 — không
    sleep thật). `now_fn` tuỳ chọn để test stale recovery điều khiển được thời
    gian (REQ-I35: không time.sleep thật > 0.05s)."""
    calls: list[float] = []
    kwargs: dict[str, Any] = {"claim_settle_s": 0, "sleep_fn": calls.append}
    if now_fn is not None:
        kwargs["now_fn"] = now_fn
    q = DriveQueue(str(tmp_path), **kwargs)
    return q, calls


def _advance_all_stages(q: DriveQueue, job: dict) -> None:
    """Chạy hết pipeline: update_progress + save_checkpoint theo đúng thứ tự
    audio_separation -> lyrics_alignment -> video_render (contracts/README.md
    §3), dừng khi resume_stage(job) trả None."""
    stage = resume_stage(job)
    while stage is not None:
        q.update_progress(job, stage, 50, f"Đang xử lý {stage}")
        q.save_checkpoint(job, _STAGE_TO_CHECKPOINT[stage])
        stage = resume_stage(job)


class _FakeClock:
    """Đồng hồ giả điều khiển được cho `now_fn` — offline, không sleep thật
    (REQ-I35). `advance()` mô phỏng thời gian trôi qua giữa các heartbeat."""

    def __init__(self, start: float | None = None) -> None:
        self.value = start if start is not None else time.time()

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


# ---------------------------------------------------------------------------
# TC-I01 — fixture hợp lệ tại nguồn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_valid_at_source(fixture_name: str) -> None:
    job = load_fixture(fixture_name)

    validate(job)
    assert JOB_ID_RE.match(job["id"])
    assert job["status"] == "pending"
    assert job["attempts"] == 0
    assert job["checkpoints"] == {
        "audio_separated": False,
        "lyrics_aligned": False,
        "video_rendered": False,
    }
    assert job["progress"]["worker_id"] is None
    assert job["progress"]["started_at"] is None
    assert job["progress"]["heartbeat_at"] is None
    assert job["progress"]["current_stage"] == "waiting"
    assert WEB_TS_RE.match(job["created_at"])
    assert WEB_TS_RE.match(job["updated_at"])
    assert "_jobFileId" not in job


# ---------------------------------------------------------------------------
# TC-I02/TC-I03 — claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_claim_accepts_webui_json_bytes(tmp_path: Path, fixture_name: str) -> None:
    fixture = load_fixture(fixture_name)
    copy_to_pending(tmp_path, fixture_name)
    q, calls = make_queue(tmp_path)

    claimed = q.poll_and_claim("worker_int")

    assert claimed is not None
    assert not (tmp_path / "queue" / "pending" / f"{fixture['id']}.json").exists()
    proc_path = tmp_path / "queue" / "processing" / f"{fixture['id']}.json"
    assert proc_path.exists()
    assert calls == [0]
    validate(read_json(proc_path))


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_claim_diff_exactly_five_fields(tmp_path: Path, fixture_name: str) -> None:
    fixture = load_fixture(fixture_name)
    copy_to_pending(tmp_path, fixture_name)
    q, _calls = make_queue(tmp_path)

    claimed = q.poll_and_claim("worker_int")
    assert claimed is not None

    proc_path = tmp_path / "queue" / "processing" / f"{fixture['id']}.json"
    on_disk = read_json(proc_path)

    assert diff_keys(fixture, on_disk) == CLAIM_MUTATED
    assert on_disk["progress"]["worker_id"] == "worker_int"
    assert on_disk["status"] == "processing"
    assert WORKER_TS_RE.match(on_disk["progress"]["started_at"])
    assert WORKER_TS_RE.match(on_disk["progress"]["heartbeat_at"])


# ---------------------------------------------------------------------------
# TC-I04 — vòng đời đủ 3 stage, diff mỗi bước nằm trong tập cho phép
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_stage_lifecycle_diffs_stay_within_allowed_fields(
    tmp_path: Path, fixture_name: str
) -> None:
    copy_to_pending(tmp_path, fixture_name)
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    proc_path = tmp_path / "queue" / "processing" / f"{job['id']}.json"
    assert resume_stage(job) == "audio_separation"

    stage_order = ("audio_separation", "lyrics_alignment", "video_render")
    next_stage_after = {
        "audio_separation": "lyrics_alignment",
        "lyrics_alignment": "video_render",
        "video_render": None,
    }

    prev = read_json(proc_path)
    for stage in stage_order:
        checkpoint = _STAGE_TO_CHECKPOINT[stage]

        q.update_progress(job, stage, 50, f"Đang xử lý {stage}")
        after_progress = read_json(proc_path)
        validate(after_progress)
        allowed_progress = {
            "progress.current_stage",
            f"progress.stages.{stage}.status",
            f"progress.stages.{stage}.progress",
            "progress.message",
            "updated_at",
            "progress.heartbeat_at",
        }
        assert diff_keys(prev, after_progress) <= allowed_progress
        prev = after_progress

        q.save_checkpoint(job, checkpoint)
        after_checkpoint = read_json(proc_path)
        validate(after_checkpoint)
        allowed_checkpoint = {
            f"checkpoints.{checkpoint}",
            f"progress.stages.{stage}.status",
            f"progress.stages.{stage}.progress",
            "updated_at",
            "progress.heartbeat_at",
        }
        assert diff_keys(prev, after_checkpoint) <= allowed_checkpoint
        prev = after_checkpoint

        assert resume_stage(job) == next_stage_after[stage]

    assert resume_stage(job) is None


# ---------------------------------------------------------------------------
# TC-I05/TC-I06 — mark_completed + bất biến input/config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_mark_completed_full_lifecycle(tmp_path: Path, fixture_name: str) -> None:
    copy_to_pending(tmp_path, fixture_name)
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None
    job_id = job["id"]

    _advance_all_stages(q, job)
    q.mark_completed(job)

    pending_dir = tmp_path / "queue" / "pending"
    processing_dir = tmp_path / "queue" / "processing"
    completed_dir = tmp_path / "queue" / "completed"
    failed_dir = tmp_path / "queue" / "failed"

    assert not (processing_dir / f"{job_id}.json").exists()
    assert list(pending_dir.glob("*.json")) == []
    assert list(processing_dir.glob("*.json")) == []
    assert list(failed_dir.glob("*.json")) == []
    assert [p.name for p in completed_dir.glob("*.json")] == [f"{job_id}.json"]

    on_disk = read_json(completed_dir / f"{job_id}.json")
    validate(on_disk)
    assert on_disk["status"] == "completed"
    assert on_disk["progress"]["current_stage"] == "done"
    assert on_disk["checkpoints"] == {
        "audio_separated": True,
        "lyrics_aligned": True,
        "video_rendered": True,
    }
    for stage in ("audio_separation", "lyrics_alignment", "video_render"):
        assert on_disk["progress"]["stages"][stage]["status"] == "completed"
        assert on_disk["progress"]["stages"][stage]["progress"] == 100
    assert on_disk["attempts"] == 0
    assert on_disk["error"] is None


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_input_and_config_unchanged_through_completion(
    tmp_path: Path, fixture_name: str
) -> None:
    fixture = load_fixture(fixture_name)
    copy_to_pending(tmp_path, fixture_name)
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    _advance_all_stages(q, job)
    q.mark_completed(job)

    on_disk = read_json(tmp_path / "queue" / "completed" / f"{job['id']}.json")
    assert on_disk["input"] == fixture["input"]
    assert on_disk["config"] == fixture["config"]


# ---------------------------------------------------------------------------
# TC-I07/TC-I08 — giá trị "dễ mất" của fixture edge
# ---------------------------------------------------------------------------


def test_edge_guide_vocal_volume_zero_survives_lifecycle(tmp_path: Path) -> None:
    copy_to_pending(tmp_path, "webui_job_edge.json")
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    _advance_all_stages(q, job)
    q.mark_completed(job)

    on_disk = read_json(tmp_path / "queue" / "completed" / f"{job['id']}.json")
    volume = on_disk["config"]["guide_vocal_volume"]
    assert volume == 0
    assert isinstance(volume, (int, float)) and not isinstance(volume, bool)
    assert on_disk["config"]["guide_vocal"] is False
    assert on_disk["input"]["lyrics_file_id"] is None
    assert on_disk["input"]["has_lyrics_input"] is False


def test_edge_vietnamese_text_and_message_survive_claim(tmp_path: Path) -> None:
    fixture = load_fixture("webui_job_edge.json")
    copy_to_pending(tmp_path, "webui_job_edge.json")
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    proc_path = tmp_path / "queue" / "processing" / f"{job['id']}.json"
    on_disk = read_json(proc_path)
    assert fixture["input"]["audio_filename"] == "Em của ngày hôm qua.mp3"
    assert on_disk["input"]["audio_filename"] == fixture["input"]["audio_filename"]
    assert fixture["progress"]["message"] == "Đang chờ worker xử lý..."
    assert on_disk["progress"]["message"] == fixture["progress"]["message"]

    raw_text = proc_path.read_text(encoding="utf-8")
    assert "\\u" not in raw_text


# ---------------------------------------------------------------------------
# TC-I09 — mark_failed
# ---------------------------------------------------------------------------


def test_mark_failed_after_claim(tmp_path: Path) -> None:
    fixture = load_fixture("webui_job_default.json")
    copy_to_pending(tmp_path, "webui_job_default.json")
    q, _calls = make_queue(tmp_path)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    q.mark_failed(job, "demucs OOM")

    job_id = job["id"]
    assert not (tmp_path / "queue" / "processing" / f"{job_id}.json").exists()
    on_disk = read_json(tmp_path / "queue" / "failed" / f"{job_id}.json")
    validate(on_disk)
    assert on_disk["status"] == "failed"
    assert on_disk["error"] == "demucs OOM"
    assert on_disk["input"] == fixture["input"]
    assert on_disk["config"] == fixture["config"]


# ---------------------------------------------------------------------------
# TC-I10/TC-I11 — stale recovery
# ---------------------------------------------------------------------------


def test_stale_recovery_requeues_null_heartbeat_fixture(tmp_path: Path) -> None:
    """Fixture web mới sinh luôn có heartbeat_at=null (buildJobData) — đặt thẳng
    vào processing/ (byte nguyên) đã đủ để recover_stale_jobs() coi là stale
    NGAY (DECISIONS.md PR-B2B3 Q4), không cần chỉnh sửa gì."""
    fixture = load_fixture("webui_job_default.json")
    assert fixture["progress"]["heartbeat_at"] is None
    processing_dst = tmp_path / "queue" / "processing" / f"{fixture['id']}.json"
    processing_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES_DIR / "webui_job_default.json", processing_dst)

    q, _calls = make_queue(tmp_path)
    recovered = q.recover_stale_jobs()

    assert recovered == [fixture["id"]]
    assert not processing_dst.exists()
    on_disk = read_json(tmp_path / "queue" / "pending" / f"{fixture['id']}.json")
    validate(on_disk)
    assert on_disk["attempts"] == 1
    assert on_disk["status"] == "pending"
    assert on_disk["progress"]["message"] == "Requeued (stale worker)"
    assert on_disk["progress"]["worker_id"] is None
    assert on_disk["progress"]["started_at"] is None
    assert on_disk["progress"]["heartbeat_at"] is None
    assert on_disk["progress"]["current_stage"] == fixture["progress"]["current_stage"]
    assert on_disk["checkpoints"] == fixture["checkpoints"]


def test_max_attempts_moves_to_failed(tmp_path: Path) -> None:
    copy_to_pending(tmp_path, "webui_job_default.json")
    clock = _FakeClock()
    q, _calls = make_queue(tmp_path, now_fn=clock)
    job = q.poll_and_claim("worker_int")
    assert job is not None

    q.update_progress(job, "audio_separation", 50, "Đang tách nhạc")
    q.save_checkpoint(job, "audio_separated")

    proc_path = tmp_path / "queue" / "processing" / f"{job['id']}.json"
    on_disk = read_json(proc_path)
    on_disk["attempts"] = 3  # mô phỏng job đã qua claim + 1 checkpoint, thử 3 lần (REQ-I20)
    with open(proc_path, "w", encoding="utf-8") as f:
        json.dump(on_disk, f, indent=2, ensure_ascii=False)

    clock.advance(11 * 60)  # > Config.STALE_AFTER_MIN (10 phút, offline, không sleep thật)

    recovered = q.recover_stale_jobs()

    assert recovered == [job["id"]]
    assert not proc_path.exists()
    failed_path = tmp_path / "queue" / "failed" / f"{job['id']}.json"
    on_disk_failed = read_json(failed_path)
    validate(on_disk_failed)
    assert on_disk_failed["attempts"] == 4
    assert on_disk_failed["status"] == "failed"
    assert on_disk_failed["error"] == "max retries exceeded"
    assert on_disk_failed["checkpoints"]["audio_separated"] is True
    assert on_disk_failed["checkpoints"]["lyrics_aligned"] is False


# ---------------------------------------------------------------------------
# TC-I12 — FIFO cùng định dạng timestamp
# ---------------------------------------------------------------------------


def test_fifo_claims_by_created_at_not_filename(tmp_path: Path) -> None:
    default_fixture = load_fixture("webui_job_default.json")
    edge_fixture = load_fixture("webui_job_edge.json")
    # Cùng định dạng giây (không mili-giây); default (10:30) sinh trước edge
    # (10:31) nhưng id "job_0edge001" đứng trước "job_c0ffee01" theo alphabet
    # -> nếu implementation lỡ sort theo tên file, thứ tự sẽ SAI.
    assert default_fixture["created_at"] < edge_fixture["created_at"]
    copy_to_pending(tmp_path, "webui_job_default.json")
    copy_to_pending(tmp_path, "webui_job_edge.json")

    q, _calls = make_queue(tmp_path)
    first = q.poll_and_claim("worker_int")
    second = q.poll_and_claim("worker_int")

    assert first is not None and second is not None
    assert first["id"] == default_fixture["id"]
    assert second["id"] == edge_fixture["id"]


# ---------------------------------------------------------------------------
# TC-I13 — offline, không import thư viện nặng, không sleep thật
# ---------------------------------------------------------------------------


_FORBIDDEN_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(torch|demucs|whisperx|googleapiclient)\b", re.MULTILINE
)


def test_offline_and_no_heavy_imports(tmp_path: Path) -> None:
    # Regex neo đầu dòng (không phải substring thô) để không tự khớp nhầm vào
    # chính chuỗi mô tả bên dưới.
    source = Path(__file__).read_text(encoding="utf-8")
    assert not _FORBIDDEN_IMPORT_RE.search(source)

    q, _calls = make_queue(tmp_path)
    assert q._sleep_fn is not time.sleep
