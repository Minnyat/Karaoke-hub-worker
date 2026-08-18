"""Helper dùng chung cho test PR-B2B3 (namespace riêng theo DECISIONS.md G1/Q2 —
không có worker/tests/conftest.py trong wave 1, mỗi PR tự đặt helper có tiền tố).

Không phải test, không được pytest tự thu thập (không có hàm bắt đầu bằng `test_`).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

# Tính từ __file__ để không phụ thuộc cwd (bẫy #14 trong PR-B2B3.md).
_CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"
_EXAMPLE_PATH = _CONTRACTS_DIR / "examples" / "job_example.json"
_SCHEMA_PATH = _CONTRACTS_DIR / "job.schema.json"

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_JOB_TEMPLATE: dict = _load_json(_EXAMPLE_PATH)
_SCHEMA: dict = _load_json(_SCHEMA_PATH)

EXAMPLE_TOP_LEVEL_KEYS = set(_JOB_TEMPLATE.keys())


def _set_nested(job: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = job
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def make_job(job_id: str = "job_a1b2c3d4", **overrides: Any) -> dict:
    """Deep-copy job_example.json, đổi id + override field (hỗ trợ dotted-key,
    ví dụ `**{"progress.heartbeat_at": None}`, hoặc key top-level thường)."""
    job = copy.deepcopy(_JOB_TEMPLATE)
    job["id"] = job_id
    for key, value in overrides.items():
        if "." in key:
            _set_nested(job, key, value)
        else:
            job[key] = value
    return job


def validate(job: dict) -> None:
    """jsonschema.validate với contracts/job.schema.json (dev-only dependency)."""
    import jsonschema

    jsonschema.validate(job, _SCHEMA)


def _write_job(path: Path, job: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
    return path


def put_pending(root: str | Path, job: dict) -> Path:
    """Ghi job JSON thẳng vào queue/pending/{id}.json (giả lập WebUI tạo job)."""
    path = Path(root) / "queue" / "pending" / f"{job['id']}.json"
    return _write_job(path, job)


def put_processing(root: str | Path, job: dict) -> Path:
    """Ghi job JSON thẳng vào queue/processing/{id}.json (giả lập job đang chạy dở,
    dùng để test recover_stale_jobs mà không cần đi qua poll_and_claim)."""
    path = Path(root) / "queue" / "processing" / f"{job['id']}.json"
    return _write_job(path, job)


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iso(offset_seconds: float = 0.0, base: float | None = None) -> str:
    """Timestamp UTC ISO, lệch `offset_seconds` so với `base` (mặc định now)."""
    ts = (base if base is not None else time.time()) + offset_seconds
    return time.strftime(ISO_FMT, time.gmtime(ts))


def parse_iso(value: str) -> float:
    """Ngược lại của iso(): parse ISO-UTC -> epoch giây bằng calendar.timegm
    (KHÔNG time.mktime, xem bẫy #2 trong PR-B2B3.md)."""
    import calendar

    return calendar.timegm(time.strptime(value, ISO_FMT))
