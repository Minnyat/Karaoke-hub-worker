"""Job queue folder-based trên Drive mount. (PR-B2B3)

Toàn bộ logic chỉ dùng os/shutil/json/time (+ zlib.crc32, calendar.timegm — stdlib
thuần, xem docs/plans/DECISIONS.md G3) → test được bằng temp dir local.
Claim protocol move-then-verify + heartbeat: contracts/README.md D3.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
import shutil
import time
import zlib

from karaokeforge.config import Config

logger = logging.getLogger(__name__)


class LostClaimError(Exception):
    """Worker mất quyền sở hữu job (worker_id trên đĩa khác in-memory hoặc file
    processing/ không còn). Ghi vào job JSON sẽ ghi đè bản của chủ mới → thay vì
    thế, raise để B4 bắt và BỎ job đang xử lý (contract D4, PR-B2B3-FIX FIX-4)."""


_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
# id job hợp lệ theo contract §2 (dùng để dựng path an toàn — FIX-2).
_JOB_ID_RE = re.compile(r"^job_[a-z0-9]{8}$")
# Dung sai lệch đồng hồ giữa các worker (FIX-5). heartbeat lệch tương lai quá
# ngưỡng này bị coi là stale (giá trị rác / attacker), dưới ngưỡng thì bỏ qua để
# không requeue oan khi 2 worker lệch giờ nhẹ.
CLOCK_SKEW_TOLERANCE_S = 120.0
_MISSING = object()  # sentinel: file processing/ không còn / không đọc được (FIX-4)
_STAGES = ("audio_separation", "lyrics_alignment", "video_render")
_CHECKPOINT_TO_STAGE = {
    "audio_separated": "audio_separation",
    "lyrics_aligned": "lyrics_alignment",
    "video_rendered": "video_render",
}
_REQUIRED_TOP_LEVEL_KEYS = (
    "id",
    "status",
    "created_at",
    "updated_at",
    "attempts",
    "input",
    "config",
    "progress",
    "checkpoints",
    "output",
    "error",
)


def _now_iso(now_fn) -> str:
    """Timestamp UTC hiện tại theo format contract (%Y-%m-%dT%H:%M:%SZ)."""
    return time.strftime(_ISO_FMT, time.gmtime(now_fn()))


def _parse_iso(value: str | None) -> float | None:
    """Parse timestamp ISO-UTC → epoch giây.

    Dùng `calendar.timegm`, KHÔNG `time.mktime` — `time.mktime` diễn giải chuỗi
    theo local time nên trên máy dev UTC+7, job vừa heartbeat xong sẽ bị tính cũ
    7 giờ (bẫy #2, docs/plans/PR-B2B3.md).

    Chấp nhận CẢ hai dạng: giây (`...:00Z`) và có mili-giây (`...:00.123Z`) — web
    ghi `new Date().toISOString()` sinh phần thập phân, worker vẫn ghi dạng giây
    (FIX-10, INT-Q2). Phần thập phân giây bị cắt (làm tròn xuống giây).
    """
    if not value or not isinstance(value, str):
        return None
    text = value
    if "." in text:
        head, _, frac = text.partition(".")
        # frac = <chữ số thập phân><hậu tố, thường "Z">; giữ lại hậu tố phi-số.
        suffix = frac.lstrip("0123456789")
        text = head + suffix
    try:
        return calendar.timegm(time.strptime(text, _ISO_FMT))
    except (ValueError, TypeError):
        return None


def _int_or_zero(value) -> int:
    """Ép về int, mọi giá trị attacker (str rác, None, dict...) → 0 (FIX-6)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class DriveQueue:
    """Job queue sử dụng các folder queue/{pending,processing,completed,failed}."""

    def __init__(
        self,
        drive_root: str,
        *,
        claim_settle_s: float | None = None,
        stale_after_min: float | None = None,
        max_attempts: int | None = None,
        partition: tuple[int, int] | None = None,
        sleep_fn=time.sleep,
        now_fn=time.time,
    ) -> None:
        """drive_root ví dụ: /content/drive/MyDrive/KaraokeForge (hoặc temp dir khi test).

        Các tham số keyword-only là seam để test (settle time = 0, sleep/now giả)
        mà không đổi contract signature — `DriveQueue(drive_root)` (PR-B4) vẫn chạy
        nguyên vẹn, mặc định đọc từ `Config` (DECISIONS.md PR-B2B3 Q1: DUYỆT).
        """
        self.drive_root = str(drive_root)
        self.claim_settle_s = (
            claim_settle_s if claim_settle_s is not None else Config.CLAIM_SETTLE_S
        )
        self.stale_after_min = (
            stale_after_min if stale_after_min is not None else Config.STALE_AFTER_MIN
        )
        self.max_attempts = (
            max_attempts if max_attempts is not None else Config.MAX_ATTEMPTS
        )
        self.partition = partition if partition is not None else Config.WORKER_PARTITION
        if self.partition is not None:
            index, total = self.partition
            if total <= 0 or not (0 <= index < total):
                raise ValueError(
                    f"partition không hợp lệ {self.partition!r}: cần total > 0 và "
                    f"0 <= index < total (FIX-9)"
                )
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn

        queue_root = os.path.join(self.drive_root, "queue")
        self.pending = os.path.join(queue_root, "pending")
        self.processing = os.path.join(queue_root, "processing")
        self.completed = os.path.join(queue_root, "completed")
        self.failed = os.path.join(queue_root, "failed")
        self.uploads = os.path.join(self.drive_root, "uploads")
        self.outputs = os.path.join(self.drive_root, "outputs")
        for path in (
            self.pending,
            self.processing,
            self.completed,
            self.failed,
            self.uploads,
            self.outputs,
        ):
            os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers riêng
    # ------------------------------------------------------------------

    def _job_path(self, folder: str, job_id: str) -> str:
        return os.path.join(folder, f"{job_id}.json")

    def _load_raw(self, path: str) -> object | None:
        """Đọc + parse JSON thô. Trả None nếu KHÔNG đọc/parse được (OSError, JSON
        hỏng — có thể là partial write đang diễn ra) — caller để nguyên file, vòng
        sau thử lại. KHÔNG validate shape ở đây (FIX-2)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Không đọc được job JSON %s: %s", path, exc)
            return None

    def _is_valid_job(self, job: object, expected_id: str) -> bool:
        """Validate shape + id an toàn (FIX-2). Hợp lệ chỉ khi mọi nhánh nested
        đúng kiểu VÀ job['id'] khớp regex VÀ == stem tên file (expected_id).

        Không bao giờ dùng job['id'] từ nội dung để dựng path — luôn dùng
        expected_id (stem tên file). `attempts` KHÔNG nằm trong điều kiện hợp lệ
        (luôn được sanitize khi dùng — FIX-6)."""
        if not isinstance(job, dict):
            return False
        if not all(key in job for key in _REQUIRED_TOP_LEVEL_KEYS):
            return False
        for key in ("input", "config", "progress", "checkpoints", "output"):
            if not isinstance(job.get(key), dict):
                return False
        if not isinstance(job["progress"].get("stages"), dict):
            return False
        if not isinstance(job.get("created_at"), str):
            return False
        job_id = job.get("id")
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return False
        if job_id != expected_id:
            return False
        return True

    def _read_job(self, path: str) -> dict | None:
        """Đọc job JSON hợp lệ từ đĩa. JSON hỏng / thiếu key / shape sai / id không
        khớp stem tên file → None, không raise (REQ-Q21, FIX-2) — worker bỏ qua và
        xử lý file khác. id an toàn được suy từ stem tên file, KHÔNG từ nội dung."""
        raw = self._load_raw(path)
        if raw is None:
            return None
        expected_id = os.path.splitext(os.path.basename(path))[0]
        if not self._is_valid_job(raw, expected_id):
            logger.warning("Job JSON %s shape/id không hợp lệ, bỏ qua", path)
            return None
        return raw  # type: ignore[return-value]

    def _disk_worker_id(self, job_id: str):
        """worker_id hiện đang ghi trên đĩa (processing/{id}.json), hoặc sentinel
        `_MISSING` nếu file không còn/không đọc được (FIX-4)."""
        raw = self._load_raw(self._job_path(self.processing, job_id))
        if not isinstance(raw, dict):
            return _MISSING
        progress = raw.get("progress")
        if not isinstance(progress, dict):
            return _MISSING
        return progress.get("worker_id")

    def _assert_owner(self, job: dict) -> None:
        """Raise LostClaimError nếu worker_id trên đĩa khác in-memory hoặc file
        processing/ không còn (FIX-4). Gọi TRƯỚC mọi lần ghi để không ghi đè bản
        của chủ mới."""
        job_id = job["id"]
        on_disk = self._disk_worker_id(job_id)
        in_memory = job["progress"].get("worker_id")
        if on_disk is _MISSING or on_disk != in_memory:
            raise LostClaimError(
                f"Job {job_id}: mất quyền sở hữu (đĩa={on_disk!r}, mình={in_memory!r})"
            )

    def _exists_in(self, job_id: str, *folders: str) -> bool:
        return any(os.path.exists(self._job_path(folder, job_id)) for folder in folders)

    def _save_job(self, path: str, job: dict) -> None:
        """Ghi job JSON xuống đĩa (indent=2, ensure_ascii=False để giữ dấu tiếng Việt)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def _in_partition(self, job_id: str) -> bool:
        """True nếu job thuộc bucket của worker này. partition=None → không lọc
        (mặc định tắt, D3.6)."""
        if self.partition is None:
            return True
        index, total = self.partition
        return zlib.crc32(job_id.encode("utf-8")) % total == index

    def _touch(self, job: dict) -> None:
        now = _now_iso(self._now_fn)
        job["updated_at"] = now
        job["progress"]["heartbeat_at"] = now

    def _transition(self, job: dict, src_path: str, dst_folder: str) -> None:
        """Ghi bản mới in-place tại `src_path` rồi `shutil.move` sang `dst_folder`.

        FIX-1 (D6): move = rename → GIỮ NGUYÊN fileId, đúng như bước claim. TUYỆT
        ĐỐI KHÔNG create-new-then-delete (đổi fileId khiến WebUI poll-theo-fileId
        mất dấu job). Nếu `src_path` đã không còn (gọi lặp) → ghi thẳng dst từ dict
        in-memory + log; worker không được chết ở bước kết thúc job.
        """
        job_id = job["id"]
        dst_path = self._job_path(dst_folder, job_id)
        if not os.path.exists(src_path):
            logger.warning(
                "Job %s: src %s đã không còn khi transition sang %s; ghi dst từ memory",
                job_id, src_path, dst_folder,
            )
            self._save_job(dst_path, job)
            return
        self._save_job(src_path, job)  # ghi in-place -> giữ fileId
        try:
            shutil.move(src_path, dst_path)  # rename same-volume -> giữ fileId
        except (OSError, shutil.Error) as exc:
            logger.warning(
                "Job %s: move %s → %s lỗi (%s); fallback ghi dst + xoá src",
                job_id, src_path, dst_folder, exc,
            )
            try:
                self._save_job(dst_path, job)
                os.remove(src_path)
            except OSError:
                pass

    def _quarantine(self, src_path: str, job_id: str) -> None:
        """Đưa file processing/ hỏng-shape sang failed/ với error rõ ràng (FIX-2,
        H3) — giải phóng slot, job không kẹt vĩnh viễn. Giữ fileId (ghi in-place
        rồi move) khi nội dung là dict; nếu là JSON phi-dict (list/scalar) thì dựng
        record failed tối thiểu."""
        dst_path = self._job_path(self.failed, job_id)
        raw = self._load_raw(src_path)
        if isinstance(raw, dict):
            raw["status"] = "failed"
            raw["error"] = "invalid/corrupt job json"
            record = raw
        else:
            record = {"id": job_id, "status": "failed", "error": "invalid/corrupt job json"}
        try:
            self._save_job(src_path, record)  # in-place -> giữ fileId
            shutil.move(src_path, dst_path)
        except (OSError, shutil.Error) as exc:
            logger.warning("Quarantine job %s lỗi: %s", job_id, exc)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def poll_and_claim(self, worker_id: str) -> dict | None:
        """Claim 1 job theo protocol D3 (move → ghi worker_id → settle → verify).

        Trả về job dict nếu claim thành công (bản đọc lại từ đĩa sau verify — nguồn
        sự thật), None nếu không có job / thua claim. Khi thua claim (worker khác
        đã ghi đè trong lúc settle) → trả None ngay, KHÔNG đụng gì thêm vào file
        (D3.3, DECISIONS.md Q5) — vòng poll sau sẽ thử job khác.
        Tôn trọng Config.WORKER_PARTITION nếu được set (D3.6).
        """
        try:
            filenames = [f for f in os.listdir(self.pending) if f.endswith(".json")]
        except OSError as exc:  # FIX-7: DriveFS I/O lỗi -> không claim được, không chết
            logger.warning("Không list được pending/: %s", exc)
            return None

        candidates: list[tuple[bool, str, str]] = []
        for filename in filenames:
            job_id = filename[: -len(".json")]
            if not _JOB_ID_RE.match(job_id):  # stem không phải id hợp lệ -> bỏ
                continue
            if not self._in_partition(job_id):
                continue
            job = self._read_job(self._job_path(self.pending, job_id))
            if job is None:
                continue
            created_at = job["created_at"]  # đã validate là str
            # FIFO: created_at parse được xếp trước theo thời gian; không parse được
            # (rỗng / rác) xếp CUỐI (FIX-6) — key[0]=False sắp trước True.
            candidates.append((_parse_iso(created_at) is None, created_at, job_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        for _unparsable, _created_at, job_id in candidates:
            # FIX-3: một job = một file = một chủ. Nếu id đã tồn tại ở processing/
            # completed/failed → bản pending này là rác/trùng, skip (không move đè).
            if self._exists_in(job_id, self.processing, self.completed, self.failed):
                logger.warning(
                    "Job %s đã tồn tại ở folder khác (processing/completed/failed); "
                    "skip candidate pending trùng",
                    job_id,
                )
                continue
            return self._try_claim(worker_id, job_id)

        return None

    def _try_claim(self, worker_id: str, job_id: str) -> dict | None:
        """Thực thi claim protocol D3 cho 1 job_id: move → stamp → settle → verify.
        Trả về job đã verify hoặc None (thua claim / I/O lỗi)."""
        src_path = self._job_path(self.pending, job_id)
        dst_path = self._job_path(self.processing, job_id)
        try:
            shutil.move(src_path, dst_path)
        except (OSError, shutil.Error) as exc:  # FIX-7: gồm FileNotFoundError
            logger.warning("Claim thất bại khi move job %s: %s", job_id, exc)
            return None

        job = self._read_job(dst_path)
        if job is None:
            return None

        now = _now_iso(self._now_fn)
        job["status"] = "processing"
        job["progress"]["worker_id"] = worker_id
        job["progress"]["started_at"] = now
        job["progress"]["heartbeat_at"] = now
        job["updated_at"] = now
        try:
            self._save_job(dst_path, job)
        except OSError as exc:
            logger.warning("Không ghi được job %s khi stamp owner: %s", job_id, exc)
            return None

        self._sleep_fn(self.claim_settle_s)

        verify = self._read_job(dst_path)
        if verify is None or verify["progress"]["worker_id"] != worker_id:
            logger.warning(
                "Thua claim job %s (worker khác đã ghi đè trong lúc settle); bỏ qua, "
                "không đụng gì thêm",
                job_id,
            )
            return None

        return verify

    def heartbeat(self, job: dict) -> None:
        """Cập nhật progress.heartbeat_at = now (gọi tối thiểu mỗi HEARTBEAT_INTERVAL).

        FIX-4: xác thực quyền sở hữu trước khi ghi — raise LostClaimError nếu chủ
        trên đĩa đã đổi (không ghi đè bản của worker khác)."""
        self._assert_owner(job)
        job["progress"]["heartbeat_at"] = _now_iso(self._now_fn)
        self._save_job(self._job_path(self.processing, job["id"]), job)

    def update_progress(
        self, job: dict, stage: str, progress: float, message: str = ""
    ) -> None:
        """Ghi progress vào job JSON trong processing/ (kèm heartbeat_at, D3.4).

        `stage` chỉ nhận đúng 3 key contract; `progress` clamp về [0, 100].
        """
        if stage not in _STAGES:
            raise ValueError(f"stage không hợp lệ: {stage!r}, phải là một trong {_STAGES}")

        self._assert_owner(job)  # FIX-4
        clamped = max(0.0, min(100.0, float(progress)))
        job["progress"]["current_stage"] = stage
        job["progress"]["stages"][stage]["status"] = "running"
        job["progress"]["stages"][stage]["progress"] = clamped
        job["progress"]["message"] = message
        self._touch(job)

        self._save_job(self._job_path(self.processing, job["id"]), job)

    def save_checkpoint(self, job: dict, checkpoint_name: str) -> None:
        """Set checkpoints[checkpoint_name]=True + đánh dấu stage tương ứng completed.

        Map cố định (contracts/README.md §3): audio_separated→audio_separation,
        lyrics_aligned→lyrics_alignment, video_rendered→video_render.
        """
        if checkpoint_name not in _CHECKPOINT_TO_STAGE:
            raise ValueError(f"checkpoint không hợp lệ: {checkpoint_name!r}")
        self._assert_owner(job)  # FIX-4
        stage = _CHECKPOINT_TO_STAGE[checkpoint_name]

        job["checkpoints"][checkpoint_name] = True
        job["progress"]["stages"][stage]["status"] = "completed"
        job["progress"]["stages"][stage]["progress"] = 100
        self._touch(job)

        self._save_job(self._job_path(self.processing, job["id"]), job)

    def mark_completed(self, job: dict) -> None:
        """status=completed, current_stage=done, move processing/ → completed/."""
        job["status"] = "completed"
        job["progress"]["current_stage"] = "done"
        job["updated_at"] = _now_iso(self._now_fn)

        self._transition(job, self._job_path(self.processing, job["id"]), self.completed)

    def mark_failed(self, job: dict, error: str) -> None:
        """status=failed, ghi error, move processing/ → failed/."""
        job["status"] = "failed"
        job["error"] = error
        job["updated_at"] = _now_iso(self._now_fn)

        self._transition(job, self._job_path(self.processing, job["id"]), self.failed)

    def recover_stale_jobs(self) -> list[str]:
        """Job trong processing/ có heartbeat_at cũ hơn STALE_AFTER_MIN phút:
        attempts+=1; attempts > MAX_ATTEMPTS → failed/, ngược lại → pending/.
        Trả về danh sách job id đã recover.

        QUAN TRỌNG (contract D3.5, contracts/README.md §6.3): stale được xác định
        bằng `heartbeat_at`, TUYỆT ĐỐI KHÔNG dùng `started_at` — đây là lỗi trong
        pseudo-code PRD §5.3 đã được contract sửa. Dùng started_at sẽ requeue oan
        job render dài (ví dụ 40 phút) dù heartbeat vẫn đều. `heartbeat_at = null`
        (worker chết ngay sau move, trước khi kịp stamp bước claim) cũng được coi
        là stale ngay lập tức (DECISIONS.md PR-B2B3 Q4).
        Khi requeue: GIỮ NGUYÊN `progress.current_stage` (hint resume cho worker
        sau) và `checkpoints`; chỉ set message "Requeued (stale worker)"
        (DECISIONS.md Q6, REQ-Q19).
        """
        try:
            filenames = sorted(f for f in os.listdir(self.processing) if f.endswith(".json"))
        except OSError as exc:  # FIX-7
            logger.warning("Không list được processing/: %s", exc)
            return []

        now = self._now_fn()
        stale_seconds = self.stale_after_min * 60
        recovered: list[str] = []

        for filename in filenames:
            job_id = filename[: -len(".json")]
            if not _JOB_ID_RE.match(job_id):
                logger.warning("Bỏ qua file processing tên id không hợp lệ: %s", filename)
                continue
            path = self._job_path(self.processing, job_id)

            raw = self._load_raw(path)
            if raw is None:
                # unreadable (JSON hỏng / partial write) -> để nguyên, vòng sau thử lại
                continue
            if not self._is_valid_job(raw, job_id):
                # parse được nhưng shape/id sai -> quarantine (FIX-2, H3)
                self._quarantine(path, job_id)
                continue
            job = raw  # type: ignore[assignment]

            heartbeat_at = _parse_iso(job["progress"].get("heartbeat_at"))
            if heartbeat_at is not None:
                age = now - heartbeat_at
                future = heartbeat_at - now
                # còn sống: heartbeat chưa quá hạn VÀ không lệch tương lai quá dung
                # sai clock skew (FIX-5: heartbeat tương lai xa = rác = stale).
                if age <= stale_seconds and future <= CLOCK_SKEW_TOLERANCE_S:
                    continue

            # FIX-3: nếu job đã có bản terminal (completed/failed) → chỉ xoá bản
            # processing/ thừa, KHÔNG requeue (tránh chạy lại job đã xong).
            if self._exists_in(job_id, self.completed, self.failed):
                logger.warning(
                    "Job %s đã ở completed/failed; xoá bản processing/ thừa, không requeue",
                    job_id,
                )
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Không xoá được %s: %s", path, exc)
                continue

            attempts = max(0, _int_or_zero(job.get("attempts"))) + 1  # FIX-6
            job["attempts"] = attempts
            job["updated_at"] = _now_iso(self._now_fn)

            if attempts > self.max_attempts:
                job["status"] = "failed"
                job["error"] = "max retries exceeded"
                self._transition(job, path, self.failed)
            else:
                job["status"] = "pending"
                job["progress"]["worker_id"] = None
                job["progress"]["started_at"] = None
                job["progress"]["heartbeat_at"] = None
                job["progress"]["message"] = "Requeued (stale worker)"
                # checkpoints VÀ current_stage giữ nguyên (REQ-Q19, DECISIONS.md Q6)
                self._transition(job, path, self.pending)

            recovered.append(job["id"])

        return recovered
