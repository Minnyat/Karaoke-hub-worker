"""Main worker entry point — polling loop. (PR-B4, wave 2)

Ghép DriveQueue + 3 pipeline stage theo checkpoint recovery (contracts/README.md
D3/D4). `KaraokeWorker.run_forever` là vòng lặp bất tận (poll → process → mark),
`process_job` chạy tuần tự 3 stage bắt đầu từ checkpoint chưa xong
(`karaokeforge.drive.resume_stage`).

Ba rủi ro chính đã xử lý (xem docs/plans/PR-B4.md mục 1):
1. Heartbeat: `separator.separate()`/`transcriber.transcribe_and_align()` không có
   callback tiến độ → `_HeartbeatWorker` đập `queue.heartbeat()` từ thread nền
   trong lúc 2 stage này chạy.
2. Guide vocal: renderer chỉ nhận 1 đường audio; khi `config.guide_vocal=true`,
   `_mix_guide_vocal` mix vocals vào instrumental bằng ffmpeg TRƯỚC khi render.
3. Resume trên Colab mới: `Config.TEMP_DIR` là local SSD, mất sau khi runtime
   chết — `_resolve_stage_input` tìm lại input đã publish trên Drive
   `outputs/{job_id}/`, hạ checkpoint và chạy lại nếu không còn ở đâu cả.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from karaokeforge.config import Config
from karaokeforge.drive import DriveQueue, DriveStorage, LostClaimError, resume_stage
from karaokeforge.pipeline.renderer import KaraokeRenderer
from karaokeforge.pipeline.separator import AudioSeparator
from karaokeforge.pipeline.transcriber import LyricsTranscriber
from karaokeforge.utils.audio import get_audio_duration
from karaokeforge.utils.logger import get_logger

logger = get_logger(__name__)

# Tên stage & checkpoint — khớp từng ký tự với contracts/README.md §3.
STAGE_SEPARATION = "audio_separation"
STAGE_ALIGNMENT = "lyrics_alignment"
STAGE_RENDER = "video_render"
STAGE_ORDER = (STAGE_SEPARATION, STAGE_ALIGNMENT, STAGE_RENDER)
CHECKPOINT_OF = {
    STAGE_SEPARATION: "audio_separated",
    STAGE_ALIGNMENT: "lyrics_aligned",
    STAGE_RENDER: "video_rendered",
}

_FFMPEG_STDERR_TAIL_LINES = 20


class _HeartbeatWorker:
    """Đập heartbeat mỗi `interval` giây trong lúc 1 stage dài chạy (contract D3.4).

    Dùng như context manager, sống đúng bằng phạm vi 1 stage. BẮT BUỘC dừng +
    join thread trước khi job rời `processing/` (`mark_completed`/`mark_failed`)
    — nếu không, `DriveQueue.heartbeat()` sẽ ghi lại `queue/processing/{id}.json`
    sau khi file đã bị move đi, hồi sinh job đã xong và khiến worker khác/chính
    nó render lại từ đầu (review #1 / attacker H2, bẫy #4 mục 8 của plan).

    Dùng `Event.wait(interval)` thay `time.sleep` để dừng ngay khi stage kết
    thúc, không phải chờ hết chu kỳ heartbeat.
    """

    def __init__(
        self,
        queue: DriveQueue,
        job: dict,
        lock: threading.Lock,
        interval: float,
    ) -> None:
        self._queue = queue
        self._job = job
        self._lock = lock
        self._interval = max(interval, 0.001)
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_HeartbeatWorker":
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"hb-{self._job['id']}"
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._stop is not None
        while not self._stop.wait(self._interval):
            try:
                with self._lock:
                    self._queue.heartbeat(self._job)
            except LostClaimError:
                # Mất quyền sở hữu — dừng đập heartbeat ngay, lỗi thật sẽ nổ lại
                # ở lần gọi queue.* tiếp theo trên thread chính (save_checkpoint/
                # update_progress cũng gọi _assert_owner).
                logger.warning(
                    "Heartbeat job %s: mất quyền sở hữu, dừng thread", self._job["id"]
                )
                self._stop.set()
            except Exception:
                logger.warning("Heartbeat job %s lỗi", self._job["id"], exc_info=True)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval * 2))
            if self._thread.is_alive():
                logger.error(
                    "Heartbeat thread job %s không join được trong %.1fs",
                    self._job["id"],
                    max(5.0, self._interval * 2),
                )
        return False  # không nuốt exception (kể cả KeyboardInterrupt)


class KaraokeWorker:
    """Worker chính chạy trên Colab: poll → resume theo checkpoint → 3 stage →
    publish → mark_completed/failed."""

    def __init__(
        self,
        worker_id: str,
        drive_root: str,
        *,
        queue: DriveQueue | None = None,
        storage: DriveStorage | None = None,
        poll_interval: float | None = None,
        heartbeat_interval: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """`queue`/`storage`/`poll_interval`/`heartbeat_interval`/`sleep_fn` là
        keyword-only param có default — seam test theo tiền lệ đã DUYỆT cho
        `DriveQueue` (DECISIONS.md PR-B2B3 Q1). `KaraokeWorker(worker_id,
        drive_root)` (cách notebook gọi) không đổi hành vi."""
        self.worker_id = worker_id
        self.drive_root = drive_root
        self.jobs_done = 0
        self.current_job: str | None = None

        self.queue = queue if queue is not None else DriveQueue(drive_root)
        self.storage = storage if storage is not None else DriveStorage(drive_root)
        self.poll_interval = (
            poll_interval if poll_interval is not None else Config.POLL_INTERVAL
        )
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else Config.HEARTBEAT_INTERVAL
        )
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # vòng lặp
    # ------------------------------------------------------------------

    def run_forever(self, *, max_iterations: int | None = None) -> None:
        """Polling loop: `recover_stale_jobs()` → `poll_and_claim()` →
        `process_job()` → `mark_completed`/`mark_failed`.

        `max_iterations` (test-only) dừng sau N vòng thay vì chạy vô hạn.
        `KeyboardInterrupt` (Ctrl-C / stop button Colab) dừng vòng lặp sạch sẽ,
        KHÔNG `mark_failed` job đang dở — job ở lại `processing/`, worker sau sẽ
        resume theo checkpoint hoặc `recover_stale_jobs` đưa về `pending/` sau
        `STALE_AFTER_MIN` phút (D3.5).
        """
        logger.info(
            "Worker %s khởi động (drive_root=%s)", self.worker_id, self.drive_root
        )
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                iterations += 1
                self._run_one_iteration()
        except KeyboardInterrupt:
            logger.warning(
                "Nhận Ctrl-C — dừng worker. Job đang dở (nếu có) ở lại processing/, "
                "worker sau sẽ resume theo checkpoint."
            )
        finally:
            logger.info(
                "Worker %s dừng. Jobs done: %d", self.worker_id, self.jobs_done
            )

    def _run_one_iteration(self) -> None:
        try:
            recovered = self.queue.recover_stale_jobs()
            if recovered:
                logger.info("Đã recover %d job stale: %s", len(recovered), recovered)
        except Exception:
            logger.exception("Lỗi khi recover_stale_jobs")

        job: dict | None = None
        try:
            job = self.queue.poll_and_claim(self.worker_id)
        except Exception:
            logger.exception("Lỗi khi poll_and_claim")
            job = None

        if job is None:
            self._sleep_fn(self.poll_interval)
            return

        self.current_job = job["id"]
        try:
            self.process_job(job)
        except LostClaimError:
            logger.warning(
                "Job %s: mất quyền sở hữu giữa chừng (worker khác đã claim) — "
                "bỏ job, không mark_failed/mark_completed",
                job["id"],
            )
        except Exception as exc:
            logger.exception("Job %s thất bại", job["id"])
            try:
                self.queue.mark_failed(job, str(exc))
            except Exception:
                logger.exception("Lỗi khi mark_failed job %s", job["id"])
        else:
            try:
                self.queue.mark_completed(job)
                self.jobs_done += 1
            except LostClaimError:
                logger.warning(
                    "Job %s: mất quyền sở hữu khi mark_completed — bỏ qua", job["id"]
                )
            except Exception:
                logger.exception("Lỗi khi mark_completed job %s", job["id"])
        finally:
            self._clear_temp(job["id"])
            self.current_job = None

    # ------------------------------------------------------------------
    # 1 job
    # ------------------------------------------------------------------

    def process_job(self, job: dict) -> None:
        """Chạy các stage còn thiếu theo checkpoint (`resume_stage`), heartbeat
        đều đặn, publish output, cleanup sau khi render xong."""
        audio_local, user_lyrics = self._prepare_input(job)
        stage = resume_stage(job)
        if stage is None:
            return  # đã đủ 3 checkpoint — run_forever sẽ mark_completed

        ctx: dict[str, Any] = {
            "job_id": job["id"],
            "audio_local": audio_local,
            "user_lyrics": user_lyrics,
        }
        idx = STAGE_ORDER.index(stage)
        while idx < len(STAGE_ORDER):
            current = STAGE_ORDER[idx]

            if current == STAGE_SEPARATION:
                self._run_separation(job, ctx)
                idx += 1
                continue

            if current == STAGE_ALIGNMENT:
                vocals_path = self._resolve_stage_input(job, "vocals.wav")
                if vocals_path is None:
                    logger.warning(
                        "Job %s: thiếu vocals.wav (temp lẫn Drive) dù checkpoint "
                        "audio_separated=true — chạy lại tách nhạc (REQ-P08)",
                        job["id"],
                    )
                    job["checkpoints"]["audio_separated"] = False
                    idx = STAGE_ORDER.index(STAGE_SEPARATION)
                    continue
                ctx["vocals_path"] = vocals_path
                self._run_alignment(job, ctx)
                idx += 1
                continue

            # STAGE_RENDER
            instrumental_path = self._resolve_stage_input(job, "instrumental.wav")
            if instrumental_path is None:
                logger.warning(
                    "Job %s: thiếu instrumental.wav (temp lẫn Drive) — chạy lại "
                    "tách nhạc (REQ-P08)",
                    job["id"],
                )
                job["checkpoints"]["audio_separated"] = False
                idx = STAGE_ORDER.index(STAGE_SEPARATION)
                continue
            lyrics_path = self._resolve_stage_input(job, "lyrics_aligned.json")
            if lyrics_path is None:
                logger.warning(
                    "Job %s: thiếu lyrics_aligned.json — chạy lại alignment (REQ-P08)",
                    job["id"],
                )
                job["checkpoints"]["lyrics_aligned"] = False
                idx = STAGE_ORDER.index(STAGE_ALIGNMENT)
                continue

            ctx["instrumental_path"] = instrumental_path
            ctx["lyrics_path"] = lyrics_path
            self._run_render(job, ctx)
            idx += 1

    # ------------------------------------------------------------------
    # stage: chuẩn bị input
    # ------------------------------------------------------------------

    def _prepare_input(self, job: dict) -> tuple[str, str | None]:
        """Tìm audio nguồn `uploads/{job_id}/original.<ext>` (fallback glob
        `original.*`), điền `input.audio_duration`, đọc `lyrics_input.txt` nếu
        `has_lyrics_input=true` (REQ-P02/P07/P18)."""
        job_id = job["id"]
        upload_dir = self.storage.upload_dir(job_id)
        audio_filename = job["input"].get("audio_filename") or ""
        ext = os.path.splitext(audio_filename)[1]

        audio_local: str | None = None
        if ext:
            candidate = os.path.join(upload_dir, f"original{ext}")
            if os.path.isfile(candidate):
                audio_local = candidate
        if audio_local is None:
            try:
                for name in sorted(os.listdir(upload_dir)):
                    if name.startswith("original."):
                        candidate = os.path.join(upload_dir, name)
                        if os.path.isfile(candidate):
                            audio_local = candidate
                            break
            except OSError:
                pass

        if audio_local is None:
            raise RuntimeError(
                f"Không tìm thấy audio gốc trong {upload_dir} "
                f"(cần file 'original.<ext>')"
            )

        if job["input"].get("audio_duration") is None:
            try:
                job["input"]["audio_duration"] = get_audio_duration(audio_local)
            except Exception:
                logger.warning(
                    "Job %s: không đọc được audio_duration", job_id, exc_info=True
                )

        user_lyrics: str | None = None
        if job["input"].get("has_lyrics_input"):
            lyrics_path = os.path.join(upload_dir, "lyrics_input.txt")
            try:
                with open(lyrics_path, "r", encoding="utf-8") as f:
                    user_lyrics = f.read()
            except OSError:
                logger.warning(
                    "Job %s: has_lyrics_input=true nhưng thiếu %s",
                    job_id,
                    lyrics_path,
                )
                user_lyrics = None

        return audio_local, user_lyrics

    def _resolve_stage_input(self, job: dict, filename: str) -> str | None:
        """Tìm `filename` ở `TEMP_DIR/{job_id}/` trước, rồi `outputs/{job_id}/`
        trên Drive (REQ-P08 — resume trên Colab mới, temp đã mất). Không tìm
        thấy ở đâu cả → None (caller hạ checkpoint và chạy lại)."""
        job_id = job["id"]
        temp_path = os.path.join(self._job_temp_dir(job_id), filename)
        if os.path.isfile(temp_path):
            return temp_path
        drive_path = os.path.join(self.storage.output_dir(job_id), filename)
        if os.path.isfile(drive_path):
            return drive_path
        return None

    # ------------------------------------------------------------------
    # stage 1: tách nhạc
    # ------------------------------------------------------------------

    def _run_separation(self, job: dict, ctx: dict) -> None:
        self._update(job, STAGE_SEPARATION, 0.0, "Đang tách nhạc và giọng hát...")
        model_name = job["config"].get("demucs_model") or Config.DEFAULT_DEMUCS_MODEL
        temp_dir = self._job_temp_dir(job["id"])

        separator = AudioSeparator()
        with _HeartbeatWorker(self.queue, job, self._lock, self.heartbeat_interval):
            separator.load_model(model_name)
            result = separator.separate(ctx["audio_local"], temp_dir)
        # REQ-P05: unload trước khi stage 2 load Whisper (tuần tự trên GPU).
        separator.unload()

        self._publish(
            job["id"],
            {"vocals.wav": result["vocals"], "instrumental.wav": result["instrumental"]},
        )
        with self._lock:
            self.queue.save_checkpoint(job, "audio_separated")

    # ------------------------------------------------------------------
    # stage 2: nhận diện + align lời
    # ------------------------------------------------------------------

    def _run_alignment(self, job: dict, ctx: dict) -> None:
        self._update(job, STAGE_ALIGNMENT, 0.0, "Đang nhận diện lời bài hát...")
        whisper_model = job["config"].get("whisper_model") or Config.DEFAULT_WHISPER_MODEL
        language = job["input"].get("language", "vi")

        transcriber = LyricsTranscriber()
        with _HeartbeatWorker(self.queue, job, self._lock, self.heartbeat_interval):
            segments = transcriber.transcribe_and_align(
                ctx["vocals_path"],
                language=language,
                user_lyrics=ctx.get("user_lyrics"),
                whisper_model=whisper_model,
            )

        temp_dir = self._job_temp_dir(job["id"])
        lyrics_path = os.path.join(temp_dir, "lyrics_aligned.json")
        with open(lyrics_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)

        self._publish(job["id"], {"lyrics_aligned.json": lyrics_path})
        with self._lock:
            self.queue.save_checkpoint(job, "lyrics_aligned")

    # ------------------------------------------------------------------
    # stage 3: render video (+ guide vocal mix)
    # ------------------------------------------------------------------

    def _run_render(self, job: dict, ctx: dict) -> None:
        self._update(job, STAGE_RENDER, 0.0, "Đang render video...")

        with open(ctx["lyrics_path"], "r", encoding="utf-8") as f:
            lyrics = json.load(f)

        render_audio = ctx["instrumental_path"]
        config = job["config"]
        if config.get("guide_vocal"):
            vocals_path = self._resolve_stage_input(job, "vocals.wav")
            if vocals_path is None:
                raise RuntimeError(
                    f"Job {job['id']}: guide_vocal=true nhưng không tìm thấy "
                    "vocals.wav để mix (đã bị cleanup?)"
                )
            mix_path = os.path.join(self._job_temp_dir(job["id"]), "instrumental_guide.wav")
            render_audio = self._mix_guide_vocal(
                render_audio,
                vocals_path,
                config.get("guide_vocal_volume", 0.15),
                mix_path,
            )

        render_config = self._build_render_config(job)
        output_path = os.path.join(self._job_temp_dir(job["id"]), "karaoke_final.mp4")

        def on_progress(pct: float) -> None:
            self._update(job, STAGE_RENDER, pct, "Đang render video...")

        renderer = KaraokeRenderer()
        renderer.render(render_audio, lyrics, output_path, render_config, on_progress=on_progress)

        self._publish(job["id"], {"karaoke_final.mp4": output_path})
        with self._lock:
            self.queue.save_checkpoint(job, "video_rendered")

        # REQ-P15: cleanup CHỈ sau khi video đã publish thành công.
        self.storage.cleanup_intermediate(job["id"])

    def _build_render_config(self, job: dict) -> dict:
        """Bản copy của `job["config"]` cộng thêm field renderer cần — TUYỆT ĐỐI
        không mutate `job["config"]` (schema `additionalProperties: false`,
        REQ-P11). `fps` chốt cứng `Config.DEFAULT_FPS` (REQ-P13). Font: B4 bỏ
        REQ-P12 — mapping tên family → tên file font đã làm ở
        `video/templates/base.py` (PR-A4-FIX), worker chỉ truyền `font_dir`."""
        cfg = job["config"]
        return {
            **cfg,
            "fps": Config.DEFAULT_FPS,
            "font_dir": Config.FONT_DIR,
            "ffmpeg_preset": Config.FFMPEG_PRESET,
            "ffmpeg_crf": Config.FFMPEG_CRF,
        }

    def _mix_guide_vocal(
        self, instrumental_path: str, vocals_path: str, volume: float, out_path: str
    ) -> str:
        """Mix vocals (âm lượng `volume`) vào instrumental bằng ffmpeg `amix`
        (REQ-G01→G05). `normalize=0` bắt buộc — mặc định `amix` chia biên độ cho
        số input, làm instrumental giảm ~50% âm lượng (bẫy #6 mục 8 plan).
        `duration=first`: đầu ra bám theo độ dài instrumental, không để vocals
        kéo dài/cắt ngắn video (REQ-G03). Output PCM (không nén) vì renderer sẽ
        encode lại sang AAC."""
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            instrumental_path,
            "-i",
            vocals_path,
            "-filter_complex",
            f"[1:a]volume={volume}[gv];"
            f"[0:a][gv]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Không tìm thấy binary 'ffmpeg' để mix guide vocal. "
                "Cài đặt FFmpeg trước khi chạy worker."
            ) from exc

        if result.returncode != 0:
            stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")
            tail = "\n".join(stderr_text.splitlines()[-_FFMPEG_STDERR_TAIL_LINES:])
            raise RuntimeError(
                f"ffmpeg mix guide vocal thất bại (mã lỗi {result.returncode}):\n"
                f"--- stderr (tail) ---\n{tail}"
            )
        return out_path

    # ------------------------------------------------------------------
    # helper chung
    # ------------------------------------------------------------------

    def _publish(self, job_id: str, files: dict[str, str]) -> dict[str, str]:
        """Publish file temp lên `outputs/{job_id}/` (REQ-P03/P16)."""
        return self.storage.publish_outputs(job_id, files)

    def _update(self, job: dict, stage: str, pct: float, message: str) -> None:
        """`update_progress` có lock chung (REQ-H03) — heartbeat thread + callback
        `on_progress` của renderer có thể chạy song song."""
        with self._lock:
            self.queue.update_progress(job, stage, pct, message)

    def _job_temp_dir(self, job_id: str) -> str:
        path = os.path.join(Config.TEMP_DIR, job_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _clear_temp(self, job_id: str) -> None:
        """Xoá `TEMP_DIR/{job_id}/` sau khi job kết thúc — cả thành công lẫn fail
        (REQ-P17)."""
        path = os.path.join(Config.TEMP_DIR, job_id)
        shutil.rmtree(path, ignore_errors=True)
