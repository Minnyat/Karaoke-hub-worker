"""Test PR-B4: worker main loop + heartbeat + guide-vocal mix + Colab notebook.

Nguyên tắc (CLAUDE.md, DECISIONS.md G1): RED trước, `DriveQueue`/`DriveStorage`
dùng THẬT trên `tmp_path`, chỉ mock 3 pipeline class (`helpers_b4.make_worker`).
Không test nào sleep thật > 0.05s trừ 2 test heartbeat (TC-H02/H03, < 2s tổng).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

import karaokeforge.utils.gpu as gpu_module
import karaokeforge.worker as worker_module
from karaokeforge.config import Config
from karaokeforge.drive import LostClaimError

from . import helpers_b4 as b4
from .helpers_b import EXAMPLE_TOP_LEVEL_KEYS
from .helpers_b4 import (
    CONTROL,
    DEFAULT_FAKE_SEGMENTS,
    calls_of,
    call_names,
    completed_path,
    fake_lyrics_bytes,
    failed_path,
    make_job,
    make_worker,
    output_path,
    processing_path,
    read_json,
    read_json_retry,
    record_call,
    seed_outputs,
    setup_drive,
    validate,
    wait_until,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ======================================================================
# process_job — happy path
# ======================================================================


def test_full_pipeline_three_stages_completes_job(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert call_names() == [
        "separator.load_model",
        "separator.separate",
        "separator.unload",
        "transcriber.transcribe_and_align",
        "renderer.render",
    ]
    for name in ("instrumental.wav", "lyrics_aligned.json", "karaoke_final.mp4"):
        assert output_path(root, job["id"], name).is_file()

    data = read_json(completed_path(root, job["id"]))
    assert data["status"] == "completed"
    assert data["progress"]["current_stage"] == "done"
    assert data["checkpoints"] == {
        "audio_separated": True,
        "lyrics_aligned": True,
        "video_rendered": True,
    }
    assert worker.jobs_done == 1


def test_completed_job_validates_against_schema(tmp_path, monkeypatch):
    job = make_job()
    original_keys = set(job.keys())
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    data = read_json(completed_path(root, job["id"]))
    validate(data)
    assert set(data.keys()) == original_keys == EXAMPLE_TOP_LEVEL_KEYS


def test_progress_updated_for_every_stage(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    seen: list[tuple[str, float]] = []
    original_update = worker.queue.update_progress

    def spy(job_arg, stage, pct, message=""):
        seen.append((stage, pct))
        return original_update(job_arg, stage, pct, message)

    monkeypatch.setattr(worker.queue, "update_progress", spy)

    worker.run_forever(max_iterations=1)

    seen_stages = {stage for stage, _pct in seen}
    assert seen_stages == {"audio_separation", "lyrics_alignment", "video_render"}
    for stage in seen_stages:
        assert any(pct == 0.0 for s, pct in seen if s == stage)

    data = read_json(completed_path(root, job["id"]))
    for stage in ("audio_separation", "lyrics_alignment", "video_render"):
        assert data["progress"]["stages"][stage]["status"] == "completed"
        assert data["progress"]["stages"][stage]["progress"] == 100


# ======================================================================
# process_job — resume theo checkpoint
# ======================================================================


def test_resume_skips_completed_separation(tmp_path, monkeypatch):
    job = make_job(**{"checkpoints.audio_separated": True})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    seed_outputs(
        root,
        job["id"],
        {"vocals.wav": b"OLDVOCALS", "instrumental.wav": b"OLDINSTRUMENTAL"},
    )
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert "separator.separate" not in call_names()
    assert "transcriber.transcribe_and_align" in call_names()
    assert "renderer.render" in call_names()
    assert read_json(completed_path(root, job["id"]))["status"] == "completed"


def test_resume_only_render_stage(tmp_path, monkeypatch):
    job = make_job(
        **{"checkpoints.audio_separated": True, "checkpoints.lyrics_aligned": True}
    )
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    seed_outputs(
        root,
        job["id"],
        {
            "instrumental.wav": b"OLDINSTRUMENTAL",
            "lyrics_aligned.json": fake_lyrics_bytes(),
        },
    )
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert call_names() == ["renderer.render"]
    render_call = calls_of("renderer.render")[0]
    assert render_call["lyrics"] == DEFAULT_FAKE_SEGMENTS
    assert read_json(completed_path(root, job["id"]))["status"] == "completed"


def test_resume_reruns_stage_when_output_missing(tmp_path, monkeypatch):
    job = make_job(**{"checkpoints.audio_separated": True})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    # outputs/{id}/ rỗng: temp local đã mất (Colab mới) -> phải chạy lại stage 1.
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert "separator.separate" in call_names()
    data = read_json(completed_path(root, job["id"]))
    assert data["status"] == "completed"
    assert data["checkpoints"]["audio_separated"] is True


def test_all_checkpoints_true_completes_immediately(tmp_path, monkeypatch):
    job = make_job(
        **{
            "checkpoints.audio_separated": True,
            "checkpoints.lyrics_aligned": True,
            "checkpoints.video_rendered": True,
        }
    )
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    seed_outputs(
        root,
        job["id"],
        {
            "instrumental.wav": b"X",
            "lyrics_aligned.json": fake_lyrics_bytes(),
            "karaoke_final.mp4": b"X",
        },
    )
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert call_names() == []
    assert read_json(completed_path(root, job["id"]))["status"] == "completed"


# ======================================================================
# process_job — lỗi
# ======================================================================


def test_stage_failure_marks_job_failed(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    CONTROL["transcriber_raise"] = RuntimeError("whisper boom")

    worker.run_forever(max_iterations=1)

    data = read_json(failed_path(root, job["id"]))
    assert data["status"] == "failed"
    assert "whisper boom" in data["error"]
    assert data["checkpoints"]["audio_separated"] is True
    assert not processing_path(root, job["id"]).exists()
    assert worker.jobs_done == 0


def test_missing_audio_marks_failed_not_crash(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, with_audio=False)
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    data = read_json(failed_path(root, job["id"]))
    assert data["status"] == "failed"
    assert "uploads" in data["error"]

    # Vòng lặp vẫn sống — job khác chạy được ở vòng sau.
    job2 = make_job(job_id="job_zzzzzzz9")
    setup_drive(tmp_path, job2, lyrics="xin chào\n")
    worker.run_forever(max_iterations=1)
    assert read_json(completed_path(root, job2["id"]))["status"] == "completed"


def test_user_lyrics_passed_when_has_lyrics_input(tmp_path, monkeypatch):
    job_true = make_job(job_id="job_aaaaaaa1", **{"input.has_lyrics_input": True})
    root = setup_drive(tmp_path, job_true, lyrics="Lời tiếng Việt có dấu\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    call = calls_of("transcriber.transcribe_and_align")[0]
    assert call["user_lyrics"] == "Lời tiếng Việt có dấu\n"

    job_false = make_job(job_id="job_bbbbbbb2", **{"input.has_lyrics_input": False})
    setup_drive(tmp_path, job_false)
    worker.run_forever(max_iterations=1)

    call2 = calls_of("transcriber.transcribe_and_align")[1]
    assert call2["user_lyrics"] is None


def test_has_lyrics_input_true_but_file_missing_warns_not_fails(tmp_path, monkeypatch):
    job = make_job(**{"input.has_lyrics_input": True})
    root = setup_drive(tmp_path, job, lyrics=None)  # cờ true nhưng không tạo file
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    call = calls_of("transcriber.transcribe_and_align")[0]
    assert call["user_lyrics"] is None
    assert read_json(completed_path(root, job["id"]))["status"] == "completed"


def test_models_from_job_config(tmp_path, monkeypatch):
    job = make_job(
        **{"config.demucs_model": "htdemucs", "config.whisper_model": "medium"}
    )
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert calls_of("separator.load_model")[0]["model_name"] == "htdemucs"
    assert calls_of("transcriber.transcribe_and_align")[0]["whisper_model"] == "medium"


def test_separator_unloaded_before_alignment(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    names = call_names()
    assert names.index("separator.unload") < names.index(
        "transcriber.transcribe_and_align"
    )


def test_render_config_is_copy_and_enriched(tmp_path, monkeypatch):
    job = make_job()
    original_config_keys = set(job["config"].keys())
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    cfg = calls_of("renderer.render")[0]["config"]
    assert cfg["fps"] == 30
    assert cfg["font_dir"] == Config.FONT_DIR
    assert cfg["ffmpeg_preset"] == Config.FFMPEG_PRESET
    assert cfg["ffmpeg_crf"] == Config.FFMPEG_CRF
    assert cfg["video_resolution"] == "1080p"
    # B4 bỏ REQ-P12 (font seam do PR-A4-FIX xử lý ở video/templates/base.py) —
    # worker truyền nguyên tên family, KHÔNG map sang tên file ở đây.
    assert cfg["font"] == "Be Vietnam Pro"

    data = read_json(completed_path(root, job["id"]))
    assert set(data["config"].keys()) == original_config_keys


def test_audio_duration_filled(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    monkeypatch.setattr(worker_module, "get_audio_duration", lambda path: 123.45)

    worker.run_forever(max_iterations=1)

    data = read_json(completed_path(root, job["id"]))
    assert data["input"]["audio_duration"] == 123.45


def test_audio_duration_stays_null_on_read_error(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    def boom(path):
        raise RuntimeError("no ffprobe")

    monkeypatch.setattr(worker_module, "get_audio_duration", boom)

    worker.run_forever(max_iterations=1)

    data = read_json(completed_path(root, job["id"]))
    assert data["input"]["audio_duration"] is None
    assert data["status"] == "completed"


def test_temp_dir_cleaned_after_success_and_failure(tmp_path, monkeypatch):
    job_ok = make_job(job_id="job_ccccccc3")
    root = setup_drive(tmp_path, job_ok, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)
    assert not (tmp_path / "temp" / job_ok["id"]).exists()

    job_fail = make_job(job_id="job_ddddddd4")
    setup_drive(tmp_path, job_fail, lyrics="xin chào\n")
    CONTROL["transcriber_raise"] = RuntimeError("boom")

    worker.run_forever(max_iterations=1)
    assert not (tmp_path / "temp" / job_fail["id"]).exists()


def test_no_unexpected_files_published(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    out_dir = Path(root) / "outputs" / job["id"]
    names = {p.name for p in out_dir.iterdir()}
    allowed = {"vocals.wav", "instrumental.wav", "lyrics_aligned.json", "karaoke_final.mp4"}
    assert names <= allowed
    assert not any(n.endswith(".part") for n in names)
    assert "instrumental_guide.wav" not in names


# ======================================================================
# heartbeat
# ======================================================================


def test_heartbeat_updated_during_long_stage(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root, heartbeat_interval=0.01)
    proc_path = processing_path(root, job["id"])
    result: dict = {}

    def hook():
        initial = read_json_retry(proc_path)["progress"]["heartbeat_at"]

        def changed() -> bool:
            try:
                data = read_json(proc_path)
            except (OSError, json.JSONDecodeError):
                return False
            return data["progress"]["heartbeat_at"] != initial

        result["changed"] = wait_until(changed, timeout=3.0, interval=0.02)
        snapshot = read_json_retry(proc_path)
        result["stage_status"] = snapshot["progress"]["stages"]["audio_separation"]["status"]
        result["started_at_mid"] = snapshot["progress"]["started_at"]

    CONTROL["separator_before_return"] = hook

    worker.run_forever(max_iterations=1)

    assert result["changed"] is True
    assert result["stage_status"] == "running"

    data = read_json(completed_path(root, job["id"]))
    assert data["progress"]["started_at"] == result["started_at_mid"]


def test_heartbeat_thread_stops_before_mark_completed(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root, heartbeat_interval=0.01)

    worker.run_forever(max_iterations=1)

    proc_dir = Path(root) / "queue" / "processing"
    assert list(proc_dir.glob("*.json")) == []

    time.sleep(worker.heartbeat_interval * 10)
    assert list(proc_dir.glob("*.json")) == []
    assert [t for t in threading.enumerate() if t.name.startswith("hb-")] == []


def test_heartbeat_thread_stops_when_stage_raises(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root, heartbeat_interval=0.01)
    CONTROL["transcriber_raise"] = RuntimeError("boom")

    worker.run_forever(max_iterations=1)

    proc_dir = Path(root) / "queue" / "processing"
    assert list(proc_dir.glob("*.json")) == []

    time.sleep(worker.heartbeat_interval * 10)
    assert list(proc_dir.glob("*.json")) == []
    assert [t for t in threading.enumerate() if t.name.startswith("hb-")] == []
    assert read_json(failed_path(root, job["id"]))["status"] == "failed"


def test_render_progress_callback_updates_job(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    proc_path = processing_path(root, job["id"])
    CONTROL["renderer_progress_calls"] = [10, 55.5]

    seen: dict = {}

    def hook():
        snapshot = read_json_retry(proc_path)
        seen["progress"] = snapshot["progress"]["stages"]["video_render"]["progress"]
        seen["current_stage"] = snapshot["progress"]["current_stage"]
        seen["heartbeat_at"] = snapshot["progress"]["heartbeat_at"]

    CONTROL["renderer_after_progress"] = hook

    worker.run_forever(max_iterations=1)

    assert seen["progress"] == 55.5
    assert seen["current_stage"] == "video_render"
    assert seen["heartbeat_at"] is not None


def test_concurrent_writes_keep_json_valid(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root, heartbeat_interval=0.001)
    proc_path = processing_path(root, job["id"])

    n = 200
    CONTROL["renderer_progress_calls"] = [i / n * 100 for i in range(1, n + 1)]
    read_errors: list[Exception] = []

    def progress_hook(i, _pct):
        if i % 20 != 0:
            return
        try:
            data = read_json(proc_path)
        except (OSError, json.JSONDecodeError) as exc:
            read_errors.append(exc)
            return
        validate(data)

    CONTROL["renderer_progress_hook"] = progress_hook

    worker.run_forever(max_iterations=1)

    assert read_errors == []
    validate(read_json(completed_path(root, job["id"])))


# ======================================================================
# guide vocal & cleanup
# ======================================================================


def test_guide_vocal_mix_runs_before_render(tmp_path, monkeypatch):
    job = make_job(**{"config.guide_vocal": True})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg"):
            return real_run(cmd, *args, **kwargs)
        record_call("ffmpeg.mix", cmd=cmd)
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"FAKE-MIXED-WAV")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)

    worker.run_forever(max_iterations=1)

    names = call_names()
    assert names.index("ffmpeg.mix") < names.index("renderer.render")
    render_call = calls_of("renderer.render")[0]
    assert render_call["instrumental_path"].endswith("instrumental_guide.wav")


def test_guide_vocal_ffmpeg_command_shape(tmp_path, monkeypatch):
    job = make_job(**{"config.guide_vocal": True, "config.guide_vocal_volume": 0.15})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    captured: dict = {}
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg"):
            return real_run(cmd, *args, **kwargs)
        captured["cmd"] = cmd
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"X")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)

    worker.run_forever(max_iterations=1)

    cmd = captured["cmd"]
    joined = " ".join(cmd)
    assert "volume=0.15" in joined
    assert "amix=inputs=2" in joined
    assert "normalize=0" in joined
    assert "duration=first" in joined
    i_indices = [idx for idx, val in enumerate(cmd) if val == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[0] + 1].endswith("instrumental.wav")
    assert cmd[i_indices[1] + 1].endswith("vocals.wav")


def test_guide_vocal_disabled_no_ffmpeg_call(tmp_path, monkeypatch):
    job = make_job(**{"config.guide_vocal": False})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    called: list = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg":
            called.append(cmd)
            raise AssertionError("ffmpeg không được gọi khi guide_vocal=false")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)

    worker.run_forever(max_iterations=1)

    assert called == []
    render_call = calls_of("renderer.render")[0]
    assert render_call["instrumental_path"].endswith("instrumental.wav")
    assert not render_call["instrumental_path"].endswith("instrumental_guide.wav")


def test_guide_vocal_ffmpeg_failure_fails_job(tmp_path, monkeypatch):
    job = make_job(**{"config.guide_vocal": True})
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg"):
            return real_run(cmd, *args, **kwargs)
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"mix failed: bad filter")

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)

    worker.run_forever(max_iterations=1)

    data = read_json(failed_path(root, job["id"]))
    assert data["status"] == "failed"
    assert "mix failed" in data["error"]
    assert not output_path(root, job["id"], "karaoke_final.mp4").exists()


def test_cleanup_runs_after_render_publish(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)

    worker.run_forever(max_iterations=1)

    assert output_path(root, job["id"], "karaoke_final.mp4").is_file()
    assert not output_path(root, job["id"], "vocals.wav").exists()
    assert output_path(root, job["id"], "instrumental.wav").is_file()


def test_cleanup_not_called_when_render_fails(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    CONTROL["renderer_raise"] = RuntimeError("ffmpeg crash")

    worker.run_forever(max_iterations=1)

    data = read_json(failed_path(root, job["id"]))
    assert data["status"] == "failed"
    assert output_path(root, job["id"], "vocals.wav").is_file()


# ======================================================================
# vòng lặp & shutdown
# ======================================================================


def test_recover_stale_called_every_poll_before_claim(tmp_path, monkeypatch):
    root = str(tmp_path / "drive")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    order: list[str] = []
    original_recover = worker.queue.recover_stale_jobs
    original_claim = worker.queue.poll_and_claim

    def spy_recover():
        order.append("recover")
        return original_recover()

    def spy_claim(worker_id):
        order.append("claim")
        return original_claim(worker_id)

    monkeypatch.setattr(worker.queue, "recover_stale_jobs", spy_recover)
    monkeypatch.setattr(worker.queue, "poll_and_claim", spy_claim)

    worker.run_forever(max_iterations=3)

    assert order == ["recover", "claim"] * 3


def test_idle_loop_sleeps_poll_interval(tmp_path, monkeypatch):
    root = str(tmp_path / "drive")
    worker = make_worker(tmp_path, monkeypatch, root=root, poll_interval=15)

    worker.run_forever(max_iterations=2)

    assert worker.sleep_calls == [15, 15]


def test_failure_does_not_kill_loop(tmp_path, monkeypatch):
    job_a = make_job(job_id="job_eeeeeee5")
    root = setup_drive(tmp_path, job_a, with_audio=False)  # sẽ fail (thiếu audio)
    job_b = make_job(job_id="job_fffffff6")
    setup_drive(tmp_path, job_b, lyrics="xin chào\n")

    worker = make_worker(tmp_path, monkeypatch, root=root)
    worker.run_forever(max_iterations=2)

    assert read_json(failed_path(root, job_a["id"]))["status"] == "failed"
    assert read_json(completed_path(root, job_b["id"]))["status"] == "completed"
    assert worker.jobs_done == 1


def test_mark_completed_error_does_not_kill_loop(tmp_path, monkeypatch):
    job_a = make_job(job_id="job_ggggggg7")
    root = setup_drive(tmp_path, job_a, lyrics="xin chào\n")
    job_b = make_job(job_id="job_hhhhhhh8")
    setup_drive(tmp_path, job_b, lyrics="xin chào\n")

    worker = make_worker(tmp_path, monkeypatch, root=root)
    original_mark_completed = worker.queue.mark_completed
    state = {"count": 0}

    def flaky_mark_completed(job):
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("drive hiccup")
        return original_mark_completed(job)

    monkeypatch.setattr(worker.queue, "mark_completed", flaky_mark_completed)

    worker.run_forever(max_iterations=2)

    assert read_json(completed_path(root, job_b["id"]))["status"] == "completed"
    assert worker.jobs_done == 1
    assert processing_path(root, job_a["id"]).is_file()  # job_a kẹt processing/, không crash


def test_keyboard_interrupt_exits_cleanly(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    CONTROL["transcriber_raise"] = KeyboardInterrupt()

    worker.run_forever(max_iterations=1)  # KHÔNG được raise ra ngoài

    assert processing_path(root, job["id"]).is_file()
    data = read_json(processing_path(root, job["id"]))
    assert data["checkpoints"]["audio_separated"] is True
    assert not failed_path(root, job["id"]).exists()
    assert not completed_path(root, job["id"]).exists()
    assert [t for t in threading.enumerate() if t.name.startswith("hb-")] == []
    assert not (tmp_path / "temp" / job["id"]).exists()


def test_keyboard_interrupt_in_idle_sleep_exits(tmp_path, monkeypatch):
    root = str(tmp_path / "drive")

    def sleep_raises(seconds):
        raise KeyboardInterrupt()

    worker = make_worker(tmp_path, monkeypatch, root=root, sleep_fn=sleep_raises)

    worker.run_forever()  # queue rỗng -> sleep_fn raise -> phải return sạch


def test_current_job_and_jobs_done_tracking(tmp_path, monkeypatch):
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    observed: dict = {}

    def hook():
        observed["current_job"] = worker.current_job

    CONTROL["separator_before_return"] = hook

    assert worker.current_job is None
    worker.run_forever(max_iterations=1)

    assert observed["current_job"] == job["id"]
    assert worker.current_job is None
    assert worker.jobs_done == 1


def test_lost_claim_returns_none_loop_continues(tmp_path, monkeypatch):
    root = str(tmp_path / "drive")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    monkeypatch.setattr(worker.queue, "poll_and_claim", lambda worker_id: None)

    worker.run_forever(max_iterations=2)

    assert worker.sleep_calls == [worker.poll_interval, worker.poll_interval]


def test_lost_claim_error_abandons_job_without_mark(tmp_path, monkeypatch):
    """DECISIONS-WAVE2.md PR-B4: 'B4 bắt LostClaimError từ DriveQueue → bỏ job
    đang xử lý' — không mark_failed, không mark_completed, job ở lại
    processing/ cho tới khi worker khác (chủ mới) xử lý xong."""
    job = make_job()
    root = setup_drive(tmp_path, job, lyrics="xin chào\n")
    worker = make_worker(tmp_path, monkeypatch, root=root)
    CONTROL["transcriber_raise"] = LostClaimError("mất quyền sở hữu (test)")

    worker.run_forever(max_iterations=1)

    assert not failed_path(root, job["id"]).exists()
    assert not completed_path(root, job["id"]).exists()
    assert worker.jobs_done == 0


# ======================================================================
# Config.auto_select_models
# ======================================================================


@pytest.fixture
def restore_config_models():
    saved = (
        Config.DEFAULT_DEMUCS_MODEL,
        Config.DEFAULT_WHISPER_MODEL,
        Config.WHISPER_COMPUTE_TYPE,
    )
    yield
    (
        Config.DEFAULT_DEMUCS_MODEL,
        Config.DEFAULT_WHISPER_MODEL,
        Config.WHISPER_COMPUTE_TYPE,
    ) = saved


@pytest.mark.parametrize(
    "available,vram,expected",
    [
        (True, 15.8, ("htdemucs_ft", "large-v3", "float16")),
        (True, 10, ("htdemucs_ft", "medium", "float16")),
        (True, 6, ("htdemucs", "small", "int8")),
        (True, 2, ("htdemucs", "tiny", "int8")),
        (False, 0, ("htdemucs", "small", "int8")),
    ],
)
def test_auto_select_by_vram(monkeypatch, restore_config_models, available, vram, expected):
    monkeypatch.setattr(
        gpu_module,
        "detect_gpu",
        lambda: {"available": available, "name": "fake-gpu", "vram_gb": vram},
    )
    Config.auto_select_models()
    assert (
        Config.DEFAULT_DEMUCS_MODEL,
        Config.DEFAULT_WHISPER_MODEL,
        Config.WHISPER_COMPUTE_TYPE,
    ) == expected


def test_auto_select_is_idempotent(monkeypatch, restore_config_models):
    monkeypatch.setattr(
        gpu_module,
        "detect_gpu",
        lambda: {"available": True, "name": "fake-gpu", "vram_gb": 10},
    )
    Config.auto_select_models()
    first = (
        Config.DEFAULT_DEMUCS_MODEL,
        Config.DEFAULT_WHISPER_MODEL,
        Config.WHISPER_COMPUTE_TYPE,
    )
    Config.auto_select_models()
    second = (
        Config.DEFAULT_DEMUCS_MODEL,
        Config.DEFAULT_WHISPER_MODEL,
        Config.WHISPER_COMPUTE_TYPE,
    )
    assert first == second


def test_auto_select_does_not_touch_protocol_constants(monkeypatch, restore_config_models):
    monkeypatch.setattr(
        gpu_module,
        "detect_gpu",
        lambda: {"available": True, "name": "fake-gpu", "vram_gb": 15.8},
    )
    Config.auto_select_models()
    assert Config.CLAIM_SETTLE_S == 5
    assert Config.STALE_AFTER_MIN == 10
    assert Config.MAX_ATTEMPTS == 3
    assert Config.POLL_INTERVAL == 15
    assert Config.HEARTBEAT_INTERVAL == 60


def test_config_module_has_no_torch_and_no_total_mem():
    src_path = (
        Path(__file__).resolve().parents[1] / "karaokeforge" / "config.py"
    )
    src = src_path.read_text(encoding="utf-8")
    assert "total_mem" not in src.replace("total_memory", "")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            pytest.fail(f"import torch ở top-level config.py: {line!r}")


def test_auto_select_survives_detect_gpu_error(monkeypatch, restore_config_models):
    def boom():
        raise RuntimeError("no cuda driver")

    monkeypatch.setattr(gpu_module, "detect_gpu", boom)
    Config.auto_select_models()  # không được raise
    assert Config.DEFAULT_DEMUCS_MODEL
    assert Config.DEFAULT_WHISPER_MODEL
    assert Config.WHISPER_COMPUTE_TYPE


# ======================================================================
# notebook (static, không chạy Colab)
# ======================================================================

_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks" / "KaraokeForge_Worker.ipynb"
)


def _load_notebook() -> dict:
    with open(_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def _full_text(notebook: dict) -> str:
    return "\n".join(_cell_text(c) for c in notebook["cells"])


def test_notebook_is_valid_and_has_ten_cells():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 10
    assert notebook["cells"][0]["cell_type"] == "markdown"
    for cell in notebook["cells"][1:]:
        assert cell["cell_type"] == "code"


def test_notebook_font_source_is_google_fonts():
    text = _full_text(_load_notebook())
    assert "bettergoogle" not in text
    assert "google/fonts" in text
    assert "BeVietnamPro-Bold.ttf" in text
    assert "BeVietnamPro-Regular.ttf" in text
    assert "/usr/share/fonts/custom" in text


def test_install_cell_uses_requirements_file():
    notebook = _load_notebook()
    cell3 = _cell_text(notebook["cells"][2])
    assert "requirements.txt" in cell3
    assert "apt-get" in cell3
    assert "ffmpeg" in cell3
    assert not ("pip install" in cell3 and "torch==" in cell3)


def test_notebook_has_no_prd_bugs():
    text = _full_text(_load_notebook())
    assert "total_mem" not in text.replace("total_memory", "")
    assert "STALE_TIMEOUT" not in text
    assert "timeout_minutes" not in text
    assert "done/" not in text


def test_notebook_creates_contract_folders():
    notebook = _load_notebook()
    cell4 = _cell_text(notebook["cells"][3])
    for name in ("pending", "processing", "completed", "failed", "uploads", "outputs", "models_cache"):
        assert name in cell4


def test_start_worker_cell():
    notebook = _load_notebook()
    cell9 = _cell_text(notebook["cells"][8])
    assert "KaraokeWorker(" in cell9
    assert "run_forever()" in cell9


def test_notebook_outputs_are_clean():
    notebook = _load_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None


def test_notebook_worker_id_and_drive_root_match_contract():
    notebook = _load_notebook()
    cell2 = _cell_text(notebook["cells"][1])
    assert "/content/drive/MyDrive/KaraokeForge" in cell2
    assert "STALE_AFTER_MIN" in cell2
    assert "10" in cell2
