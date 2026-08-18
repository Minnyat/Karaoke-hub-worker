"""Test `KaraokeRenderer.render` + `video/frame_generator.py`.

PR-A4. Toàn bộ test dùng mock `subprocess.Popen` + `get_audio_duration` +
`get_template` — không cần FFmpeg binary thật, không phụ thuộc PR-A1 đã xong.
Tham chiếu ID theo docs/plans/PR-A4.md mục 5.2.
"""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from karaokeforge.pipeline.renderer import KaraokeRenderer, RenderError

from .helpers_a4 import FakeTemplate, RaisingTemplate, lyrics_fixture, make_fake_process, renderer_config


@contextmanager
def patched_render(popen_side_effect=None, popen_return=None, duration: float = 0.1):
    """Patch Popen/get_audio_duration/get_template cho renderer test.

    `get_template` luôn dựng `FakeTemplate` với đúng (width, height) mà renderer
    tính ra (tránh mismatch shape khi resolution thay đổi giữa test).
    """
    templates_created: list[FakeTemplate] = []

    def _make_template(name, width, height, config):
        t = FakeTemplate(width, height, config)
        templates_created.append(t)
        return t

    with patch("karaokeforge.pipeline.renderer.subprocess.Popen") as mock_popen, patch(
        "karaokeforge.pipeline.renderer.get_audio_duration", return_value=duration
    ) as mock_duration, patch(
        "karaokeforge.pipeline.renderer.get_template", side_effect=_make_template
    ) as mock_get_template:
        if popen_side_effect is not None:
            mock_popen.side_effect = popen_side_effect
        else:
            mock_popen.return_value = popen_return if popen_return is not None else make_fake_process(0)
        yield mock_popen, mock_get_template, templates_created, mock_duration


@pytest.fixture
def tiny_resolution(monkeypatch):
    """Thêm resolution 64x36 vào RESOLUTIONS — test nhiều frame (progress) nhanh,
    ít RAM, không cần render 1080p/4k thật."""
    from karaokeforge.pipeline import renderer as renderer_module

    monkeypatch.setitem(renderer_module.RESOLUTIONS, "tiny", (64, 36))
    return "tiny", 64, 36


# ---------------------------------------------------------------------------
# REQ-A4-01/02 — signature, không phải generator
# ---------------------------------------------------------------------------


def test_r01_returns_output_path(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        result = KaraokeRenderer().render("instr.wav", lyrics_fixture(), output_path, renderer_config())
    assert result == output_path


def test_r02_render_not_generator(tmp_path):
    assert inspect.isgeneratorfunction(KaraokeRenderer.render) is False
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        result = KaraokeRenderer().render("instr.wav", [], output_path, renderer_config())
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# REQ-A4-03/06/07 — argv FFmpeg, resolution, fps, duration
# ---------------------------------------------------------------------------


def test_r03_ffmpeg_argv_order(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        KaraokeRenderer().render("instrumental.wav", [], output_path, renderer_config(video_resolution="720p"))
    cmd = mock_popen.call_args.args[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24"
    i_indices = [i for i, v in enumerate(cmd) if v == "-i"]
    assert cmd[i_indices[0] + 1] == "-"
    assert cmd[i_indices[1] + 1] == "instrumental.wav"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[-1] == output_path


@pytest.mark.parametrize(
    "res_key,expected",
    [("720p", "1280x720"), ("1080p", "1920x1080"), ("4k", "3840x2160"), (None, "1920x1080")],
)
def test_r04_resolution_mapping(tmp_path, res_key, expected):
    output_path = str(tmp_path / "out.mp4")
    config = renderer_config()
    if res_key is None:
        config.pop("video_resolution", None)
    else:
        config["video_resolution"] = res_key
    with patched_render() as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, config)
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("-s") + 1] == expected


def test_r05_resolution_uppercase_4k(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, renderer_config(video_resolution="4K"))
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("-s") + 1] == "3840x2160"


def test_r06_invalid_resolution_raises(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        with pytest.raises(ValueError):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config(video_resolution="9000p"))
    mock_popen.assert_not_called()


def test_r07_default_fps(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("-r") + 1] == "30"


def test_r15_get_audio_duration_called_once(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, mock_duration):
        KaraokeRenderer().render("instrumental.wav", [], output_path, renderer_config())
    mock_duration.assert_called_once_with("instrumental.wav")


def test_r16_very_short_duration(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render(duration=0.01) as (mock_popen, _, templates_created, _):
        result = KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert result == output_path
    assert len(templates_created[0].calls) == 1


# ---------------------------------------------------------------------------
# REQ-A4-08 — frame bytes
# ---------------------------------------------------------------------------


def test_r08_total_bytes_written(tmp_path, tiny_resolution):
    key, w, h = tiny_resolution
    output_path = str(tmp_path / "out.mp4")
    fake_proc = make_fake_process(0)
    with patched_render(popen_return=fake_proc, duration=2.0) as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, renderer_config(video_resolution=key))
    total_written = sum(len(c.args[0]) for c in fake_proc.stdin.write.call_args_list)
    total_frames = max(1, int(2.0 * 30))
    assert total_written == total_frames * w * h * 3


def test_r27_frame_to_bytes_invalid_shape():
    from karaokeforge.video.frame_generator import frame_to_bytes

    with pytest.raises(ValueError):
        frame_to_bytes(np.zeros((10, 10, 4), dtype=np.uint8), 10, 10)
    with pytest.raises(ValueError):
        frame_to_bytes(np.zeros((10, 10, 3), dtype=np.float32), 10, 10)


def test_r28_frame_to_bytes_noncontiguous():
    from karaokeforge.video.frame_generator import frame_to_bytes

    base = np.zeros((3, 10, 20), dtype=np.uint8)
    noncontig = np.transpose(base, (1, 2, 0))  # shape (10, 20, 3), không liền kề
    assert not noncontig.flags["C_CONTIGUOUS"]
    data = frame_to_bytes(noncontig, 20, 10)
    assert len(data) == 10 * 20 * 3


def test_r29_iter_frame_bytes_count_and_timestamp():
    from karaokeforge.video.frame_generator import iter_frame_bytes

    template = FakeTemplate(4, 3, {})
    results = list(iter_frame_bytes(template, [], 5, 10, 4, 3))
    assert len(results) == 5
    assert [idx for idx, _ in results] == [0, 1, 2, 3, 4]
    assert [ts for ts, _ in template.calls] == pytest.approx([i / 10 for i in range(5)])
    assert all(len(chunk) == 4 * 3 * 3 for _, chunk in results)


# ---------------------------------------------------------------------------
# REQ-A4-04 — đóng stdin trước wait()
# ---------------------------------------------------------------------------


def test_r09_stdin_close_before_wait(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    fake_proc = make_fake_process(returncode=0)
    manager = Mock()
    manager.attach_mock(fake_proc.stdin.close, "stdin_close")
    manager.attach_mock(fake_proc.wait, "wait")
    with patched_render(popen_return=fake_proc) as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    names = [c[0] for c in manager.mock_calls]
    assert "stdin_close" in names and "wait" in names
    assert names.index("stdin_close") < names.index("wait")


def test_r10_stdin_closed_even_if_template_raises(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    fake_proc = make_fake_process(returncode=0)
    raising_template = RaisingTemplate(1280, 720, {}, raise_after=1)
    with patch("karaokeforge.pipeline.renderer.subprocess.Popen", return_value=fake_proc), patch(
        "karaokeforge.pipeline.renderer.get_audio_duration", return_value=1.0
    ), patch("karaokeforge.pipeline.renderer.get_template", return_value=raising_template):
        with pytest.raises(RuntimeError, match="boom trong template"):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config(video_resolution="720p"))
    fake_proc.stdin.close.assert_called_once()


# ---------------------------------------------------------------------------
# REQ-A4-02 — on_progress callback
# ---------------------------------------------------------------------------


def test_r11_r12_r13_progress_callback_timing(tmp_path, tiny_resolution):
    key, w, h = tiny_resolution
    output_path = str(tmp_path / "out.mp4")
    percents: list[float] = []
    with patched_render(duration=20.0) as (mock_popen, _, _, _):
        KaraokeRenderer().render(
            "i.wav", [], output_path, renderer_config(video_resolution=key), on_progress=percents.append
        )
    assert percents[:4] == pytest.approx([0.0, 25.0, 50.0, 75.0])
    assert percents[-1] == 100.0
    assert all(a <= b for a, b in zip(percents, percents[1:]))
    assert all(0.0 <= p <= 100.0 for p in percents)


def test_r14_on_progress_none(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        result = KaraokeRenderer().render("i.wav", [], output_path, renderer_config(), on_progress=None)
    assert result == output_path


def test_r30_on_progress_exception_not_swallowed_no_retry(tmp_path):
    output_path = str(tmp_path / "out.mp4")

    def bad_progress(percent):
        raise ValueError("boom from on_progress")

    with patched_render() as (mock_popen, _, _, _):
        with pytest.raises(ValueError, match="boom from on_progress"):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config(), on_progress=bad_progress)
    assert mock_popen.call_count == 1


# ---------------------------------------------------------------------------
# REQ-A4-05 — retry / crash handling
# ---------------------------------------------------------------------------


def test_r17_retry_success_on_second_attempt(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    call_count = {"n": 0}

    def popen_side_effect(cmd, stdin=None, stderr=None):
        call_count["n"] += 1
        return make_fake_process(returncode=1 if call_count["n"] == 1 else 0)

    with patched_render(popen_side_effect=popen_side_effect) as (mock_popen, _, _, _):
        result = KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert result == output_path
    assert mock_popen.call_count == 2


def test_r18_retry_both_fail_raises(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render(popen_return=make_fake_process(returncode=1)) as (mock_popen, _, _, _):
        with pytest.raises(RenderError):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert mock_popen.call_count == 2


def test_r19_broken_pipe_triggers_retry(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    call_count = {"n": 0}

    def popen_side_effect(cmd, stdin=None, stderr=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return make_fake_process(returncode=0, write_side_effect=BrokenPipeError("pipe closed"))
        return make_fake_process(returncode=0)

    with patched_render(popen_side_effect=popen_side_effect) as (mock_popen, _, _, _):
        result = KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert result == output_path
    assert mock_popen.call_count == 2


def test_r20_error_message_contains_stderr_tail(tmp_path):
    output_path = str(tmp_path / "out.mp4")

    def popen_side_effect(cmd, stdin=None, stderr=None):
        stderr.write(b"line1\nline2\nFFMPEG FAKE ERROR TAIL\n")
        stderr.flush()
        return make_fake_process(returncode=1)

    with patched_render(popen_side_effect=popen_side_effect) as (mock_popen, _, _, _):
        with pytest.raises(RenderError) as exc_info:
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert "FFMPEG FAKE ERROR TAIL" in str(exc_info.value)


def test_r21_partial_saved_before_retry(tmp_path):
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"fake partial video data")
    call_count = {"n": 0}

    def popen_side_effect(cmd, stdin=None, stderr=None):
        call_count["n"] += 1
        return make_fake_process(returncode=1 if call_count["n"] == 1 else 0)

    with patched_render(popen_side_effect=popen_side_effect) as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], str(output_path), renderer_config())

    partial_path = Path(str(output_path) + ".partial")
    assert partial_path.exists()
    assert partial_path.stat().st_size > 0


def test_r22_ffmpeg_binary_not_found_no_retry(tmp_path):
    output_path = str(tmp_path / "out.mp4")

    def popen_side_effect(cmd, stdin=None, stderr=None):
        raise FileNotFoundError("No such file or directory: 'ffmpeg'")

    with patched_render(popen_side_effect=popen_side_effect) as (mock_popen, _, _, _):
        with pytest.raises(RenderError, match="(?i)ffmpeg"):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert mock_popen.call_count == 1


def test_r31_mkstemp_filenotfound_not_mislabeled(tmp_path):
    """PR-A4-FIX (Low): FileNotFoundError từ `tempfile.mkstemp` KHÔNG được báo
    nhầm "thiếu ffmpeg" — chỉ Popen ENOENT mới thành RenderError ffmpeg."""
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, _, _, _):
        with patch(
            "karaokeforge.pipeline.renderer.tempfile.mkstemp",
            side_effect=FileNotFoundError("no temp dir"),
        ):
            with pytest.raises(FileNotFoundError, match="no temp dir"):
                KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    mock_popen.assert_not_called()


def test_r32_stdin_close_raises_becomes_rendererror_after_retry(tmp_path):
    """PR-A4-FIX (Critical a): `proc.stdin.close()` tự raise OSError (ffmpeg đã
    chết) không được lọt ra thô — phải gom vào crash và thành RenderError sau
    khi retry cạn."""
    output_path = str(tmp_path / "out.mp4")
    fake_proc = make_fake_process(returncode=0)
    fake_proc.stdin.close.side_effect = OSError("broken pipe on close")

    with patched_render(popen_return=fake_proc) as (mock_popen, _, _, _):
        with pytest.raises(RenderError):
            KaraokeRenderer().render("i.wav", [], output_path, renderer_config())
    assert mock_popen.call_count == 2


def test_r33_iter_frame_raises_kills_and_waits_ffmpeg(tmp_path):
    """PR-A4-FIX (Critical b): iter_frame_bytes raise giữa chừng KHÔNG được để
    ffmpeg mồ côi — phải gọi proc.kill() (khi còn sống) rồi proc.wait()."""
    output_path = str(tmp_path / "out.mp4")
    fake_proc = make_fake_process(returncode=0)
    fake_proc.poll.return_value = None  # ffmpeg vẫn đang chạy khi unwind
    raising_template = RaisingTemplate(1280, 720, {}, raise_after=1)

    with patch(
        "karaokeforge.pipeline.renderer.subprocess.Popen", return_value=fake_proc
    ), patch(
        "karaokeforge.pipeline.renderer.get_audio_duration", return_value=1.0
    ), patch(
        "karaokeforge.pipeline.renderer.get_template", return_value=raising_template
    ):
        with pytest.raises(RuntimeError, match="boom trong template"):
            KaraokeRenderer().render(
                "i.wav", [], output_path, renderer_config(video_resolution="720p")
            )

    fake_proc.kill.assert_called_once()
    fake_proc.wait.assert_called()
    # stdin đóng trước khi wait (không đổi contract REQ-A4-04)
    fake_proc.stdin.close.assert_called_once()


# ---------------------------------------------------------------------------
# REQ-A4-16/24 — get_template + default template
# ---------------------------------------------------------------------------


def test_r23_get_template_called_with_correct_args(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patched_render() as (mock_popen, mock_get_template, _, _):
        KaraokeRenderer().render(
            "i.wav", [], output_path, renderer_config(video_resolution="1080p", video_template="classic")
        )
    args = mock_get_template.call_args.args
    assert args[0] == "classic"
    assert args[1] == 1920
    assert args[2] == 1080


def test_r24_default_template_modern(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    config = renderer_config()
    config.pop("video_template", None)
    with patched_render() as (mock_popen, mock_get_template, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, config)
    assert mock_get_template.call_args.args[0] == "modern"


# ---------------------------------------------------------------------------
# REQ-A4-17 (không mutate caller) + edge case lyrics rỗng
# ---------------------------------------------------------------------------


def test_r25_does_not_mutate_config(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    config = renderer_config()
    before = copy.deepcopy(config)
    with patched_render() as (mock_popen, _, _, _):
        KaraokeRenderer().render("i.wav", [], output_path, config)
    assert config == before


def test_r26_empty_lyrics_still_renders(tmp_path, tiny_resolution):
    key, w, h = tiny_resolution
    output_path = str(tmp_path / "out.mp4")
    with patched_render(duration=1.0) as (mock_popen, _, templates_created, _):
        result = KaraokeRenderer().render("i.wav", [], output_path, renderer_config(video_resolution=key))
    assert result == output_path
    assert len(templates_created[0].calls) == max(1, int(1.0 * 30))
