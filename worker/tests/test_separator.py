"""Test cho karaokeforge.pipeline.separator.AudioSeparator (PR-A1).

Xem plan §5.3 (S1-S26) + edge case §5.4.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys

import pytest

from karaokeforge.config import Config
from tests.helpers_a1 import FakeAudioTensor, FakeStemTensor, install_fake_torch, make_wav


@pytest.fixture(autouse=True)
def _clean_torch_home_env():
    """An toàn: khôi phục TORCH_HOME sau mỗi test.

    Sau PR-PIPE-FIX (Fix 5) `load_model` KHÔNG còn ghi `os.environ["TORCH_HOME"]`
    toàn cục nữa (chuyển sang `torch.hub.set_dir` cục bộ), nên fixture này chỉ còn
    là lớp bảo hiểm phòng rò rỉ giữa các test."""
    original = os.environ.get("TORCH_HOME")
    yield
    if original is None:
        os.environ.pop("TORCH_HOME", None)
    else:
        os.environ["TORCH_HOME"] = original


@pytest.fixture(autouse=True)
def _default_models_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MODELS_CACHE", str(tmp_path / "models_cache"))


def _make_separator():
    from karaokeforge.pipeline.separator import AudioSeparator

    return AudioSeparator()


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_s1_load_model_default_calls_get_model_htdemucs_ft(monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()

    sep.load_model()

    fakes.get_model.assert_called_once_with("htdemucs_ft")


def test_s2_load_model_sets_hub_dir_without_touching_global_env(monkeypatch, tmp_path):
    """Fix 5: dùng torch.hub.set_dir cục bộ, KHÔNG ghi os.environ["TORCH_HOME"] toàn
    cục (nếu không, model align whisperx cũng rơi vào <cache>/demucs, lệch layout
    models_cache/{demucs,whisper} — contracts/README §2)."""
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()
    before = os.environ.get("TORCH_HOME")

    sep.load_model()

    expected_cache = os.path.join(Config.MODELS_CACHE, "demucs")
    assert os.path.isdir(expected_cache)
    fakes.set_dir.assert_called_once_with(os.path.join(expected_cache, "hub"))
    # Không đụng biến môi trường toàn cục.
    assert os.environ.get("TORCH_HOME") == before


def test_s3_load_model_calls_cuda_when_available(monkeypatch):
    install_fake_torch(monkeypatch, cuda=True)
    sep = _make_separator()

    sep.load_model()

    sep.model.cuda.assert_called_once()


def test_s3b_load_model_no_cuda_does_not_call_cuda_and_warns(monkeypatch, caplog):
    install_fake_torch(monkeypatch, cuda=False)
    sep = _make_separator()

    with caplog.at_level(logging.WARNING):
        sep.load_model()

    sep.model.cuda.assert_not_called()
    assert any("CUDA" in record.message for record in caplog.records)


def test_s4_load_model_twice_same_name_is_idempotent(monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()

    sep.load_model("htdemucs_ft")
    sep.load_model("htdemucs_ft")

    fakes.get_model.assert_called_once_with("htdemucs_ft")


def test_s5_load_model_different_name_unloads_and_reloads(monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()

    sep.load_model("htdemucs_ft")
    sep.load_model("htdemucs")

    assert fakes.get_model.call_count == 2
    fakes.get_model.assert_any_call("htdemucs_ft")
    fakes.get_model.assert_any_call("htdemucs")
    assert sep.model_name == "htdemucs"


# ---------------------------------------------------------------------------
# separate — happy path
# ---------------------------------------------------------------------------


def _input_wav(tmp_path, **kwargs):
    return make_wav(str(tmp_path / "input.wav"), seconds=0.2, **kwargs)


def test_s6_separate_returns_exact_keys_and_real_files(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    sep = _make_separator()
    audio_path = _input_wav(tmp_path)
    output_dir = str(tmp_path / "out")

    result = sep.separate(audio_path, output_dir)

    assert set(result.keys()) == {"vocals", "instrumental"}
    assert os.path.exists(result["vocals"])
    assert os.path.exists(result["instrumental"])
    assert os.path.basename(result["vocals"]) == "vocals.wav"
    assert os.path.basename(result["instrumental"]) == "instrumental.wav"


def test_s7_instrumental_is_drums_bass_other_not_by_index(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    captured = {}
    original_save = fakes.save.side_effect

    def _capture_save(path, tensor, sample_rate, *a, **k):
        if os.path.basename(path) == "instrumental.wav":
            captured["name"] = tensor.name
        return original_save(path, tensor, sample_rate, *a, **k)

    fakes.save.side_effect = _capture_save

    sep = _make_separator()
    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert captured["name"] == "drums+bass+other"


def test_s8_separate_auto_loads_default_model_when_none(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()
    assert sep.model is None

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    fakes.get_model.assert_called_once_with(Config.DEFAULT_DEMUCS_MODEL)


def test_s9_separate_creates_missing_output_dir(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    sep = _make_separator()
    output_dir = str(tmp_path / "does" / "not" / "exist")

    result = sep.separate(_input_wav(tmp_path), output_dir)

    assert os.path.isdir(output_dir)
    assert os.path.exists(result["vocals"])


def test_s10_resample_called_when_sr_differs(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch, samplerate=44100)
    fakes.load.return_value = (FakeAudioTensor(channels=2), 22050)
    sep = _make_separator()

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    fakes.resample.assert_called_once()


def test_s10b_resample_not_called_when_sr_matches(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch, samplerate=44100)
    fakes.load.return_value = (FakeAudioTensor(channels=2), 44100)
    sep = _make_separator()

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    fakes.resample.assert_not_called()


def test_s11_mono_input_expanded_to_model_channels(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch, audio_channels=2)
    fakes.load.return_value = (FakeAudioTensor(channels=1), 44100)
    sep = _make_separator()

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    passed_wav = fakes.apply_model.call_args.args[1]
    assert passed_wav.channels == 2


# ---------------------------------------------------------------------------
# OOM fallback
# ---------------------------------------------------------------------------


def _oom_then_ok(fakes, sources=("drums", "bass", "other", "vocals")):
    state = {"n": 0}

    def _apply(model, wav, split=True, overlap=0.25, device="cpu"):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return [[FakeStemTensor(s) for s in model.sources]]

    fakes.apply_model.side_effect = _apply
    return state


def test_s12_oom_once_falls_back_to_htdemucs_and_succeeds(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    state = _oom_then_ok(fakes)
    sep = _make_separator()

    result = sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert state["n"] == 2
    fakes.get_model.assert_any_call("htdemucs")
    fakes.empty_cache.assert_called()
    assert set(result.keys()) == {"vocals", "instrumental"}


def test_s13_oom_twice_raises_and_model_is_none(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    fakes.apply_model.side_effect = RuntimeError("CUDA out of memory")
    sep = _make_separator()

    with pytest.raises(RuntimeError, match="OOM"):
        sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert sep.model is None


def test_s14_already_htdemucs_oom_no_fallback(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()
    sep.load_model("htdemucs")
    fakes.apply_model.side_effect = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="OOM"):
        sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert fakes.apply_model.call_count == 1


def test_s15_non_oom_runtimeerror_reraises_unchanged(tmp_path, monkeypatch):
    fakes = install_fake_torch(monkeypatch)
    fakes.apply_model.side_effect = RuntimeError("shape mismatch")
    sep = _make_separator()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert fakes.get_model.call_count == 1


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------


def test_s16_silent_instrumental_warns_but_does_not_raise(tmp_path, monkeypatch, caplog):
    fakes = install_fake_torch(
        monkeypatch,
        stem_amplitudes={"drums": 0.0, "bass": 0.0, "other": 0.0, "vocals": 0.5},
    )
    sep = _make_separator()

    with caplog.at_level(logging.WARNING):
        result = sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert set(result.keys()) == {"vocals", "instrumental"}
    assert len(sep.warnings) == 1
    assert "không lời" in sep.warnings[0]
    assert any("không lời" in r.message for r in caplog.records)


def test_s17_normal_instrumental_has_no_warnings(tmp_path, monkeypatch):
    fakes = install_fake_torch(
        monkeypatch,
        stem_amplitudes={"drums": 0.4, "bass": 0.4, "other": 0.4, "vocals": 0.5},
    )
    sep = _make_separator()

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert sep.warnings == []


def test_s18_silent_vocals_warns_about_singing(tmp_path, monkeypatch):
    fakes = install_fake_torch(
        monkeypatch,
        stem_amplitudes={"drums": 0.4, "bass": 0.4, "other": 0.4, "vocals": 0.0},
    )
    sep = _make_separator()

    result = sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert set(result.keys()) == {"vocals", "instrumental"}
    assert len(sep.warnings) == 1
    assert "giọng hát" in sep.warnings[0]


def test_s19_warnings_reset_between_calls(tmp_path, monkeypatch):
    fakes = install_fake_torch(
        monkeypatch,
        stem_amplitudes={"drums": 0.0, "bass": 0.0, "other": 0.0, "vocals": 0.5},
    )
    sep = _make_separator()
    sep.separate(_input_wav(tmp_path), str(tmp_path / "out1"))
    assert len(sep.warnings) == 1

    fakes.apply_model.side_effect = lambda model, wav, split=True, overlap=0.25, device="cpu": [
        [FakeStemTensor(s, amplitude=0.5) for s in model.sources]
    ]
    sep.separate(_input_wav(tmp_path), str(tmp_path / "out2"))
    assert sep.warnings == []


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


def test_s20_unload_after_load_clears_model_and_empties_cache(monkeypatch):
    fakes = install_fake_torch(monkeypatch, cuda=True)
    sep = _make_separator()
    sep.load_model()

    sep.unload()

    assert sep.model is None
    fakes.empty_cache.assert_called_once()


def test_s21_unload_without_load_does_not_raise(monkeypatch):
    install_fake_torch(monkeypatch)
    sep = _make_separator()

    sep.unload()

    assert sep.model is None


def test_s22_unload_twice_does_not_raise(monkeypatch):
    fakes = install_fake_torch(monkeypatch, cuda=True)
    sep = _make_separator()
    sep.load_model()

    sep.unload()
    sep.unload()

    assert sep.model is None


def test_s23_unload_without_torch_does_not_raise(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    sep = _make_separator()

    sep.unload()

    assert sep.model is None


def test_s24_unload_no_cuda_does_not_call_empty_cache(monkeypatch):
    fakes = install_fake_torch(monkeypatch, cuda=False)
    sep = _make_separator()
    sep.load_model()

    sep.unload()

    fakes.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Guard import top-level + signature
# ---------------------------------------------------------------------------


def test_s25_import_succeeds_without_torch_or_demucs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    monkeypatch.setitem(sys.modules, "demucs", None)
    monkeypatch.delitem(sys.modules, "karaokeforge.pipeline.separator", raising=False)
    import karaokeforge.pipeline.separator  # noqa: F401


def test_s26_signatures_match_skeleton():
    from karaokeforge.pipeline.separator import AudioSeparator

    load_sig = inspect.signature(AudioSeparator.load_model)
    assert list(load_sig.parameters) == ["self", "model_name"]
    assert load_sig.parameters["model_name"].default == "htdemucs_ft"

    separate_sig = inspect.signature(AudioSeparator.separate)
    assert list(separate_sig.parameters) == ["self", "audio_path", "output_dir"]

    unload_sig = inspect.signature(AudioSeparator.unload)
    assert list(unload_sig.parameters) == ["self"]


# ---------------------------------------------------------------------------
# Edge case §5.4 #7 — model không có stem "vocals"
# ---------------------------------------------------------------------------


def test_missing_vocals_stem_raises_clear_runtimeerror(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch, model_sources=["vocals_alt", "no_vocals"])
    sep = _make_separator()

    with pytest.raises(RuntimeError, match="vocals"):
        sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# PR-PIPE-FIX Fix 1 [Critical] — torchaudio.save phải ghi PCM_S 16-bit
# ---------------------------------------------------------------------------


def test_pipefix1_save_uses_pcm_s_16bit(tmp_path, monkeypatch):
    """torchaudio.save mặc định ghi PCM_F 32-bit → wave.open raise 'unknown format: 3'
    + instrumental.wav phình gấp đôi. Mọi lần save phải truyền PCM_S/16-bit."""
    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()

    sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert fakes.save.call_count == 2
    for call in fakes.save.call_args_list:
        assert call.kwargs.get("encoding") == "PCM_S"
        assert call.kwargs.get("bits_per_sample") == 16


# ---------------------------------------------------------------------------
# PR-PIPE-FIX Fix 2 [Critical] — _check_silence nuốt lỗi (chỉ log, không fail stage)
# ---------------------------------------------------------------------------


def test_pipefix2_check_silence_swallows_rms_error(tmp_path, monkeypatch):
    """Cảnh báo im lặng chỉ log, không fail stage (DECISIONS PR-A1). Khi rms_level
    ném lỗi (file không đọc được + không có ffmpeg) → separate vẫn trả kết quả,
    chỉ log warning, KHÔNG fail stage.

    Kích hoạt lỗi rms_level THẬT (ghi file WAV hỏng qua torchaudio.save giả +
    che ffmpeg) thay vì monkeypatch tên module rms_level — vốn không bền vì test
    khác (s25/a17) reload module. `logger` là singleton theo tên nên bắt
    `logger.warning` vẫn ổn định."""
    import shutil

    import karaokeforge.pipeline.separator as sep_mod

    fakes = install_fake_torch(monkeypatch)
    sep = _make_separator()

    def _corrupt_save(path, tensor, sr, *args, **kwargs):
        with open(path, "wb") as f:
            f.write(b"RIFF....WAVEfmt not a real pcm header")

    fakes.save.side_effect = _corrupt_save
    monkeypatch.setattr(shutil, "which", lambda name: None)  # không có ffmpeg

    logged: list[str] = []

    def _capture_warning(msg, *args, **kwargs):
        logged.append(msg % args if args else str(msg))

    monkeypatch.setattr(sep_mod.logger, "warning", _capture_warning)

    result = sep.separate(_input_wav(tmp_path), str(tmp_path / "out"))

    assert set(result.keys()) == {"vocals", "instrumental"}
    assert any("RMS" in m for m in logged)


# ---------------------------------------------------------------------------
# PR-PIPE-FIX Fix 3 [Medium] — normalize chia 0 khi audio im lặng/DC
# ---------------------------------------------------------------------------


def test_pipefix3_zero_std_replaced_with_one(tmp_path, monkeypatch):
    """input DC/im lặng → ref.std()=0 → NaN toàn stem. Guard: std<1e-8 → 1.0."""
    fakes = install_fake_torch(monkeypatch)
    fakes.load.return_value = (FakeAudioTensor(channels=2, std_value=0.0), 44100)
    sep = _make_separator()
    sep.load_model()

    stems, _names, _sr = sep._apply(_input_wav(tmp_path))

    assert stems[0].mul_operand == 1.0


def test_pipefix3_nonzero_std_passthrough(tmp_path, monkeypatch):
    """std bình thường không bị guard đụng vào."""
    fakes = install_fake_torch(monkeypatch)
    fakes.load.return_value = (FakeAudioTensor(channels=2, std_value=0.5), 44100)
    sep = _make_separator()
    sep.load_model()

    stems, _names, _sr = sep._apply(_input_wav(tmp_path))

    assert stems[0].mul_operand == 0.5
