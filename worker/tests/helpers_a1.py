"""Test helpers riêng cho PR-A1 (separator + utils/gpu + utils/audio).

KHÔNG chứa logic sản phẩm — chỉ dùng trong worker/tests/test_*.py thuộc A1.
Theo quyết định G1 (docs/plans/DECISIONS.md): không có conftest.py dùng chung
trong wave 1, mỗi PR namespace helper riêng.
"""

from __future__ import annotations

import math
import struct
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock
import wave

import numpy as np


# ---------------------------------------------------------------------------
# WAV fixtures thật (stdlib wave — không cần soundfile)
# ---------------------------------------------------------------------------

_SAMPWIDTH_MAX = {1: 127, 2: 32767, 3: 8388607, 4: 2147483647}


def make_wav(
    path: str,
    seconds: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
    sampwidth: int = 2,
    amplitude: float = 0.5,
    waveform: str = "sine",
    channel_amplitudes: Sequence[float] | None = None,
) -> str:
    """Sinh file WAV PCM thật bằng stdlib `wave` (không cần soundfile).

    amplitude: biên độ tỉ lệ full-scale trong [0, 1] (áp dụng đều cho mọi kênh).
    channel_amplitudes: nếu truyền, ghi đè amplitude riêng cho từng kênh
        (dùng để test stereo bất đối xứng, ví dụ A12).
    waveform: "sine" | "square" | "silence".
    """
    if sampwidth not in _SAMPWIDTH_MAX:
        raise ValueError(f"sampwidth không hỗ trợ: {sampwidth}")

    amps = list(channel_amplitudes) if channel_amplitudes is not None else [amplitude] * channels
    if len(amps) != channels:
        raise ValueError("channel_amplitudes phải có đúng độ dài channels")

    nframes = max(int(round(seconds * sample_rate)), 0)
    freq = 440.0
    t = np.arange(nframes, dtype=np.float64) / sample_rate if nframes else np.zeros(0)

    def _gen(a: float) -> np.ndarray:
        if waveform == "silence" or a == 0.0:
            return np.zeros(nframes, dtype=np.float64)
        if waveform == "square":
            return a * np.sign(np.sin(2 * math.pi * freq * t))
        return a * np.sin(2 * math.pi * freq * t)

    channel_data = [_gen(a) for a in amps]
    interleaved = np.empty(nframes * channels, dtype=np.float64)
    for i, data in enumerate(channel_data):
        interleaved[i::channels] = data

    max_val = _SAMPWIDTH_MAX[sampwidth]
    if sampwidth == 1:
        raw = np.clip(np.round(interleaved * max_val) + 128, 0, 255).astype(np.uint8)
        byte_data = raw.tobytes()
    elif sampwidth == 2:
        raw = np.clip(np.round(interleaved * max_val), -32768, 32767).astype("<i2")
        byte_data = raw.tobytes()
    elif sampwidth == 3:
        raw = np.clip(np.round(interleaved * max_val), -8388608, 8388607).astype(np.int64)
        byte_data = b"".join(
            struct.pack("<i", int(v))[:3] for v in raw
        )
    else:  # sampwidth == 4
        raw = np.clip(np.round(interleaved * max_val), -2147483648, 2147483647).astype("<i4")
        byte_data = raw.tobytes()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(byte_data)
    return path


# ---------------------------------------------------------------------------
# Fake tensors dùng trong pipeline._apply (không cần torch thật)
# ---------------------------------------------------------------------------


class FakeAudioTensor:
    """Giả lập torch.Tensor tối thiểu cho `wav` trong `AudioSeparator._apply`.

    Chỉ track (channels, samples) đủ để assert resample/channel-matching;
    không tính toán số học thật (không cần cho unit test — xem DoD §B).
    """

    def __init__(
        self,
        channels: int,
        samples: int = 100,
        mean_value: float = 0.0,
        std_value: float = 0.5,
    ) -> None:
        self.channels = channels
        self.samples = samples
        # Giá trị vô hướng trả về từ mean()/std() — cho phép test guard chia 0
        # (std_value=0.0 → normalize gặp std=0).
        self.mean_value = mean_value
        self.std_value = std_value

    @property
    def shape(self):
        return (self.channels, self.samples)

    def mean(self, dim=None):
        # mean(0) trả tensor (giảm chiều kênh); mean() trả vô hướng float.
        if dim is None:
            return self.mean_value
        return FakeAudioTensor(
            self.channels, self.samples,
            mean_value=self.mean_value, std_value=self.std_value,
        )

    def std(self, dim=None):
        return self.std_value

    def __sub__(self, other):
        return FakeAudioTensor(self.channels, self.samples)

    def __truediv__(self, other):
        return FakeAudioTensor(self.channels, self.samples)

    def __mul__(self, other):
        return FakeAudioTensor(self.channels, self.samples)

    __rmul__ = __mul__

    def __add__(self, other):
        return FakeAudioTensor(self.channels, self.samples)

    def repeat(self, *sizes):
        factor = sizes[0] if sizes else 1
        return FakeAudioTensor(self.channels * factor, self.samples)

    def __getitem__(self, item):
        if isinstance(item, slice):
            stop = item.stop if item.stop is not None else self.channels
            return FakeAudioTensor(min(stop, self.channels), self.samples)
        return FakeAudioTensor(1, self.samples)

    def unsqueeze(self, dim):
        return self

    def to(self, device):
        return self

    def cpu(self):
        return self


class FakeStemTensor:
    """Giả lập 1 stem trả về từ `apply_model`.

    `.name` để assert thành phần instrumental ghép đúng tên (test S7 — bẫy #1).
    `.amplitude` để `torchaudio.save` giả ghi WAV thật với biên độ cấu hình được
    (dùng cho test silence detection S16-S19).
    """

    def __init__(self, name: str, amplitude: float = 0.5) -> None:
        self.name = name
        self.amplitude = amplitude
        # Ghi lại toán hạng nhân gần nhất (ref_std trong `_apply`) để test guard
        # chia 0 (`stem * ref_std`): std=0 phải được thay bằng 1.0 trước khi nhân.
        self.mul_operand = None

    def __add__(self, other: "FakeStemTensor") -> "FakeStemTensor":
        if isinstance(other, FakeStemTensor):
            combined = min(1.0, self.amplitude + other.amplitude)
            return FakeStemTensor(f"{self.name}+{other.name}", amplitude=combined)
        return self

    def __radd__(self, other):
        return self

    def __mul__(self, other):
        result = FakeStemTensor(self.name, amplitude=self.amplitude)
        result.mul_operand = other
        return result

    __rmul__ = __mul__

    def __sub__(self, other):
        return self

    def cpu(self):
        return self


def _fake_save(path, tensor, sample_rate, *args, **kwargs) -> None:
    """side_effect của torchaudio.save giả — ghi WAV thật để rms_level đọc được."""
    amplitude = getattr(tensor, "amplitude", 0.5)
    make_wav(
        path,
        seconds=0.2,
        sample_rate=8000,
        channels=1,
        amplitude=amplitude,
        waveform="silence" if amplitude == 0.0 else "sine",
    )


# ---------------------------------------------------------------------------
# Fake torch/torchaudio/demucs modules
# ---------------------------------------------------------------------------


def install_fake_torch(
    monkeypatch,
    *,
    cuda: bool = True,
    name: str = "Tesla T4",
    vram_bytes: int = 15_843_721_216,
    model_sources: Sequence[str] | None = None,
    samplerate: int = 44100,
    audio_channels: int = 2,
    stem_amplitudes: dict[str, float] | None = None,
) -> SimpleNamespace:
    """Nhét module giả vào sys.modules cho torch/torchaudio/demucs.

    Dùng monkeypatch.setitem để tự khôi phục sys.modules sau mỗi test.
    Trả về SimpleNamespace chứa các mock quan trọng để test assert/override.
    """
    sources = list(model_sources) if model_sources is not None else ["drums", "bass", "other", "vocals"]
    amps = stem_amplitudes or {s: 0.5 for s in sources}

    def _make_model(model_name: str):
        model = MagicMock()
        model.sources = list(sources)
        model.samplerate = samplerate
        model.audio_channels = audio_channels
        return model

    get_model = MagicMock(side_effect=_make_model)

    def _apply_model(model, wav, split=True, overlap=0.25, device="cpu"):
        return [[FakeStemTensor(s, amplitude=amps.get(s, 0.5)) for s in model.sources]]

    apply_model = MagicMock(side_effect=_apply_model)

    load = MagicMock(return_value=(FakeAudioTensor(channels=1), samplerate))
    resample = MagicMock(side_effect=lambda wav, orig_sr, new_sr: wav)
    save = MagicMock(side_effect=_fake_save)

    cuda_props = MagicMock()
    cuda_props.name = name
    cuda_props.total_memory = vram_bytes

    fake_torch = ModuleType("torch")
    fake_torch_cuda = ModuleType("torch.cuda")
    fake_torch_cuda.is_available = MagicMock(return_value=cuda)
    fake_torch_cuda.get_device_properties = MagicMock(return_value=cuda_props)
    fake_torch_cuda.empty_cache = MagicMock()
    fake_torch.cuda = fake_torch_cuda
    fake_torch_hub = ModuleType("torch.hub")
    fake_torch_hub.set_dir = MagicMock()
    fake_torch.hub = fake_torch_hub

    fake_torchaudio = ModuleType("torchaudio")
    fake_torchaudio.load = load
    fake_torchaudio.save = save
    fake_torchaudio_functional = ModuleType("torchaudio.functional")
    fake_torchaudio_functional.resample = resample
    fake_torchaudio.functional = fake_torchaudio_functional

    fake_demucs = ModuleType("demucs")
    fake_demucs_pretrained = ModuleType("demucs.pretrained")
    fake_demucs_pretrained.get_model = get_model
    fake_demucs_apply = ModuleType("demucs.apply")
    fake_demucs_apply.apply_model = apply_model
    fake_demucs.pretrained = fake_demucs_pretrained
    fake_demucs.apply = fake_demucs_apply

    modules = {
        "torch": fake_torch,
        "torch.cuda": fake_torch_cuda,
        "torch.hub": fake_torch_hub,
        "torchaudio": fake_torchaudio,
        "torchaudio.functional": fake_torchaudio_functional,
        "demucs": fake_demucs,
        "demucs.pretrained": fake_demucs_pretrained,
        "demucs.apply": fake_demucs_apply,
    }
    for mod_name, mod in modules.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    return SimpleNamespace(
        torch=fake_torch,
        torchaudio=fake_torchaudio,
        get_model=get_model,
        apply_model=apply_model,
        save=save,
        load=load,
        resample=resample,
        cuda_is_available=fake_torch_cuda.is_available,
        get_device_properties=fake_torch_cuda.get_device_properties,
        empty_cache=fake_torch_cuda.empty_cache,
        set_dir=fake_torch_hub.set_dir,
        cuda_props=cuda_props,
    )
