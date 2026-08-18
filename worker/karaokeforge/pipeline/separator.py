"""Stage 1: Demucs wrapper — tách vocals/instrumental. (PR-A1)

Import demucs/torch BÊN TRONG method — máy dev không cài được lib GPU.
"""

from __future__ import annotations

import gc
import os

from karaokeforge.config import Config
from karaokeforge.utils.audio import rms_level
from karaokeforge.utils.gpu import detect_gpu
from karaokeforge.utils.logger import get_logger

logger = get_logger(__name__)

SILENCE_RMS_THRESHOLD = 1e-3  # D-A1-3, ~ -60 dBFS
OOM_FALLBACK_MODEL = "htdemucs"  # REQ-A1-06


class AudioSeparator:
    """Demucs wrapper với VRAM management."""

    def __init__(self) -> None:
        self.model = None
        self.model_name: str | None = None
        self.warnings: list[str] = []

    def load_model(self, model_name: str = "htdemucs_ft") -> None:
        """Load Demucs model lên GPU (cache ở Config.MODELS_CACHE nếu có)."""
        if self.model is not None and self.model_name == model_name:
            return
        if self.model is not None:
            self.unload()

        cache_dir = os.path.join(Config.MODELS_CACHE, "demucs")
        os.makedirs(cache_dir, exist_ok=True)

        import torch
        import demucs.pretrained

        # Fix 5: KHÔNG set os.environ["TORCH_HOME"] toàn cục (kéo cả model align
        # whisperx vào <cache>/demucs, lệch layout models_cache/{demucs,whisper}
        # — contracts/README §2). Dùng torch.hub.set_dir cục bộ cho Demucs.
        torch.hub.set_dir(os.path.join(cache_dir, "hub"))

        self.model = demucs.pretrained.get_model(model_name)

        gpu = detect_gpu()
        if gpu["available"]:
            self.model.cuda()
        else:
            logger.warning("Không có CUDA — Demucs chạy CPU, sẽ rất chậm")

        self.model_name = model_name
        logger.info("Đã load model Demucs %s (GPU: %s)", model_name, gpu["name"])

    def separate(self, audio_path: str, output_dir: str) -> dict[str, str]:
        """Tách audio. Trả về dict tên track → path file WAV.

        Keys bắt buộc: "vocals", "instrumental" (drums+bass+other).
        """
        os.makedirs(output_dir, exist_ok=True)
        self.warnings.clear()

        if self.model is None:
            self.load_model(Config.DEFAULT_DEMUCS_MODEL)

        stems, source_names, sr = self._apply_with_oom_retry(audio_path)
        stem_map = dict(zip(source_names, stems))
        if "vocals" not in stem_map:
            raise RuntimeError(
                f"Model Demucs '{self.model_name}' không có stem 'vocals' "
                f"(sources={list(source_names)})"
            )

        import torchaudio

        vocals_path = os.path.join(output_dir, "vocals.wav")
        inst_path = os.path.join(output_dir, "instrumental.wav")

        # Fix 1: ghi PCM_S 16-bit — mặc định torchaudio.save là PCM_F 32-bit khiến
        # wave.open raise 'unknown format: 3' (rms_level rơi xuống ffmpeg decode
        # 105MB/timeout) và instrumental.wav phình gấp đôi (phá ngân sách 15GB §6.6).
        torchaudio.save(
            vocals_path, stem_map["vocals"].cpu(), sr,
            encoding="PCM_S", bits_per_sample=16,
        )

        inst = None
        for name, tensor in zip(source_names, stems):
            if name == "vocals":
                continue
            inst = tensor if inst is None else inst + tensor  # bẫy #1: lọc theo TÊN, không theo index
        if inst is None:
            raise RuntimeError(
                f"Model Demucs '{self.model_name}' chỉ có stem 'vocals', "
                "không có stem nào để ghép instrumental"
            )
        torchaudio.save(
            inst_path, inst.cpu(), sr,
            encoding="PCM_S", bits_per_sample=16,
        )

        self._check_silence(vocals_path, inst_path)

        return {"vocals": vocals_path, "instrumental": inst_path}

    def unload(self) -> None:
        """Giải phóng VRAM (del model + torch.cuda.empty_cache)."""
        self.model = None  # không dùng `del self.model` (bẫy #3 — hỏng khi gọi lần 2)
        self.model_name = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- helpers privates --------------------------------------------------

    def _apply_with_oom_retry(self, audio_path: str):
        """Chạy `_apply`; nếu CUDA OOM thì fallback đúng 1 lần sang htdemucs (REQ-A1-06)."""
        try:
            return self._apply(audio_path)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if self.model_name == OOM_FALLBACK_MODEL:
                self.unload()
                raise RuntimeError(
                    "CUDA OOM: Demucs không đủ VRAM kể cả với htdemucs"
                ) from exc

            logger.warning(
                "CUDA OOM với model %s — fallback sang %s", self.model_name, OOM_FALLBACK_MODEL
            )
            self.unload()  # trả VRAM trước khi load model fallback (bẫy #7)
            self.load_model(OOM_FALLBACK_MODEL)
            try:
                return self._apply(audio_path)
            except RuntimeError as exc2:
                if "out of memory" not in str(exc2).lower():
                    raise
                self.unload()
                raise RuntimeError(
                    "CUDA OOM: Demucs không đủ VRAM kể cả với htdemucs"
                ) from exc2

    def _apply(self, audio_path: str):
        """Load audio, chuẩn hoá, chạy Demucs. Trả (stems, source_names, sample_rate)."""
        import torchaudio
        from demucs.apply import apply_model

        wav, sr = torchaudio.load(audio_path)

        if sr != self.model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, self.model.samplerate)
            sr = self.model.samplerate

        target_channels = self.model.audio_channels
        wav_channels = wav.shape[0]
        if wav_channels == 1 and target_channels > 1:
            wav = wav.repeat(target_channels, 1)
        elif wav_channels > target_channels:
            wav = wav[:target_channels]

        ref = wav.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        # Fix 3: input DC/im lặng → ref.std()=0 → chia 0 → NaN toàn stem.
        if ref_std < 1e-8:
            ref_std = 1.0
        normalized = (wav - ref_mean) / ref_std

        device = "cuda" if detect_gpu()["available"] else "cpu"
        sources = apply_model(
            self.model, normalized.unsqueeze(0), split=True, overlap=0.25, device=device
        )
        stems = [stem * ref_std + ref_mean for stem in sources[0]]

        return stems, self.model.sources, sr

    def _check_silence(self, vocals_path: str, inst_path: str) -> None:
        """REQ-A1-07: cảnh báo (không fail) khi 1 track gần như im lặng.

        Fix 2: đây chỉ là cảnh báo (DECISIONS PR-A1: log, KHÔNG fail stage). Nếu
        `rms_level` ném lỗi (file lỗi, sampwidth lạ...) thì nuốt lỗi + log warning,
        không được để chết cả stage separation vốn đã hoàn tất.
        """
        try:
            inst_rms = rms_level(inst_path)
            if inst_rms < SILENCE_RMS_THRESHOLD:
                msg = "instrumental gần như im lặng — có thể là nhạc không lời"
                logger.warning(msg)
                self.warnings.append(msg)

            vocals_rms = rms_level(vocals_path)
            if vocals_rms < SILENCE_RMS_THRESHOLD:
                msg = "vocals gần như im lặng — không phát hiện giọng hát"
                logger.warning(msg)
                self.warnings.append(msg)
        except Exception as exc:
            logger.warning("Bỏ qua kiểm tra im lặng (lỗi đọc RMS): %s", exc)
