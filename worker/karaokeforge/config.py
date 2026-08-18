"""Cấu hình worker. Các hằng số protocol lấy từ contracts/README.md (D3)."""


class Config:
    # Worker
    WORKER_ID = "worker_a"            # unique per Colab tab
    POLL_INTERVAL = 15                # giây giữa các lần poll
    HEARTBEAT_INTERVAL = 60           # giây — cập nhật progress.heartbeat_at tối thiểu mỗi chừng này

    # Claim protocol (contract D3)
    CLAIM_SETTLE_S = 5                # chờ sync sau khi claim rồi verify ownership
    STALE_AFTER_MIN = 10              # heartbeat cũ hơn → job stale, trả về pending
    MAX_ATTEMPTS = 3                  # quá số lần → failed
    WORKER_PARTITION = None           # None hoặc (index, total) — bật để loại trừ collision

    # Google Drive
    DRIVE_MOUNT = "/content/drive"
    DRIVE_ROOT = "/content/drive/MyDrive/KaraokeForge"

    # Models
    MODELS_CACHE = f"{DRIVE_ROOT}/models_cache"
    DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
    DEFAULT_WHISPER_MODEL = "large-v3"
    WHISPER_COMPUTE_TYPE = "float16"  # float16 cho GPU, int8 cho VRAM thấp
    WHISPER_BATCH_SIZE = 16

    # Video
    DEFAULT_FPS = 30
    FONT_DIR = "/usr/share/fonts/custom"
    DEFAULT_FONT = "BeVietnamPro-Bold.ttf"
    FFMPEG_PRESET = "fast"
    FFMPEG_CRF = 23

    # Temp — local SSD nhanh hơn Drive; ghi xong mới copy sang Drive
    TEMP_DIR = "/content/temp"

    LOG_LEVEL = "INFO"

    @classmethod
    def auto_select_models(cls) -> None:
        """Tự chọn model theo VRAM khả dụng (PRD S7.2).

        Dùng `utils.gpu.detect_gpu()` — KHÔNG `import torch` ở đây và KHÔNG tự
        đọc thuộc tính VRAM sai như pseudo-code PRD (tên thuộc tính đúng đã
        được `detect_gpu()` xử lý, xem DECISIONS.md G4). Idempotent, KHÔNG đụng
        hằng số protocol D3
        (`CLAIM_SETTLE_S`, `STALE_AFTER_MIN`, `MAX_ATTEMPTS`, `POLL_INTERVAL`,
        `HEARTBEAT_INTERVAL`).
        """
        from karaokeforge.utils.gpu import detect_gpu  # import trong hàm (REQ-CF03)

        try:
            gpu = detect_gpu()
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "detect_gpu() lỗi khi auto_select_models — giữ model mặc định an toàn",
                exc_info=True,
            )
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs"
            cls.DEFAULT_WHISPER_MODEL = "small"
            cls.WHISPER_COMPUTE_TYPE = "int8"
            return

        if not gpu.get("available"):
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs"
            cls.DEFAULT_WHISPER_MODEL = "small"
            cls.WHISPER_COMPUTE_TYPE = "int8"
            return

        vram = gpu.get("vram_gb") or 0.0
        if vram >= 15:
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
            cls.DEFAULT_WHISPER_MODEL = "large-v3"
            cls.WHISPER_COMPUTE_TYPE = "float16"
        elif vram >= 8:
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
            cls.DEFAULT_WHISPER_MODEL = "medium"
            cls.WHISPER_COMPUTE_TYPE = "float16"
        elif vram >= 4:
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs"
            cls.DEFAULT_WHISPER_MODEL = "small"
            cls.WHISPER_COMPUTE_TYPE = "int8"
        else:
            cls.DEFAULT_DEMUCS_MODEL = "htdemucs"
            cls.DEFAULT_WHISPER_MODEL = "tiny"
            cls.WHISPER_COMPUTE_TYPE = "int8"
