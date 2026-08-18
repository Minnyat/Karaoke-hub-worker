"""GPU detection. Import torch bên trong hàm. (PR-A1)"""

from __future__ import annotations

from karaokeforge.utils.logger import get_logger

logger = get_logger(__name__)

_UNAVAILABLE: dict = {"available": False, "name": None, "vram_gb": 0.0}


def detect_gpu() -> dict:
    """Trả về {"available": bool, "name": str | None, "vram_gb": float}."""
    try:
        import torch
    except ImportError:
        return dict(_UNAVAILABLE)

    try:
        if not torch.cuda.is_available():
            return dict(_UNAVAILABLE)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1e9, 2)
        return {"available": True, "name": props.name, "vram_gb": vram_gb}
    except Exception:
        logger.warning("Không phát hiện được GPU (lỗi khi truy vấn torch.cuda)", exc_info=True)
        return dict(_UNAVAILABLE)
