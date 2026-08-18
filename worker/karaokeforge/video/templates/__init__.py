"""Video templates: classic, modern, neon. Spec: PRD S3. (PR-A4)"""

from __future__ import annotations

from karaokeforge.video.templates.base import BaseTemplate
from karaokeforge.video.templates.classic import ClassicTemplate
from karaokeforge.video.templates.modern import ModernTemplate
from karaokeforge.video.templates.neon import NeonTemplate

_TEMPLATES: dict[str, type[BaseTemplate]] = {
    "classic": ClassicTemplate,
    "modern": ModernTemplate,
    "neon": NeonTemplate,
}


def get_template(name: str, width: int, height: int, config: dict) -> BaseTemplate:
    """Factory: trả về instance `BaseTemplate` theo tên (`"classic"|"modern"|"neon"`,
    so khớp case-insensitive). Tên khác -> `ValueError` liệt kê tên hợp lệ."""
    key = (name or "").strip().lower()
    template_cls = _TEMPLATES.get(key)
    if template_cls is None:
        valid = ", ".join(sorted(_TEMPLATES))
        raise ValueError(f"Template không hợp lệ: {name!r}. Tên hợp lệ: {valid}")
    return template_cls(width, height, config)
