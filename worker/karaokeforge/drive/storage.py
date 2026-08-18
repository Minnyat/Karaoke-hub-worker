"""Upload/download file outputs qua Drive mount. (PR-B2B3)

Pattern bắt buộc: ghi file lớn vào Config.TEMP_DIR trước, xong mới copy
sang Drive (mount chậm và không atomic khi ghi trực tiếp).
"""

from __future__ import annotations

import os
import shutil

# Separated tracks xoá sau render (giữ instrumental.wav — PRD dòng 1013, UI có nút
# "Tải Instrumental WAV"). v1 không publish vocals.wav ra ngoài giới hạn worker
# (DECISIONS.md PR-B2B3 Q3: output.vocals_file_id luôn null); vocals.wav bị dọn
# cùng các stem khác.
_INTERMEDIATE = ("vocals.wav", "drums.wav", "bass.wav", "other.wav")


def _safe_component(job_id: str) -> str:
    """Đảm bảo job_id là 1 path-component an toàn trước khi ghép vào path (FIX-2).

    Reject rỗng, ".", "..", chứa "/" "\\" ".." hoặc absolute path — chặn path
    traversal ghi file ra ngoài uploads/outputs. KHÔNG ép regex job-id đầy đủ ở
    đây (helper còn phục vụ id test dạng "job_x")."""
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"job_id không hợp lệ: {job_id!r}")
    if (
        job_id in (".", "..")
        or "/" in job_id
        or "\\" in job_id
        or ".." in job_id
        or os.path.isabs(job_id)
    ):
        raise ValueError(f"job_id không an toàn (path traversal): {job_id!r}")
    return job_id


class DriveStorage:
    """Path helpers + copy outputs cho 1 job."""

    def __init__(self, drive_root: str) -> None:
        self.drive_root = str(drive_root)
        self._uploads_root = os.path.join(self.drive_root, "uploads")
        self._outputs_root = os.path.join(self.drive_root, "outputs")

    def upload_dir(self, job_id: str) -> str:
        """Path uploads/{job_id}/ trên mount."""
        return os.path.join(self._uploads_root, _safe_component(job_id))

    def output_dir(self, job_id: str) -> str:
        """Path outputs/{job_id}/ trên mount (tạo nếu chưa có)."""
        path = os.path.join(self._outputs_root, _safe_component(job_id))
        os.makedirs(path, exist_ok=True)
        return path

    def publish_outputs(self, job_id: str, local_files: dict[str, str]) -> dict[str, str]:
        """Copy các file local (tên chuẩn → path) vào outputs/{job_id}/.
        Trả về map tên → path đích trên Drive.

        Pattern bắt buộc: copy vào tên tạm `.{name}.part` trong chính folder đích
        rồi `os.replace` sang tên chuẩn — WebUI (poll theo tên chuẩn) không bao giờ
        thấy file nửa vời (REQ-S03). Nếu 1 source thiếu → raise FileNotFoundError
        ngay lập tức; các file đã publish thành công trước đó trong cùng lần gọi
        vẫn giữ nguyên (không rollback).
        """
        dst_dir = self.output_dir(job_id)
        # FIX-8: dọn .part mồ côi từ lần crash trước (không để rác tích luỹ).
        for name in os.listdir(dst_dir):
            if name.endswith(".part"):
                try:
                    os.remove(os.path.join(dst_dir, name))
                except OSError:
                    pass
        result: dict[str, str] = {}
        for name, src in local_files.items():
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            tmp_path = os.path.join(dst_dir, f".{name}.part")
            final_path = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, tmp_path)
                os.replace(tmp_path, final_path)
            except Exception:
                # FIX-8: copy/replace ném giữa chừng -> xoá .part dở, không để rác.
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise
            result[name] = final_path
        return result

    def cleanup_intermediate(self, job_id: str) -> None:
        """Xoá separated tracks không cần giữ sau render (bắt buộc — storage 15GB,
        contracts/README.md §6.6).

        Giữ lại instrumental.wav và toàn bộ lyrics JSON + video. Idempotent: gọi
        nhiều lần liên tiếp, hoặc gọi trên job_id chưa từng có folder, đều không
        raise (REQ-S06 — worker có thể retry cleanup).
        """
        out_dir = os.path.join(self._outputs_root, job_id)
        for name in _INTERMEDIATE:
            path = os.path.join(out_dir, name)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
