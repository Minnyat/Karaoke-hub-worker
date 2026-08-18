# KaraokeForge — Worker (Colab)

Phần **worker** của KaraokeForge, chạy trên Google Colab GPU: tách nhạc (Demucs) →
nhận dạng + align lời (WhisperX) → render video karaoke (FFmpeg). Đọc/ghi job qua
Google Drive (mount như filesystem).

> Đây là **repo public deploy** để Colab clone. Nơi phát triển chính (gồm cả WebUI)
> là monorepo private riêng. Đừng sửa trực tiếp ở đây — sửa ở monorepo rồi sync.

## Chạy trên Colab

Mở `notebooks/KaraokeForge_Worker.ipynb` trên [Colab](https://colab.research.google.com)
(File → Open notebook → GitHub → repo này), chọn Runtime GPU T4, chạy lần lượt các cell.
Chi tiết + checklist verify: xem RUNBOOK trong monorepo.

## Cấu trúc

```
worker/karaokeforge/   # pipeline (separator, transcriber, renderer), drive queue, worker loop
worker/tests/          # pytest (mock torch/demucs/whisperx — chạy được không cần GPU)
notebooks/             # KaraokeForge_Worker.ipynb
contracts/             # job.schema.json, lyrics.schema.json — hợp đồng dữ liệu WebUI↔worker
```

## Test local (không cần GPU)

```bash
cd worker
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Runtime thật cần `worker/requirements.txt` (torch/demucs/whisperx) — chỉ cài trên Colab.
