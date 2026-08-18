# KaraokeForge — Contract v1 (NGUỒN SỰ THẬT DUY NHẤT)

> **Quy tắc cứng:** Mọi worktree/PR coi thư mục `contracts/` là **read-only**.
> Muốn đổi contract → mở PR riêng chỉ sửa `contracts/`, đối chiếu cả phía web lẫn worker.
> Khi PRD (`karaoke-prd-v3-full.md`) mâu thuẫn với file này → **file này thắng**.

Phiên bản: `v1` — 2026-08-18

---

## 1. Các quyết định kiến trúc đã chốt (giải quyết mâu thuẫn trong PRD)

### D1 — OAuth scope: dùng `https://www.googleapis.com/auth/drive` (full scope)

PRD đề xuất `drive.file`, nhưng scope đó chỉ thấy file **do chính app tạo**. Worker Colab
ghi output qua mount filesystem → file do DriveFS tạo, KHÔNG phải app tạo → WebUI với
`drive.file` sẽ không đọc được job JSON đã move sang `completed/` lẫn video output.

**Quyết định v1:** full `drive` scope, OAuth app giữ ở trạng thái **Testing** (tối đa 100
test user, dùng cá nhân). Trước khi public → thiết kế lại (ví dụ: WebUI pre-create
placeholder files, worker chỉ overwrite giữ nguyên fileId).

### D2 — Tài khoản: v1 chạy SINGLE-ACCOUNT

Một tài khoản Google duy nhất: sở hữu Drive, chạy Colab worker, đăng nhập WebUI.
Sơ đồ multi-account (Account A/B/C) trong PRD là mục tiêu v2 — cần giải bài toán
share folder + shortcut "Add to My Drive", **ngoài phạm vi v1**. Code không được
giả định đường tắt nào phụ thuộc multi-account.

### D3 — Claim protocol: move-then-verify + heartbeat

`shutil.move` trên DriveFS **không atomic** (cache, sync trễ). Protocol bắt buộc:

1. **Claim:** move `queue/pending/{job_id}.json` → `queue/processing/{job_id}.json`.
2. Ghi ngay `progress.worker_id = <mình>`, `progress.started_at`, `progress.heartbeat_at`.
3. Chờ `CLAIM_SETTLE_S = 5` giây, đọc lại file. Nếu `progress.worker_id != <mình>`
   → thua claim, **bỏ qua job, không đụng gì thêm**.
4. **Heartbeat:** trong suốt quá trình xử lý, cập nhật `progress.heartbeat_at`
   tối thiểu mỗi 60 giây (mỗi lần update progress đều set luôn).
5. **Stale recovery:** job trong `processing/` có `heartbeat_at` cũ hơn
   `STALE_AFTER_MIN = 10` phút → move về `pending/`, tăng `attempts`.
   `attempts > 3` → move sang `failed/` với `error = "max retries exceeded"`.
   - KHÔNG dùng `started_at` để xác định stale (render dài sẽ bị requeue oan).
6. **Partition (tuỳ chọn, tắt mặc định):** config `WORKER_PARTITION = (index, total)`
   → worker chỉ claim job có `crc32(job_id) % total == index`. Bật khi chạy nhiều
   worker để loại trừ collision tuyệt đối.

### D4 — Quyền ghi job JSON

- WebUI **tạo** job JSON trong `pending/`; sau đó chỉ **đọc** job JSON.
- Worker là chủ sở hữu duy nhất của job JSON từ lúc claim đến khi move sang
  `completed/`/`failed/`.
- Lyrics do user sửa: WebUI ghi file **riêng** `outputs/{job_id}/lyrics_edited.json`,
  không sửa job JSON.

### D5 — Chống trùng folder gốc trên Drive

Drive cho phép folder trùng tên. `ensureFolder` phía WebUI phải:
`find → nếu có ≥1 kết quả lấy cái **cũ nhất** (createdTime tăng dần) → chỉ create khi
không có`. Worker phía mount dùng path cố định `/content/drive/MyDrive/KaraokeForge`.

Folder gốc `KaraokeForge` phía WebUI: query BẮT BUỘC kèm `'root' in parents`
(ngay dưới My Drive) — nếu không, folder trùng tên nested ở nơi khác có thể được
chọn và worker (mount path cố định) sẽ không bao giờ thấy job → job "mất tích".

### D6 — Polling phía WebUI

Không list 4 folder mỗi lần poll. WebUI lưu `_jobFileId` khi tạo job (fileId không đổi
khi move giữa các folder) → poll = 1 call `files/{id}?fields=parents` + 1 call đọc
content khi cần. Dashboard list toàn bộ: tối đa 1 lần / 30s.

---

## 2. Cấu trúc folder trên Drive (cố định)

```
KaraokeForge/
├── queue/
│   ├── pending/          # {job_id}.json — WebUI tạo
│   ├── processing/       # worker đang xử lý
│   ├── completed/        # xong
│   └── failed/           # lỗi / quá số lần retry
├── uploads/{job_id}/     # original.<ext>, lyrics_input.txt
├── outputs/{job_id}/     # vocals.wav, instrumental.wav, lyrics_raw.json,
│                         # lyrics_aligned.json, lyrics_edited.json,
│                         # preview.mp4, karaoke_final.mp4
└── models_cache/         # demucs/, whisper/
```

Tên folder queue: đúng 4 tên `pending|processing|completed|failed` (KHÔNG có `done/`).

## 3. Tên stage & checkpoint (cố định — khớp từng ký tự)

| Stage key           | Checkpoint key      | Ý nghĩa            |
|---------------------|---------------------|--------------------|
| `audio_separation`  | `audio_separated`   | Demucs tách nhạc   |
| `lyrics_alignment`  | `lyrics_aligned`    | Whisper + align    |
| `video_render`      | `video_rendered`    | FFmpeg render      |

`status` của job: `pending | processing | completed | failed`.
`status` của stage: `pending | running | completed | failed`.

## 4. Job JSON schema

Schema máy đọc: [`job.schema.json`](./job.schema.json) — validate mọi fixture/test bằng file này.
Ví dụ chuẩn: [`examples/job_example.json`](./examples/job_example.json).

Format lyrics (nội dung `lyrics_raw.json` / `lyrics_aligned.json` / `lyrics_edited.json`):
[`lyrics.schema.json`](./lyrics.schema.json) — array of LyricsSegment, đơn vị giây.
`web/types/index.ts` (LyricsSegment/WordTiming) và worker pipeline phải khớp file này.

Khác biệt so với PRD (đã hợp nhất §2.2 + S2.3):

- Thêm `input.upload_folder_id`, `output.output_folder_id`,
  `config.guide_vocal_volume` (từ S2.3).
- Thêm `attempts` (int, mặc định 0) và `progress.heartbeat_at` (D3).
- `_jobFileId` chỉ tồn tại phía WebUI in-memory, **không được ghi vào file**.

## 5. Ranh giới sở hữu file trong repo (chống conflict giữa các worktree)

| Khu vực                              | Chủ sở hữu (PR)  |
|--------------------------------------|------------------|
| `contracts/`                         | PR riêng, review 2 phía |
| `web/package.json`, config web       | PR-C1C2          |
| `web/lib/drive-client.ts`, `web/lib/job-service.ts` | PR-D2 |
| `web/app/`, `web/components/`        | PR-C1C2          |
| `web/types/index.ts`                 | baseline (sync từ contract, sửa qua PR contract) |
| `worker/karaokeforge/pipeline/separator.py`, `worker/karaokeforge/utils/` | PR-A1 |
| `worker/karaokeforge/pipeline/transcriber.py`, `aligner.py`, `vietnamese.py` | PR-A2A3 |
| `worker/karaokeforge/pipeline/renderer.py`, `worker/karaokeforge/video/` | PR-A4 |
| `worker/karaokeforge/drive/`         | PR-B2B3          |
| `worker/karaokeforge/worker.py`, `config.py` | PR-B4 (wave 2) |

Interface giữa các module = signatures trong skeleton baseline. Đổi signature
= đổi contract → PR riêng.

## 6. Ghi chú kỹ lỗi PRD đã sửa trong contract này

1. §2.2 vs S2.3 schema lệch nhau → hợp nhất tại `job.schema.json`.
2. `KaraokeRenderer.render()` trong PRD là generator nhưng được gọi như hàm thường
   → v1: `render()` nhận callback `on_progress(percent: float)`, trả về path video.
3. `recover_stale_jobs` dùng `started_at` + 30 phút → đổi sang heartbeat 10 phút (D3).
4. Ước tính quota "50 calls/job" sai → poll theo fileId (D6).
5. URL font `github.com/bettergoogle/...` không tin cậy → tải Be Vietnam Pro trực tiếp
   từ Google Fonts (`https://fonts.google.com/download?family=Be%20Vietnam%20Pro`)
   hoặc github.com/google/fonts (OFL), verify khi viết notebook.
6. Storage: models_cache ~6.3GB nằm trong 15GB free → thực tế còn ~8.7GB cho jobs
   (~40–80 bài). Cleanup separated tracks sau render là bắt buộc, không phải tuỳ chọn.
