# Face Cover — تغطية الوجوه في الفيديو

Local web app (FastAPI + MediaPipe + FFmpeg) that covers faces in short videos without jitter.
تطبيق ويب محلي لتغطية وجوه الموديلز في الفيديوهات بدون اهتزاز. كل المعالجة على جهازك.

## Features / المزايا
- Face tracking with MediaPipe Face Landmarker + One Euro smoothing (no jitter)
- Covers: mid-face, full oval, eye bar, soft or brush edges, 15 ready-made colored brush stamps
- Fill: auto-match hijab color, manual color picker, Gaussian blur, pixelate
- Optional hand protection (MediaPipe Hands): the hand stays visible over the cover
- Live preview on any frame, then export with FFmpeg (same resolution, fps and audio)

## Requirements / المتطلبات
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) available on `PATH` (`ffmpeg -version` must work)

## Run / التشغيل
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:8000 — on Windows you can just double-click `run.bat`.

## Notes / ملاحظات
- Model files are included in `models/` (Google MediaPipe `face_landmarker.task`, `hand_landmarker.task`).
- Uploads and exports are stored in the system temp folder (`face_cover_work`).
- Add your own stamps: put a transparent PNG in `assets/stamps/` (e.g. `s20.png`).
- If the face is fully hidden (feather, mask, back of head) it cannot be tracked; the app reports uncovered frames after export.
