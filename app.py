"""خادم FastAPI لتطبيق تغطية الوجوه في الفيديو."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import engine
from engine import Settings

BASE = Path(__file__).resolve().parent
# مسار ASCII لأن OpenCV يتعثر أحياناً بالمسارات العربية على ويندوز
WORK = Path(tempfile.gettempdir()) / "face_cover_work"
WORK.mkdir(exist_ok=True)

app = FastAPI(title="Face Cover")
videos: dict[str, dict] = {}
jobs: dict[str, dict] = {}
_frame_cache: dict[tuple, tuple] = {}  # (vid, t_ms, hands) -> (frame, lm, hands)


def ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def read_frame(path: str, t: float):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idx = max(0, min(n - 1, int(round(t * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "تعذّر قراءة الإطار")
    return fr


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:10]
    ext = Path(file.filename or "v.mp4").suffix.lower() or ".mp4"
    dst = WORK / f"{vid}{ext}"
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    cap = cv2.VideoCapture(str(dst))
    ok, fr = cap.read()
    if not ok:
        cap.release()
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "ملف فيديو غير مدعوم")
    h, w = fr.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    videos[vid] = dict(path=str(dst), name=file.filename, w=w, h=h, fps=fps, frames=n)
    return dict(id=vid, name=file.filename, width=w, height=h, fps=fps, frames=n, duration=n / fps)


@app.post("/api/preview")
async def preview(body: dict):
    v = videos.get(body.get("id"))
    if not v:
        raise HTTPException(404, "فيديو غير موجود")
    s = Settings.from_dict(body.get("settings", {}))
    t = float(body.get("t", 0))
    key = (body["id"], round(t, 3), s.hands)
    if key not in _frame_cache:
        fr = read_frame(v["path"], t)
        lm, hl = engine.detect_still(fr, s.hands)
        if len(_frame_cache) > 12:
            _frame_cache.pop(next(iter(_frame_cache)))
        _frame_cache[key] = (fr, lm, hl)
    fr, lm, hl = _frame_cache[key]
    out = fr.copy()
    found = lm is not None
    if found:
        col = engine.sample_hijab_color(out, lm) if s.fill == "auto" else None
        engine.render(out, lm, hl if s.hands else None, s, col)
    ph = out.shape[0]
    if max(out.shape[:2]) > 1280:
        k = 1280 / max(out.shape[:2])
        out = cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Face-Found": "1" if found else "0"})


def run_job(job_id: str, vid: str, s: Settings):
    j = jobs[job_id]
    v = videos[vid]
    try:
        cap = cv2.VideoCapture(v["path"])
        fps = v["fps"]
        n = v["frames"]
        finder = engine.FaceFinder()
        hands = engine.make_hands(True) if s.hands else None
        track, hand_track, colors = [], [], []
        # ---- المرحلة 1: الكشف
        j.update(stage="تتبع الوجه", pct=0)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            ts = int(i * 1000 / fps)
            lm = finder.find(fr)
            hl = engine.detect_hands(hands, fr, ts) if hands is not None else []
            track.append(lm)
            hand_track.append(hl)
            colors.append(engine.hijab_candidates(fr, lm) if (lm is not None and s.fill == "auto") else None)
            i += 1
            if i % 5 == 0:
                j["pct"] = min(45.0, 45.0 * i / max(n, 1))
            if j.get("cancel"):
                raise RuntimeError("أُلغيت العملية")
        cap.release()
        total = len(track)
        track, lost = engine.fill_gaps(track, s.max_gap)
        track = engine.smooth_track(track, fps, s.smoothing)
        colors = engine.smooth_colors(engine.pick_colors(colors)) if s.fill == "auto" else colors
        # ---- المرحلة 2: الرسم والتصدير
        j.update(stage="رسم القناع وتصدير الفيديو", pct=45)
        out_path = WORK / f"out_{job_id}.mp4"
        w, h = v["w"], v["h"]
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-i", v["path"], "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(v["path"])
        for i in range(total):
            ok, fr = cap.read()
            if not ok:
                break
            if track[i] is not None:
                engine.render(fr, track[i], hand_track[i] if s.hands else None, s,
                              colors[i] if s.fill == "auto" else None)
            proc.stdin.write(fr.tobytes())
            if i % 5 == 0:
                j["pct"] = 45 + 55.0 * i / max(total, 1)
            if j.get("cancel"):
                proc.kill()
                raise RuntimeError("أُلغيت العملية")
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg: " + err[-400:])
        uncovered = sum(1 for t in track if t is None)
        j.update(stage="تم", pct=100, done=True, file=str(out_path), lost=lost, uncovered=uncovered, total=total)
    except Exception as e:  # noqa: BLE001
        j.update(error=str(e), done=True)


@app.post("/api/process")
async def process(body: dict):
    if body.get("id") not in videos:
        raise HTTPException(404, "فيديو غير موجود")
    s = Settings.from_dict(body.get("settings", {}))
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = dict(pct=0, stage="بدء", done=False, started=time.time())
    threading.Thread(target=run_job, args=(job_id, body["id"], s), daemon=True).start()
    return {"job": job_id}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404)
    return {k: v for k, v in j.items() if k != "file"} | {"ready": "file" in j}


@app.post("/api/cancel/{job_id}")
async def cancel(job_id: str):
    if job_id in jobs:
        jobs[job_id]["cancel"] = True
    return {"ok": True}


@app.get("/api/download/{job_id}")
async def download(job_id: str, inline: int = 0):
    j = jobs.get(job_id)
    if not j or "file" not in j:
        raise HTTPException(404)
    name = Path(j.get("name") or "video").stem
    return FileResponse(j["file"], media_type="video/mp4",
                        filename=None if inline else "covered.mp4")


@app.get("/api/stamps")
async def stamps():
    return sorted(p.stem for p in (BASE / "assets" / "stamps").glob("*.png"))


app.mount("/stamps", StaticFiles(directory=BASE / "assets" / "stamps"), name="stamps")
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
