"""محرك تغطية الوجوه: تتبع (MediaPipe) + تنعيم (OneEuro) + رسم القناع."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
BRUSH_PATH = BASE / "assets" / "brush.png"

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions


# ----------------------------------------------------------------------------
# One Euro Filter (vectorised)
# ----------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x.copy(), np.zeros_like(x)
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


# ----------------------------------------------------------------------------
# الإعدادات
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    mode: str = "mid"            # mid | oval | eyes
    edge: str = "soft"           # soft | brush
    stamp: str = ""              # اسم ختم جاهز من assets/stamps (فارغ = بدون)
    stamp_tint: bool = False     # تلوين الختم بلون التعبئة المختار بدل لونه الأصلي
    fill: str = "auto"           # auto | manual | blur | pixel
    color: str = "#0b5a3c"       # للتلوين اليدوي
    width: float = 100           # % عرض القناع
    height: float = 100          # % ارتفاع القناع
    offset_y: float = 0          # % إزاحة عمودية (موجب = للأعلى)
    feather: float = 25          # نعومة الحواف 0-100
    opacity: float = 100         # % التعتيم
    blur: float = 60             # قوة التمويه/البكسلة 0-100
    smoothing: float = 60        # قوة التنعيم 0-100
    hands: bool = False          # حماية اليد
    max_gap: int = 15            # أقصى عدد إطارات يُثبَّت فيها القناع عند ضياع الوجه

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                v = d[f.name]
                try:
                    setattr(s, f.name, type(getattr(s, f.name))(v) if f.type != "bool" else bool(v))
                except (TypeError, ValueError):
                    pass
        return s


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    if len(h) != 6:
        h = "0b5a3c"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], np.float32)


# ----------------------------------------------------------------------------
# الكاشفات
# ----------------------------------------------------------------------------
def make_face(video: bool):
    buf = (MODELS / "face_landmarker.task").read_bytes()
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(opts)


def make_hands(video: bool):
    buf = (MODELS / "hand_landmarker.task").read_bytes()
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf),
        running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(opts)


def _mp_rgb(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def _lm_from(res, w, h):
    if not res.face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)


class FaceFinder:
    """كاشف وجه قوي للوجوه الصغيرة: يقصّ ويكبّر حول آخر موضع، ثم الإطار كاملاً،
    ثم يمسح الإطار ببلاطات مكبّرة عند فقدان الوجه."""

    CROP = 512

    def __init__(self):
        self.face = make_face(False)
        self.last = None          # (cx, cy, size)
        self.since_scan = 999

    def _crop_detect(self, frame, cx, cy, side):
        H, W = frame.shape[:2]
        side = int(max(64, min(side, W, H)))
        x0 = int(np.clip(cx - side / 2, 0, W - side))
        y0 = int(np.clip(cy - side / 2, 0, H - side))
        crop = frame[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (self.CROP, self.CROP),
                          interpolation=cv2.INTER_CUBIC if side < self.CROP else cv2.INTER_AREA)
        res = self.face.detect(_mp_rgb(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        lm = _lm_from(res, side, side)
        if lm is None:
            return None
        return lm + np.array([x0, y0], np.float32)

    @staticmethod
    def _box(lm):
        mn, mx = lm.min(0), lm.max(0)
        return (mn + mx) / 2, float(max(mx - mn))

    def find(self, frame):
        H, W = frame.shape[:2]
        lm = None
        if self.last is not None:
            (cx, cy), size = self.last
            lm = self._crop_detect(frame, cx, cy, size * 2.6)
        if lm is None:
            m = max(H, W)
            f = frame if m <= 1920 else cv2.resize(frame, None, fx=1920 / m, fy=1920 / m, interpolation=cv2.INTER_AREA)
            res = self.face.detect(_mp_rgb(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            lm = _lm_from(res, W, H)
        if lm is None and self.since_scan >= 6:
            self.since_scan = 0
            lm = self._scan(frame)
        else:
            self.since_scan += 1
        if lm is not None:
            c, size = self._box(lm)
            # تحسين: إعادة الكشف على قصّ مضبوط حول الوجه
            lm2 = self._crop_detect(frame, c[0], c[1], size * 2.2)
            if lm2 is not None:
                lm = lm2
                c, size = self._box(lm)
            self.last = (c, size)
        else:
            self.last = None if self.since_scan > 30 else self.last
        return lm

    def _scan(self, frame):
        H, W = frame.shape[:2]
        for frac in (0.34, 0.55):
            side = int(min(W, H) * frac)
            stride = max(1, side // 2)
            for y in range(0, max(1, H - side + 1), stride):
                for x in range(0, max(1, W - side + 1), stride):
                    lm = self._crop_detect(frame, x + side / 2, y + side / 2, side)
                    if lm is not None:
                        return lm
        return None


def detect_hands(hands, frame_bgr, ts_ms=None):
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    f = frame_bgr if m <= 1920 else cv2.resize(frame_bgr, None, fx=1920 / m, fy=1920 / m, interpolation=cv2.INTER_AREA)
    img = _mp_rgb(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    hr = hands.detect(img) if ts_ms is None else hands.detect_for_video(img, int(ts_ms))
    return [np.array([[p.x * w, p.y * h] for p in hand], np.float32) for hand in (hr.hand_landmarks or [])]


# ----------------------------------------------------------------------------
# الهندسة: بناء مضلع القناع من النقاط (يتبع الدوران ثلاثي الأبعاد لأنه مبني على النقاط)
# ----------------------------------------------------------------------------
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
CHEEKS = [234, 127, 93, 132, 454, 356, 323, 361]
LOWER = [58, 172, 136, 150, 149, 200, 378, 379, 365, 397, 288]
EYE_CENTER_PTS = [33, 263, 159, 386, 145, 374, 105, 334]


def frame_axes(lm):
    """محاور الوجه المحلية: r (يمين الصورة)، u (للأعلى)، وأبعاد W/H."""
    rv = lm[454] - lm[234]
    W = float(np.linalg.norm(rv)) + 1e-6
    uv = lm[10] - lm[152]
    H = float(np.linalg.norm(uv)) + 1e-6
    u = uv / H
    # نجعل r عمودياً على u مع الحفاظ على اتجاه الدوران (roll)
    r = np.array([-u[1], u[0]], np.float32) * -1.0
    if np.dot(r, rv) < 0:
        r = -r
    return r.astype(np.float32), u.astype(np.float32), W, H


def to_local(pts, c, r, u):
    d = pts - c
    return np.stack([d @ r, d @ u], 1)


def from_local(a, c, r, u):
    return c + a[:, :1] * r + a[:, 1:2] * u


def rounded_rect(hw, hh, rad, n=10):
    rad = min(rad, hw, hh)
    pts = []
    for cx, cy, a0 in [(hw - rad, hh - rad, 0), (-hw + rad, hh - rad, 90),
                       (-hw + rad, -hh + rad, 180), (hw - rad, -hh + rad, 270)]:
        for k in range(n + 1):
            t = math.radians(a0 + 90 * k / n)
            pts.append((cx + rad * math.cos(t), cy + rad * math.sin(t)))
    return np.array(pts, np.float32)


def build_polygon(lm, s: Settings):
    r, u, W, H = frame_axes(lm)
    sx, sy = s.width / 100.0, s.height / 100.0
    shift = s.offset_y / 100.0 * H
    if s.mode == "eyes":
        c = lm[EYE_CENTER_PTS].mean(0)
        hw, hh = W * 0.60 * sx, H * 0.115 * sy
        loc = rounded_rect(hw, hh, hh * 0.9)
        poly = from_local(loc, c, r, u) + u * shift
        return poly, r, u, W, H
    if s.mode == "oval":
        pts = lm[OVAL].copy()
        c = pts.mean(0)
        loc = to_local(pts, c, r, u)
        loc[:, 0] *= sx * 1.04
        loc[:, 1] *= sy * 1.04
        poly = from_local(loc, c, r, u) + u * shift
        return poly, r, u, W, H
    # mid: من أسفل الحواجب (مرفوعة قليلاً) إلى أعلى الذقن
    brows = lm[BROWS] + u * (H * 0.05)
    pts = np.concatenate([brows, lm[CHEEKS], lm[LOWER]], 0)
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    c = hull.mean(0)
    loc = to_local(hull, c, r, u)
    loc[:, 0] *= sx * 1.06
    loc[:, 1] *= sy * 1.02
    poly = from_local(loc, c, r, u) + u * shift
    return poly, r, u, W, H


# ----------------------------------------------------------------------------
# اليد
# ----------------------------------------------------------------------------
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PALM = [0, 1, 5, 9, 13, 17]


def hand_mask(shape, hands, origin=(0, 0)):
    m = np.zeros(shape, np.uint8)
    ox, oy = origin
    for h in hands:
        p = h - np.array([ox, oy], np.float32)
        palm = float(np.linalg.norm(p[0] - p[9])) + 1e-6
        th = max(2, int(palm * 0.32))
        cv2.fillPoly(m, [np.round(p[PALM]).astype(np.int32)], 255, cv2.LINE_AA)
        for f in FINGERS:
            cv2.polylines(m, [np.round(p[f]).astype(np.int32)], False, 255, th, cv2.LINE_AA)
        for q in p:
            cv2.circle(m, (int(q[0]), int(q[1])), max(1, th // 2), 255, -1, cv2.LINE_AA)
        k = max(1, int(palm * 0.12))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m


# ----------------------------------------------------------------------------
# الفرشاة
# ----------------------------------------------------------------------------
_brush_cache = {}


def _imread(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        return None


STAMPS = BASE / "assets" / "stamps"


def load_stamp(name):
    key = "stamp_" + name
    if key not in _brush_cache:
        path = STAMPS / (Path(name).name + ".png")
        im = _imread(path)
        if im is None or im.shape[2] != 4:
            raise ValueError("ختم غير موجود: " + name)
        _brush_cache[key] = (np.ascontiguousarray(im[..., 3]), np.ascontiguousarray(im[..., :3]))
    return _brush_cache[key]


def brush_alpha():
    if "a" not in _brush_cache:
        im = _imread(BRUSH_PATH)
        a = im[..., 3] if im is not None and im.shape[2] == 4 else np.full((100, 100), 255, np.uint8)
        ys, xs = np.where(a > 8)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        _brush_cache["a"] = a
    return _brush_cache["a"]


# ----------------------------------------------------------------------------
# الرسم
# ----------------------------------------------------------------------------
def hijab_candidates(frame, lm, k=3):
    """عناقيد ألوان من حلقة رفيعة حول بيضاوية الوجه (حيث يقع الحجاب).
    تعيد (مراكز[k,3], أعداد[k]) أو None."""
    h, w = frame.shape[:2]
    pts = lm[OVAL]
    c = pts.mean(0)
    r, u, W, H = frame_axes(lm)

    def scaled(f):
        return np.round(c + (pts - c) * f).astype(np.int32)

    outer, inner = scaled(1.55), scaled(1.10)
    x0 = max(0, int(outer[:, 0].min()))
    y0 = max(0, int(outer[:, 1].min()))
    x1 = min(w, int(outer[:, 0].max()) + 1)
    y1 = min(h, int(outer[:, 1].max()) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], np.int32)
    cv2.fillPoly(m, [outer - off], 255)
    cv2.fillPoly(m, [inner - off], 0)
    px = frame[y0:y1, x0:x1][m > 0]
    if len(px) < 40:
        return None
    if len(px) > 1500:
        px = px[np.random.default_rng(0).choice(len(px), 1500, replace=False)]
    data = px.astype(np.float32)
    kk = min(k, len(data))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, lab, cen = cv2.kmeans(data, kk, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=kk)
    return cen.astype(np.float32), cnt


def sample_hijab_color(frame, lm):
    """لون الحجاب لإطار واحد: أكبر عنقود في الحلقة المحيطة بالوجه."""
    cand = hijab_candidates(frame, lm)
    if cand is None:
        return None
    cen, cnt = cand
    return cen[int(np.argmax(cnt))]


def pick_colors(cands: list):
    """للفيديو: اللون المرجعي = وسيط أكبر العناقيد عبر الإطارات، ثم في كل إطار
    نختار العنقود الأقرب للمرجع (يتجاوز الجدار/البشرة عند الجانب أو الدوران)."""
    ok = [c for c in cands if c is not None]
    if not ok:
        return [None] * len(cands)
    ref = np.median(np.array([cen[int(np.argmax(cnt))] for cen, cnt in ok], np.float32), 0)
    out = []
    for c in cands:
        if c is None:
            out.append(None)
            continue
        cen, cnt = c
        d = np.linalg.norm(cen - ref, axis=1)
        j = int(np.argmin(d))
        out.append(cen[j] if d[j] < 70 else None)
    return out


def render(frame, lm, hands, s: Settings, color_bgr=None):
    """يرسم القناع على frame (يعدّلها مباشرة) ويعيدها."""
    if lm is None:
        return frame
    Ih, Iw = frame.shape[:2]
    poly, r, u, W, H = build_polygon(lm, s)
    sigma = max(0.0, s.feather / 100.0 * 0.20 * W)
    pad = int(4 * sigma + 0.15 * H + 6)
    x0 = int(max(0, np.floor(poly[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(poly[:, 1].min()) - pad))
    x1 = int(min(Iw, np.ceil(poly[:, 0].max()) + pad))
    y1 = int(min(Ih, np.ceil(poly[:, 1].max()) + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    sh = (y1 - y0, x1 - x0)
    org = np.array([x0, y0], np.float32)
    local_poly = poly - org

    stamp_bgr = None
    stamp_fill = None
    if s.stamp or s.edge == "brush":
        if s.stamp:
            a, stamp_bgr = load_stamp(s.stamp)
        else:
            a = brush_alpha()
        loc = to_local(poly, poly.mean(0), r, u)
        amin, bmin = loc.min(0)
        amax, bmax = loc.max(0)
        ex = 1.24
        ca, cb = (amin + amax) / 2, (bmin + bmax) / 2
        ha, hb = (amax - amin) / 2 * ex, (bmax - bmin) / 2 * ex
        cen = poly.mean(0) - org
        P = lambda aa, bb: cen + r * (aa) + u * (bb)
        # زوايا الصورة: أعلى-يسار، أعلى-يمين، أسفل-يسار
        tl, tr, bl = P(ca - ha, cb + hb), P(ca + ha, cb + hb), P(ca - ha, cb - hb)
        ah, aw = a.shape
        src = np.float32([[0, 0], [aw, 0], [0, ah]])
        dst = np.float32([tl, tr, bl])
        M = cv2.getAffineTransform(src, dst)
        m = cv2.warpAffine(a, M, (sh[1], sh[0]), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        if stamp_bgr is not None and not s.stamp_tint:
            pm = (stamp_bgr.astype(np.float32) * (a.astype(np.float32) / 255.0)[..., None])
            wp = cv2.warpAffine(pm, M, (sh[1], sh[0]), flags=cv2.INTER_LINEAR)
            stamp_fill = wp / np.maximum(m[..., None], 1e-3)
        if sigma > 0 and not s.stamp:
            m = cv2.GaussianBlur(m, (0, 0), max(0.6, sigma * 0.25))
    else:
        binm = np.zeros(sh, np.uint8)
        cv2.fillPoly(binm, [np.round(local_poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        rs = max(0.5, 0.05 * H if s.mode != "eyes" else 0.01 * H)
        binm = ((cv2.GaussianBlur(binm, (0, 0), rs) > 127) * 255).astype(np.uint8)
        if sigma > 0.5:
            k = int(1.5 * sigma)
            if k > 0:
                binm = cv2.dilate(binm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            m = cv2.GaussianBlur(binm.astype(np.float32) / 255.0, (0, 0), sigma)
        else:
            m = cv2.GaussianBlur(binm.astype(np.float32) / 255.0, (0, 0), 0.8)

    if hands:
        hm = hand_mask(sh, hands, (x0, y0)).astype(np.float32) / 255.0
        hm = cv2.GaussianBlur(hm, (0, 0), max(1.0, 0.012 * H))
        m = m * (1.0 - hm)
    m *= s.opacity / 100.0

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    if stamp_fill is not None:
        fillimg = stamp_fill
    elif s.fill == "blur":
        sg = max(2.0, (0.02 + s.blur / 100.0 * 0.22) * H)
        fillimg = cv2.GaussianBlur(roi, (0, 0), sg)
    elif s.fill == "pixel":
        bs = max(4, int((0.04 + s.blur / 100.0 * 0.22) * H))
        small = cv2.resize(roi, (max(1, sh[1] // bs), max(1, sh[0] // bs)), interpolation=cv2.INTER_AREA)
        fillimg = cv2.resize(small, (sh[1], sh[0]), interpolation=cv2.INTER_NEAREST)
    else:
        col = color_bgr if (s.fill == "auto" and color_bgr is not None) else hex_to_bgr(s.color)
        fillimg = np.empty_like(roi)
        fillimg[:] = col
    out = roi * (1 - m[..., None]) + fillimg * m[..., None]
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return frame


# ----------------------------------------------------------------------------
# معالجة المسار الكامل: ملء الفجوات + التنعيم
# ----------------------------------------------------------------------------
def fill_gaps(track: list, max_gap: int):
    """track: قائمة lm أو None. تملأ الفجوات بالاستيفاء/التثبيت؛ فجوة أطول من max_gap تُترك None."""
    n = len(track)
    idx = [i for i, t in enumerate(track) if t is not None]
    out = list(track)
    if not idx:
        return out, n
    lost = n - len(idx)
    for i in range(n):
        if track[i] is not None:
            continue
        prev = max([j for j in idx if j < i], default=None)
        nxt = min([j for j in idx if j > i], default=None)
        if prev is not None and nxt is not None:
            if nxt - prev - 1 > max_gap:
                out[i] = None
            else:
                t = (i - prev) / (nxt - prev)
                out[i] = track[prev] * (1 - t) + track[nxt] * t
        else:
            ref = prev if prev is not None else nxt
            out[i] = track[ref] if abs(i - ref) <= max_gap else None
    return out, lost


def smooth_track(track: list, fps: float, smoothing: float):
    """تنعيم OneEuro على الإحداثيات (مقيسة بعرض الوجه كي لا تعتمد على الدقة)."""
    valid = [t for t in track if t is not None]
    if not valid:
        return track
    ref = float(np.median([np.linalg.norm(t[454] - t[234]) for t in valid])) + 1e-6
    min_cut = 3.0 * (0.05 / 3.0) ** (smoothing / 100.0)
    f = OneEuroFilter(min_cutoff=min_cut, beta=12.0, d_cutoff=1.0)
    out = []
    for i, t in enumerate(track):
        if t is None:
            out.append(None)
            continue
        out.append((f(i / fps, t / ref) * ref).astype(np.float32))
    return out


def smooth_colors(cols: list):
    arr = [c for c in cols]
    n = len(arr)
    good = [i for i, c in enumerate(arr) if c is not None]
    if not good:
        return arr
    # استبعاد القيم الشاذة (رقعة وقعت على الخلفية/البشرة) بالنسبة للون الحجاب السائد في الفيديو
    allc = np.array([arr[i] for i in good], np.float32)
    ref = np.median(allc, 0)
    near = np.linalg.norm(allc - ref, axis=1) < 40
    if near.mean() >= 0.4:
        good = [g for g, ok in zip(good, near) if ok]
    ys = np.array([arr[i] for i in good], np.float32)
    xs = np.array(good, np.float32)
    full = np.stack([np.interp(np.arange(n), xs, ys[:, k]) for k in range(3)], 1)
    k = 15
    pad = np.pad(full, ((k, k), (0, 0)), mode="edge")
    # وسيط ثم متوسط متحرك
    med = np.stack([np.median(pad[i:i + 2 * k + 1], 0) for i in range(n)])
    pad2 = np.pad(med, ((k, k), (0, 0)), mode="edge")
    ker = np.hanning(2 * k + 3)[1:-1]
    ker /= ker.sum()
    sm = np.stack([np.convolve(pad2[:, c], ker, "valid") for c in range(3)], 1)
    return [sm[i] for i in range(n)]


# ----------------------------------------------------------------------------
# كاشفات المعاينة (صورة واحدة) — مخزنة ومحمية بقفل
# ----------------------------------------------------------------------------
_img_lock = threading.Lock()
_img_finder = None
_img_hands = None


def detect_still(frame_bgr, want_hands):
    global _img_finder, _img_hands
    with _img_lock:
        if _img_finder is None:
            _img_finder = FaceFinder()
        _img_finder.last = None
        _img_finder.since_scan = 999
        lm = _img_finder.find(frame_bgr)
        hl = []
        if want_hands:
            if _img_hands is None:
                _img_hands = make_hands(False)
            hl = detect_hands(_img_hands, frame_bgr)
        return lm, hl
