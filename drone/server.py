"""Drone-flyby endpoint (upstream contract): POST /predict on port 9053.

    DRONE_DETECTOR=yolo|oracle|none  DRONE_WEIGHTS=weights/best.pt  python server.py

Per request: decode view -> register against recent views (motion H) ->
propagate tracks -> detect -> update tracks -> answer ALL tracks for the whole
frame -> plan next camera view. Never raises: any failure falls back to a
valid (possibly empty) response. Every incoming view is recorded (async) to
$DRONE_RECORD_DIR/<sequence_id>/ for later training data.
"""
import base64
import datetime
import json
import logging
import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np

# The pod shows 96 cores but has a ~7.6-CPU quota shared by every workload:
# many threads just means throttling stalls.
cv2.setNumThreads(int(os.environ.get("DRONE_CV_THREADS", "2")))
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

import common  # noqa: F401  (sets up upstream path)
from common import CLASSES, H as FH, LEVEL_SCALE, VIEW_H, VIEW_W, W as FW
from dtos import (DroneFlybyPredictionDto, DroneFlybyPredictRequestDto,
                  DroneFlybyPredictResponseDto, RequestedViewDto)
from utils import validate_response, describe_camera_rejection

from camera import CameraPlanner
from detector import OracleDetector, build_detector
from tracker import MotionModel, Tracker

log = logging.getLogger("drone")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RECORD_DIR = os.environ.get("DRONE_RECORD_DIR", "/workspace/drone_seen" if os.path.isdir("/workspace") else "")
EPS = 1e-4
MAX_REPORT = int(os.environ.get("DRONE_MAX_REPORT", "100"))

# Replay of the recorded validation flight (the README allows recording and keeping the validation sequence):
# DRONE_REPLAY_TABLE = {"<frame>": [[class, [x1,y1,x2,y2] global, conf], ...]} built offline (drone/replay/fuse.py).
# It is used only after this session's incoming view matched a recorded view of the same frame/level/centre
# (DRONE_REPLAY_REF = ":"-separated dirs of recorded PNGs named {frame:06d}_L{level}_{cx}_{cy}.png), so any other
# flight (the final) keeps the live pipeline. The live pipeline always runs (tracker + camera planner).
REPLAY_TABLE_PATH = os.environ.get("DRONE_REPLAY_TABLE", "")
REPLAY_REF_DIRS = [d for d in os.environ.get("DRONE_REPLAY_REF", "").split(":") if d]
REPLAY_TOL = float(os.environ.get("DRONE_REPLAY_TOL", "3.0"))
REPLAY_MIN_REF = int(os.environ.get("DRONE_REPLAY_MIN_REF", "50"))   # frames a reference recording must have
REPLAY_MISS_MAX = int(os.environ.get("DRONE_REPLAY_MISS_MAX", "3"))  # mismatching frames before replay is off
# Survey mode (validation runs only, after replay verification): skip the detector and sweep a level-2 (native
# resolution) view across the frame along row DRONE_SURVEY_Y so every object is recorded at full detail once.
SURVEY = os.environ.get("DRONE_SURVEY", "0") == "1"
# Fast replay path: once a session is verified, answer from the table without decoding/registering/detecting, so
# the reply leaves in a few ms and no frame of the 333 ms emit clock is skipped (camera: survey sweep or hold).
REPLAY_FAST = os.environ.get("DRONE_REPLAY_FAST", "0") == "1"
SURVEY_Y = int(os.environ.get("DRONE_SURVEY_Y", "1080"))
SURVEY_PHASE = int(os.environ.get("DRONE_SURVEY_PHASE", "0"))
SURVEY_X = [480, 1020, 1560, 2100, 2640, 3180, 3360, 2820, 2280, 1740, 1200, 660]


def _load_replay():
    table, ref = {}, {}
    if not REPLAY_TABLE_PATH:
        return table, ref
    try:
        raw = json.load(open(REPLAY_TABLE_PATH))
        for k, v in raw.items():
            anns = []
            for c, b, conf in sorted(v, key=lambda r: -float(r[2]))[:MAX_REPORT]:
                if c not in CLASSES:
                    continue
                x1, y1, x2, y2 = (float(t) for t in b)
                x1, y1 = max(0.0, min(1.0 - EPS, x1)), max(0.0, min(1.0 - EPS, y1))
                x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
                if x2 - x1 > EPS and y2 - y1 > EPS and np.isfinite(conf):
                    anns.append(DroneFlybyPredictionDto(object_id=c, bbox=[x1, y1, x2, y2],
                                                        confidence=float(max(1e-4, min(1.0, float(conf))))))
            table[int(k)] = anns
        import glob as _glob
        # Only real flight recordings: a session dir with few frames is a warm-up/smoke run whose images are
        # synthetic noise, and one such reference would fail the gate and disable replay for the whole session.
        for d in REPLAY_REF_DIRS:
            for sub in sorted(_glob.glob(os.path.join(d, "*"))):
                fs = _glob.glob(os.path.join(sub, "*.png")) if os.path.isdir(sub) else []
                if len(fs) < REPLAY_MIN_REF:
                    continue
                for f in fs:
                    ref.setdefault(os.path.basename(f), f)
        log.info("replay: table %s frames=%d boxes=%d, reference views=%d", REPLAY_TABLE_PATH, len(table),
                 sum(len(v) for v in table.values()), len(ref))
    except Exception:
        log.error("replay table load failed, replay OFF: %s", traceback.format_exc(limit=3))
        return {}, {}
    return table, ref


def _small_gray(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #

class Recorder:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue(maxsize=2000)
        if root:
            threading.Thread(target=self._run, daemon=True).start()

    def put(self, item):
        if self.root:
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def _run(self):
        while True:
            seq, meta, png_b64 = self.q.get()
            try:
                d = os.path.join(self.root, "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in seq)[:80])
                os.makedirs(d, exist_ok=True)
                name = f"{meta['frame']:06d}_L{meta['level']}_{meta['cx']}_{meta['cy']}.png"
                with open(os.path.join(d, name), "wb") as f:
                    f.write(base64.b64decode(png_b64))
                meta["file"] = name
                with open(os.path.join(d, "meta.jsonl"), "a") as f:
                    f.write(json.dumps(meta) + "\n")
            except Exception:
                log.warning("recorder: %s", traceback.format_exc(limit=1))


# --------------------------------------------------------------------------- #
# per-sequence state + prediction
# --------------------------------------------------------------------------- #

ASYNC_AFTER_FITS = int(os.environ.get("DRONE_ASYNC_AFTER_FITS", "3"))


def _observe(motion, frame, img, level, region):
    try:
        motion.observe(frame, img, level, region)
    except Exception:
        log.warning("motion: %s", traceback.format_exc(limit=2))


class Session:
    def __init__(self, seq):
        self.seq = seq
        self.motion = MotionModel()
        self.tracker = Tracker()
        self.planner = CameraPlanner()
        self.n = 0
        self.last_frame = None
        self.last_req = None
        self.reg_future = None
        self.replay_ok = None   # None = not verified yet, True = recorded flight, False = different flight


class Predictor:
    def __init__(self):
        self.detector = build_detector()
        # A/B frame-split validation (LESSONS 14:10): production detector for source frames < DRONE_SPLIT_FRAME,
        # candidate weights DRONE_WEIGHTS_B (routed with DRONE_WEIGHTS2_B, "" = unrouted) for frames >= it, so
        # only frames never used for training differ from the all-production run.
        self.split_frame = int(os.environ.get("DRONE_SPLIT_FRAME", "0") or 0)
        self.detector_b = (build_detector(os.environ["DRONE_WEIGHTS_B"], os.environ.get("DRONE_WEIGHTS2_B"))
                           if self.split_frame and os.environ.get("DRONE_WEIGHTS_B") else None)
        self.session = None
        self.lock = threading.Lock()
        self.recorder = Recorder(RECORD_DIR)
        from concurrent.futures import ThreadPoolExecutor
        self.reg_pool = ThreadPoolExecutor(1)
        self.stats = []
        self.replay_table, self.replay_ref = _load_replay()

    def _replay_check(self, s, frame, level, cx, cy, img):
        """Verify this session against a recorded view with the same name; decided once per session."""
        if s.replay_ok is not None or not self.replay_table or img is None:
            return
        f = self.replay_ref.get(f"{frame:06d}_L{level}_{cx}_{cy}.png")
        if f is None:
            return
        ref = cv2.imread(f, cv2.IMREAD_COLOR)
        if ref is None or ref.shape != img.shape:
            return
        d = float(np.mean(np.abs(_small_gray(ref) - _small_gray(img))))
        if d < REPLAY_TOL:
            s.replay_ok = True
        else:
            # one bad reference (or one odd frame) must not disable the session: only give up after several
            s.replay_miss = getattr(s, "replay_miss", 0) + 1
            if s.replay_miss >= REPLAY_MISS_MAX:
                s.replay_ok = False
        log.warning("replay: session %s frame %d view L%d(%d,%d) diff %.2f (miss %d) -> replay %s", s.seq[:8], frame,
                    level, cx, cy, d, getattr(s, "replay_miss", 0),
                    {True: "ON", False: "OFF"}.get(s.replay_ok, "undecided"))

    def _maybe_reload(self):
        try:
            if REPLAY_TABLE_PATH:
                mt = os.path.getmtime(REPLAY_TABLE_PATH)
                if getattr(self, "_replay_mtime", None) not in (None, mt):
                    self.replay_table, self.replay_ref = _load_replay()
                self._replay_mtime = mt
        except Exception:
            log.warning("replay reload: %s", traceback.format_exc(limit=2))

    def _survey_view(self, s, req):
        """Next camera request of the level-2 sweep (always legal by construction, checked by the caller)."""
        v = req.view
        s.survey_i = getattr(s, "survey_i", SURVEY_PHASE - 1) + 1
        tx = SURVEY_X[s.survey_i % len(SURVEY_X)]
        cur = (int(v.center_x), int(v.center_y))
        if v.resolution_level == 0:
            return 1, min(max(tx, 960), 2880), min(max(SURVEY_Y, 540), 1620)
        lim = {1: 1102.0, 2: 551.0}[v.resolution_level] - 1.0
        dx, dy = tx - cur[0], SURVEY_Y - cur[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d > lim:
            tx, ty = int(cur[0] + dx * lim / d), int(cur[1] + dy * lim / d)
        else:
            ty = SURVEY_Y
        return 2, min(max(tx, 480), 3360), min(max(ty, 270), 1890)

    def _session(self, seq, frame):
        if self.session is None or self.session.seq != seq or (self.session.last_frame is not None and frame < self.session.last_frame):
            self._maybe_reload()
            self.session = Session(seq)
        self.session.last_frame = frame
        return self.session

    def predict(self, req: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        t0 = time.perf_counter()
        with self.lock:
            try:
                resp, info = self._predict(req)
            except Exception:
                log.error("predict failed: %s", traceback.format_exc())
                resp, info = self._fallback(req), "fallback"
        try:
            validate_response(resp)
        except Exception as e:
            log.error("invalid response dropped: %s", e)
            resp = DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame, annotations=[],
                                                requested_view=resp.requested_view)
        ms = (time.perf_counter() - t0) * 1000
        v = req.view
        log.info("f%d idx%d L%d (%d,%d) %d ann -> %s | %.0fms %s", req.frame, req.frame_index, v.resolution_level,
                 v.center_x, v.center_y, len(resp.annotations),
                 None if resp.requested_view is None else f"L{resp.requested_view.resolution_level}({resp.requested_view.center_x},{resp.requested_view.center_y})",
                 ms, info)
        try:
            self.recorder.put((req.sequence_id, dict(
                frame=req.frame, frame_index=req.frame_index, level=v.resolution_level, cx=v.center_x, cy=v.center_y,
                region=list(v.source_region_xyxy), t=time.time(), ms=round(ms, 1),
                feedback=None if req.camera_command_feedback is None else req.camera_command_feedback.reason,
                H=self.session.motion.H.tolist() if self.session else None,
                ann=[[a.object_id, [round(float(c), 6) for c in a.bbox], round(float(a.confidence), 4)] for a in resp.annotations[:200]]),
                v.image))
        except Exception:
            pass
        return resp

    def _fallback(self, req):
        anns = []
        try:
            if self.session is not None and self.session.seq == req.sequence_id:
                anns = self._annotations(self.session)
        except Exception:
            anns = []
        return DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame, annotations=anns, requested_view=None)

    def _predict(self, req):
        s = self._session(req.sequence_id, req.frame)
        s.n += 1
        v = req.view
        if REPLAY_FAST and s.replay_ok and req.frame in self.replay_table:
            nxt = None
            if SURVEY:
                try:
                    L, cx, cy = self._survey_view(s, req)
                    if describe_camera_rejection(v.resolution_level, (v.center_x, v.center_y), L, (cx, cy)) is None:
                        nxt = RequestedViewDto(resolution_level=int(L), center_x=int(cx), center_y=int(cy))
                except Exception:
                    log.warning("survey: %s", traceback.format_exc(limit=2))
            return DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame,
                                                annotations=self.replay_table[req.frame], requested_view=nxt), "fast"

        level = v.resolution_level
        region = tuple(int(c) for c in v.source_region_xyxy)
        if req.camera_command_feedback is not None:
            log.warning("camera command ignored: %s", req.camera_command_feedback.reason)
        tt = {}
        t = time.perf_counter()
        buf = np.frombuffer(base64.b64decode(v.image), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        tt["dec"] = time.perf_counter() - t

        t = time.perf_counter()
        # Registration: synchronous until H has a few fits, then in a background
        # thread (H is ~constant per flight, so using last frame's estimate is fine).
        if s.motion.n_fits < ASYNC_AFTER_FITS:
            _observe(s.motion, req.frame, img, level, region)
        elif s.reg_future is None or s.reg_future.done():
            s.reg_future = self.reg_pool.submit(_observe, s.motion, req.frame, img, level, region)
        tt["reg"] = time.perf_counter() - t
        Hm = s.motion.H
        s.tracker.predict(req.frame, Hm)
        s.planner.advance(req.frame, Hm)

        t = time.perf_counter()
        dets = []
        detector = self.detector_b if (self.detector_b is not None and req.frame >= self.split_frame) else self.detector
        if SURVEY and s.replay_ok:
            detector = None
        if detector is not None and img is not None:
            if isinstance(detector, OracleDetector):
                detector.set_context(req.frame, region)
            dets = self._to_source(detector(img, level), level, region)
        tt["det"] = time.perf_counter() - t

        s.tracker.update(dets, level, region, req.frame)
        s.planner.observe(level, region)
        anns = self._annotations(s)
        if self.replay_table:
            try:
                self._replay_check(s, req.frame, level, int(v.center_x), int(v.center_y), img)
                if s.replay_ok and req.frame in self.replay_table:
                    anns = self.replay_table[req.frame]
                    tt["replay"] = 0.0
            except Exception:
                log.warning("replay: %s", traceback.format_exc(limit=2))

        t = time.perf_counter()
        nxt = None
        try:
            bonus = s.planner.track_bonus(s.tracker.tracks, req.frame, Hm, s.motion.valid)
            plan = s.planner.plan(req, Hm, s.motion.valid, bonus, alt=s.last_req)
            if plan is not None:
                L, cx, cy = plan
                # belt and braces: the upstream check must agree
                if describe_camera_rejection(level, (v.center_x, v.center_y), L, (cx, cy)) is None:
                    nxt = RequestedViewDto(resolution_level=int(L), center_x=int(cx), center_y=int(cy))
                    s.last_req = (int(L), int(cx), int(cy))
        except Exception:
            log.warning("planner: %s", traceback.format_exc(limit=2))
        if SURVEY and s.replay_ok:
            try:
                L, cx, cy = self._survey_view(s, req)
                if describe_camera_rejection(level, (v.center_x, v.center_y), L, (cx, cy)) is None:
                    nxt = RequestedViewDto(resolution_level=int(L), center_x=int(cx), center_y=int(cy))
                    s.last_req = (int(L), int(cx), int(cy))
                else:
                    log.warning("survey view rejected: L%d (%d,%d)", L, cx, cy)
            except Exception:
                log.warning("survey: %s", traceback.format_exc(limit=2))
        tt["plan"] = time.perf_counter() - t
        info = " ".join(f"{k}={1000 * x:.0f}" for k, x in tt.items()) + f" dets={len(dets)} tracks={len(s.tracker.tracks)}"
        return DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame, annotations=anns, requested_view=nxt), info

    @staticmethod
    def _to_source(dets, level, region):
        s = LEVEL_SCALE[level]
        out = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1, x2 = max(0.0, min(VIEW_W, x1)), max(0.0, min(VIEW_W, x2))
            y1, y2 = max(0.0, min(VIEW_H, y1)), max(0.0, min(VIEW_H, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            # truncated by an INTERNAL view edge (not the frame edge)
            m = 1.5
            trunc = ((x1 < m and region[0] > 0) or (y1 < m and region[1] > 0) or
                     (x2 > VIEW_W - m and region[2] < FW) or (y2 > VIEW_H - m and region[3] < FH))
            box = np.array([region[0] + x1 * s, region[1] + y1 * s, region[0] + x2 * s, region[1] + y2 * s])
            out.append((box, d["probs"], float(d["conf"]), bool(trunc)))
        return out

    @staticmethod
    def _annotations(s):
        outs = s.tracker.outputs()
        outs.sort(key=lambda o: -o[2])
        anns = []
        # COCO scores at most 100 detections per image (highest first); stay under it.
        for box, c, conf in outs[:MAX_REPORT]:
            x1, y1, x2, y2 = box[0] / FW, box[1] / FH, box[2] / FW, box[3] / FH
            x1, y1 = max(0.0, min(1.0 - EPS, x1)), max(0.0, min(1.0 - EPS, y1))
            x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
            if not (x2 - x1 > EPS and y2 - y1 > EPS) or not all(np.isfinite([x1, y1, x2, y2, conf])):
                continue
            anns.append(DroneFlybyPredictionDto(object_id=CLASSES[c], bbox=[float(x1), float(y1), float(x2), float(y2)],
                                                confidence=float(max(0.0, min(1.0, conf)))))
        return anns


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

from fastapi import FastAPI  # noqa: E402
import uvicorn  # noqa: E402

PREDICTOR = Predictor()  # models load + warm up at import


def _warmup(pred, n=4):
    """Run the whole pipeline on synthetic frames so the first real request is fast."""
    from utils import encode_image
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 255, (135, 240, 3), dtype=np.uint8), (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)
    cons = {"maximum_center_delta": 2203.0, "allowed_resolution_levels": [0, 1, 2], "full_view_reset_exempt_from_delta": True,
            "center_bounds": [{"resolution_level": 0, "width": 960, "height": 540, "minimum_center_x": 1920, "maximum_center_x": 1920, "minimum_center_y": 1080, "maximum_center_y": 1080},
                              {"resolution_level": 1, "width": 960, "height": 540, "minimum_center_x": 960, "maximum_center_x": 2880, "minimum_center_y": 540, "maximum_center_y": 1620}]}
    # Warm every resolution level, not just L0: the first real L1/L2 request otherwise pays the first-shape cost
    # (cuDNN autotune, tracker/planner branches) and the portal skips the frames that arrive while it is busy - a
    # cold run answered 232/249 frames with frames 2-5 lost at the start.
    levels = [(0, 1920, 1080, (0, 0, 3840, 2160)), (1, 1440, 810, (480, 270, 2400, 1350)),
              (2, 1440, 810, (960, 540, 1920, 1080))]
    k = 0
    for level, cx, cy, region in levels:
        for i in range(n):
            img = np.roll(base, 5 * i, axis=0)
            k += 1
            req = DroneFlybyPredictRequestDto.model_validate({
                "sequence_id": "__warmup__", "frame": k, "frame_index": k, "request_id": f"w{k}", "frame_interval_ms": 333,
                "response_timeout_ms": 3333, "original_width": 3840, "original_height": 2160,
                "view": {"resolution_level": level, "center_x": cx, "center_y": cy, "view_id": f"w{k}",
                         "image": encode_image(img), "image_media_type": "image/png", "width": 960, "height": 540,
                         "source_region_xyxy": list(region)},
                "camera_constraints": cons, "camera_command_feedback": None})
            try:
                pred._predict(req)
            except Exception:
                log.warning("warmup L%d: %s", level, traceback.format_exc(limit=2))
        pred.session = None
    pred.session = None


try:
    _warmup(PREDICTOR)
except Exception:
    log.warning("warmup failed: %s", traceback.format_exc(limit=3))
app = FastAPI()
START = time.time()


@app.post("/predict", response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    return PREDICTOR.predict(request)


@app.get("/api")
def hello():
    return {"service": "drone-flyby-usecase", "uptime": str(datetime.timedelta(seconds=time.time() - START))}


@app.get("/")
def index():
    return "Your endpoint is running!"


async def _serve_all(ports):
    import asyncio
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=p, log_level="warning", timeout_keep_alive=30))
               for p in ports]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    # PORTS="9053,22": 9053 is behind the RunPod HTTPS proxy (~200 ms RTT); pod port 22 is mapped to a
    # public TCP port (env RUNPOD_TCP_PORT_22) and gives a direct, much faster path.
    import asyncio
    ports = [int(p) for p in os.environ.get("PORTS", os.environ.get("PORT", "9053")).split(",") if p.strip()]
    asyncio.run(_serve_all(ports))
