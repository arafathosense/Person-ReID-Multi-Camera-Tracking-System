"""
Person ReID & Multi-Camera Tracking System
==========================================
Tracks people across multiple video feeds with consistent global IDs.

Usage:
    python reid_multicam.py --cams cam1.mp4 cam2.mp4
    python reid_multicam.py --cams cam1.mp4 cam2.mp4 --width 640 --threshold 0.65
"""

# ══════════════════════════════════════════════════════════════════
#  PHASE 0 — AUTO-FIX DEPS + SELF-RESTART
# ══════════════════════════════════════════════════════════════════
import subprocess, sys, os, warnings
warnings.filterwarnings("ignore")

_FLAG = "__REID_DEPS_OK__"

def _pip(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--no-warn-script-location", *pkgs],
        stderr=subprocess.DEVNULL
    )

if _FLAG not in os.environ:
    print("=" * 62)
    print("  ReID Multi-Camera Tracker — Fixing environment...")
    print("=" * 62)
    _pip("numpy<2", "--force-reinstall")
    _pip("protobuf==3.20.3")
    _pip("opencv-python", "scipy")
    _pip("torch", "torchvision", "--index-url",
         "https://download.pytorch.org/whl/cpu")
    _pip("ultralytics")
    print("[INFO] Dependencies ready — restarting...\n")
    env = os.environ.copy()
    env[_FLAG] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

# ══════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════
import cv2
import numpy as np
import argparse, math, time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
PROC_W          = 960        # processing width
PERSON_CLASS    = 0          # COCO person
DET_CONF        = 0.40       # YOLO min confidence
REID_THRESHOLD  = 0.65       # cosine similarity threshold for same ID
EMB_GALLERY_MAX = 8          # max embeddings kept per global ID
TRAJ_LEN        = 80         # max trajectory points stored
IOU_THRESHOLD   = 0.30       # IoU for local tracker matching
MAX_MISSING     = 25         # frames before track is dropped
HEATMAP_ALPHA   = 0.45       # heatmap overlay transparency
DWELL_FPS_EST   = 25.0       # fallback fps for dwell time

# Colour scheme — CCTV analytics look (BGR)
C_BOX     = ( 20, 220,  80)   # bounding box green
C_TEXT_BG = (  0,   0,   0)
C_TRAJ    = ( 50, 200, 255)   # trajectory cyan
C_LABEL   = (220, 255,  80)
C_ENTRY   = (  0, 255, 100)
C_EXIT    = (  0,  80, 255)
C_GRID    = ( 40,  40,  40)
C_HUD_BG  = (  0,   0,   0)
C_HUD_ACC = ( 20, 220,  80)
C_DIM     = (100, 100, 100)

# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════
@dataclass
class Track:
    local_id:    int
    global_id:   int
    bbox:        Tuple[int,int,int,int]   # x1,y1,x2,y2
    embedding:   Optional[np.ndarray]
    trajectory:  deque = field(default_factory=lambda: deque(maxlen=TRAJ_LEN))
    missing:     int   = 0
    frame_first: int   = 0
    frame_last:  int   = 0
    cam_id:      str   = ""

    @property
    def center(self):
        x1,y1,x2,y2 = self.bbox
        return ((x1+x2)//2, (y1+y2)//2)

    @property
    def dwell_frames(self):
        return max(0, self.frame_last - self.frame_first)


class GlobalIDManager:
    """Manages cross-camera identity matching via embedding gallery."""
    def __init__(self, threshold=REID_THRESHOLD):
        self.threshold  = threshold
        self.gallery: Dict[int, List[np.ndarray]] = {}   # gid -> [embeddings]
        self._next_gid  = 1

    def assign(self, embedding: np.ndarray) -> int:
        """Return best matching global_id or create new one."""
        if embedding is None or len(self.gallery) == 0:
            return self._new_id(embedding)

        best_gid, best_sim = -1, -1.0
        for gid, embs in self.gallery.items():
            # compare against mean embedding of gallery
            mean_emb = np.mean(embs, axis=0)
            sim = float(np.dot(embedding, mean_emb) /
                        (np.linalg.norm(embedding) * np.linalg.norm(mean_emb) + 1e-8))
            if sim > best_sim:
                best_sim, best_gid = sim, gid

        if best_sim >= self.threshold:
            self._update_gallery(best_gid, embedding)
            return best_gid
        return self._new_id(embedding)

    def _new_id(self, embedding):
        gid = self._next_gid
        self._next_gid += 1
        self.gallery[gid] = [embedding] if embedding is not None else []
        return gid

    def _update_gallery(self, gid, embedding):
        self.gallery[gid].append(embedding)
        if len(self.gallery[gid]) > EMB_GALLERY_MAX:
            self.gallery[gid].pop(0)

# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — PERSON DETECTION
# ══════════════════════════════════════════════════════════════════
print("[SETUP] Loading YOLOv8n...")
_yolo = YOLO("yolov8n.pt")

def detect_people(frame: np.ndarray) -> List[Tuple]:
    """
    Returns list of (x1,y1,x2,y2,conf) for all detected persons.
    """
    results = _yolo(frame, verbose=False, conf=DET_CONF, classes=[PERSON_CLASS])
    dets = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == PERSON_CLASS:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # clamp to frame
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                if x2 > x1 and y2 > y1:
                    dets.append((x1,y1,x2,y2,conf))
    return dets

# ══════════════════════════════════════════════════════════════════
#  MODULE 2 — LOCAL TRACKER  (IoU-based byte-style tracker)
# ══════════════════════════════════════════════════════════════════
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a[:4]
    bx1,by1,bx2,by2 = b[:4]
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    ua    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)


def track_objects(
    dets: List[Tuple],
    tracks: Dict[int, Track],
    next_local_id: List[int],
    frame_idx: int,
    cam_id: str,
    gid_manager: GlobalIDManager,
    embeddings: Dict[int, np.ndarray]
) -> Dict[int, Track]:
    """
    Simple IoU-based tracker.  Matches detections to existing tracks,
    creates new tracks, marks missing ones.
    Returns updated tracks dict.
    """
    if not dets:
        for t in tracks.values():
            t.missing += 1
        return {lid: t for lid,t in tracks.items() if t.missing <= MAX_MISSING}

    active = [t for t in tracks.values() if t.missing == 0]

    if not active:
        # all new
        for det in dets:
            x1,y1,x2,y2,conf = det
            lid = next_local_id[0]; next_local_id[0] += 1
            emb = embeddings.get(len(tracks))  # placeholder
            gid = gid_manager.assign(emb)
            t   = Track(lid, gid, (x1,y1,x2,y2), emb,
                        frame_first=frame_idx, frame_last=frame_idx, cam_id=cam_id)
            t.trajectory.append(t.center)
            tracks[lid] = t
        return tracks

    # build IoU cost matrix
    cost = np.zeros((len(active), len(dets)))
    for i, t in enumerate(active):
        for j, d in enumerate(dets):
            cost[i,j] = 1.0 - _iou(t.bbox, d)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_t = set(); matched_d = set()
    for r,c in zip(row_ind, col_ind):
        if cost[r,c] < (1.0 - IOU_THRESHOLD):
            t = active[r]
            x1,y1,x2,y2,_ = dets[c]
            t.bbox       = (x1,y1,x2,y2)
            t.missing    = 0
            t.frame_last = frame_idx
            t.trajectory.append(t.center)
            matched_t.add(t.local_id)
            matched_d.add(c)

    # mark unmatched tracks
    for t in active:
        if t.local_id not in matched_t:
            t.missing += 1

    # create new tracks for unmatched detections
    for j, det in enumerate(dets):
        if j not in matched_d:
            x1,y1,x2,y2,conf = det
            lid = next_local_id[0]; next_local_id[0] += 1
            emb = embeddings.get(j)
            gid = gid_manager.assign(emb)
            t   = Track(lid, gid, (x1,y1,x2,y2), emb,
                        frame_first=frame_idx, frame_last=frame_idx, cam_id=cam_id)
            t.trajectory.append(t.center)
            tracks[lid] = t

    # prune lost tracks
    return {lid: t for lid,t in tracks.items() if t.missing <= MAX_MISSING}

# ══════════════════════════════════════════════════════════════════
#  MODULE 3 — EMBEDDING EXTRACTION  (ResNet18 backbone)
# ══════════════════════════════════════════════════════════════════
print("[SETUP] Loading ResNet18 for ReID embeddings...")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_resnet.fc = torch.nn.Identity()   # remove classifier head → 512-d
_resnet = _resnet.to(_device).eval()

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def extract_embedding(frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
    """Crop person patch, run through ResNet18, return L2-normalised embedding."""
    x1,y1,x2,y2 = bbox[:4]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
        return None
    try:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor   = _transform(crop_rgb).unsqueeze(0).to(_device)
        with torch.no_grad():
            emb = _resnet(tensor).squeeze(0).cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
#  MODULE 4 — IDENTITY MATCHING  (cross-camera ReID)
# ══════════════════════════════════════════════════════════════════
def match_identities(
    tracks: Dict[int, Track],
    gid_manager: GlobalIDManager,
    frame: np.ndarray
):
    """Re-run ReID for each active track and update global_id if needed."""
    for t in tracks.values():
        if t.missing > 0:
            continue
        emb = extract_embedding(frame, t.bbox)
        if emb is None:
            continue
        t.embedding = emb
        new_gid = gid_manager.assign(emb)
        t.global_id = new_gid

# ══════════════════════════════════════════════════════════════════
#  MODULE 5 — TRAJECTORY UPDATE
# ══════════════════════════════════════════════════════════════════
def update_trajectories(tracks: Dict[int, Track]):
    """Append current center to each active track's trajectory deque."""
    for t in tracks.values():
        if t.missing == 0:
            t.trajectory.append(t.center)

# ══════════════════════════════════════════════════════════════════
#  MODULE 6 — HEATMAP GENERATION
# ══════════════════════════════════════════════════════════════════
def generate_heatmap(
    accum: np.ndarray,
    frame: np.ndarray,
    alpha: float = HEATMAP_ALPHA
) -> np.ndarray:
    """
    Blend a colour heatmap onto `frame` using accumulated position density.
    accum: float32 array, same H×W as frame, values accumulate over time.
    """
    if accum.max() < 1e-6:
        return frame
    # log-scale normalise for better visual range
    norm  = np.log1p(accum)
    norm  = (norm / (norm.max() + 1e-8) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # mask near-zero areas
    mask  = (norm > 5).astype(np.uint8)[:,:,None]
    blend = cv2.addWeighted(frame, 1.0, color * mask, alpha, 0)
    return blend

# ══════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════
_PALETTE = [
    (  0, 230, 100), (  0, 180, 255), (230, 120,   0),
    (200,   0, 200), (  0, 220, 220), (255, 160,   0),
    (100, 255,   0), (  0, 100, 255), (255,  60, 120),
    ( 60, 200, 180),
]

def _gid_color(gid: int):
    return _PALETTE[(gid - 1) % len(_PALETTE)]

def _put_label(img, text, pos, color, scale=0.46, thick=1):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, y-th-4), (x+tw+4, y+2), C_TEXT_BG, -1)
    cv2.putText(img, text, (x+2, y-1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_track(frame, t: Track, fps: float, show_traj=True):
    if t.missing > 0:
        return
    x1,y1,x2,y2 = t.bbox
    col = _gid_color(t.global_id)

    # bounding box
    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
    # corner brackets
    b = 10
    for (cx,cy),(sx,sy) in [
        ((x1,y1),(1,1)), ((x2,y1),(-1,1)),
        ((x1,y2),(1,-1)), ((x2,y2),(-1,-1))
    ]:
        cv2.line(frame,(cx,cy),(cx+sx*b,cy),col,2)
        cv2.line(frame,(cx,cy),(cx,cy+sy*b),col,2)

    # ID label
    dwell_s = t.dwell_frames / max(fps, 1.0)
    label = f"ID:{t.global_id:02d}  {dwell_s:.0f}s"
    _put_label(frame, label, (x1, y1), col)

    # trajectory
    if show_traj and len(t.trajectory) >= 2:
        pts = list(t.trajectory)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c     = tuple(int(v * alpha) for v in col)
            cv2.line(frame, pts[i-1], pts[i], c, 2, cv2.LINE_AA)
        # head dot
        cv2.circle(frame, pts[-1], 4, col, -1)


def draw_hud(frame, cam_id: str, n_active: int, n_global: int,
             entries: int, exits: int, fidx: int, fps: float):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # top-left panel
    pw, ph = 280, 116
    ov = frame.copy()
    cv2.rectangle(ov, (6,6), (6+pw, 6+ph), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (6,6), (6+pw, 6+ph), C_HUD_ACC, 1)

    # corner brackets
    b = 12
    for (x,y),(sx,sy) in [((6,6),(1,1)),((6+pw,6),(-1,1)),
                           ((6,6+ph),(1,-1)),((6+pw,6+ph),(-1,-1))]:
        cv2.line(frame,(x,y),(x+sx*b,y),C_HUD_ACC,2)
        cv2.line(frame,(x,y),(x,y+sy*b),C_HUD_ACC,2)

    cv2.putText(frame, f"\u25cf  {cam_id.upper()}", (14,26),  font,0.44,C_HUD_ACC,1,cv2.LINE_AA)
    cv2.putText(frame, f"TRACKED  : {n_active:>3}",  (14,48),  font,0.40,(180,255,180),1,cv2.LINE_AA)
    cv2.putText(frame, f"GLOBAL IDs: {n_global:>3}", (14,68),  font,0.40,C_LABEL,     1,cv2.LINE_AA)
    cv2.putText(frame, f"ENTRIES  : {entries:>3}",   (14,88),  font,0.40,C_ENTRY,     1,cv2.LINE_AA)
    cv2.putText(frame, f"EXITS    : {exits:>3}",     (14,108), font,0.40,C_EXIT,      1,cv2.LINE_AA)

    # frame counter bottom-right
    cv2.putText(frame, f"FRAME {fidx:05d}",
                (w-155, h-10), font, 0.36, C_DIM, 1, cv2.LINE_AA)

    # subtle scanline effect (every 4 rows)
    overlay = frame.copy()
    for y in range(0, h, 4):
        cv2.line(overlay, (0,y), (w,y), (0,0,0), 1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

# ══════════════════════════════════════════════════════════════════
#  ENTRY / EXIT COUNTER
# ══════════════════════════════════════════════════════════════════
class EntryExitCounter:
    """Counts persons entering (appearing) and exiting (disappearing)."""
    def __init__(self):
        self.seen:    set = set()    # global_ids currently active
        self.entries: int = 0
        self.exits:   int = 0

    def update(self, active_gids: set):
        for gid in active_gids - self.seen:
            self.entries += 1
        for gid in self.seen - active_gids:
            self.exits   += 1
        self.seen = set(active_gids)

# ══════════════════════════════════════════════════════════════════
#  PER-CAMERA PROCESSOR
# ══════════════════════════════════════════════════════════════════
class CameraProcessor:
    def __init__(self, cam_id: str, video_path: str,
                 gid_manager: GlobalIDManager, proc_w: int):
        self.cam_id       = cam_id
        self.video_path   = video_path
        self.gid_manager  = gid_manager
        self.proc_w       = proc_w
        self.tracks:  Dict[int, Track] = {}
        self.next_lid = [1]
        self.counter  = EntryExitCounter()
        self.heatmap_accum: Optional[np.ndarray] = None

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        ow  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oh  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or DWELL_FPS_EST
        self.total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale       = proc_w / ow
        self.pw     = proc_w
        self.ph     = int(oh * scale)
        self.heatmap_accum = np.zeros((self.ph, self.pw), dtype=np.float32)

        out_path    = f"output_{cam_id}.mp4"
        fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.pw, self.ph))
        self.out_path = out_path
        self.fidx     = 0

        print(f"  [{cam_id}] {ow}x{oh} @ {self.fps:.1f}fps  {self.total} frames → {out_path}")

    def step(self) -> bool:
        """Process one frame. Returns False when video ends."""
        ret, frame = self.cap.read()
        if not ret:
            return False
        frame = cv2.resize(frame, (self.pw, self.ph))

        # 1. Detect
        dets = detect_people(frame)

        # 2. Extract embeddings for all detections
        emb_map = {}
        for i, det in enumerate(dets):
            emb_map[i] = extract_embedding(frame, det[:4])

        # 3. Track (IoU-based)
        self.tracks = track_objects(
            dets, self.tracks, self.next_lid,
            self.fidx, self.cam_id, self.gid_manager, emb_map
        )

        # 4. ReID — update global IDs
        match_identities(self.tracks, self.gid_manager, frame)

        # 5. Trajectories
        update_trajectories(self.tracks)

        # 6. Heatmap accumulate
        for t in self.tracks.values():
            if t.missing == 0:
                cx, cy = t.center
                if 0 <= cy < self.ph and 0 <= cx < self.pw:
                    cv2.circle(self.heatmap_accum.view(np.float32).reshape(self.ph, self.pw)
                               if False else self.heatmap_accum,
                               (cx,cy), 20, 1.0, -1)   # splat a blob

        # 7. Render
        vis = generate_heatmap(self.heatmap_accum, frame.copy())
        for t in self.tracks.values():
            draw_track(vis, t, self.fps)

        active_gids = {t.global_id for t in self.tracks.values() if t.missing == 0}
        self.counter.update(active_gids)

        draw_hud(vis, self.cam_id,
                 n_active  = len(active_gids),
                 n_global  = len(self.gid_manager.gallery),
                 entries   = self.counter.entries,
                 exits     = self.counter.exits,
                 fidx      = self.fidx,
                 fps       = self.fps)

        self.writer.write(vis)
        self.fidx += 1
        return True

    def close(self):
        self.cap.release()
        self.writer.release()

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Person ReID Multi-Camera Tracker")
    ap.add_argument("--cams",      nargs="+", default=["cam1.mp4","cam2.mp4"],
                    help="Input video files (one per camera)")
    ap.add_argument("--width",     type=int,   default=PROC_W)
    ap.add_argument("--threshold", type=float, default=REID_THRESHOLD,
                    help="Cosine similarity threshold for ReID (0-1)")
    args = ap.parse_args()

    # validate inputs
    for f in args.cams:
        if not os.path.exists(f):
            print(f"\n[ERROR] Video not found: '{f}'")
            print("  → Place camera videos in the same folder.")
            print("  → Usage: python reid_multicam.py --cams cam1.mp4 cam2.mp4\n")
            sys.exit(1)

    print("\n" + "="*62)
    print("  Person ReID & Multi-Camera Tracking System")
    print("="*62)
    print(f"  Cameras   : {args.cams}")
    print(f"  Width     : {args.width}")
    print(f"  ReID thr  : {args.threshold}")
    print(f"  Device    : {_device}")
    print("="*62 + "\n")

    # shared identity manager (cross-camera)
    gid_manager = GlobalIDManager(threshold=args.threshold)

    # build per-camera processors
    processors: List[CameraProcessor] = []
    for i, path in enumerate(args.cams):
        cam_id = f"cam{i+1}"
        try:
            processors.append(CameraProcessor(cam_id, path, gid_manager, args.width))
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    max_frames = max(p.total for p in processors)
    t0 = time.time()

    print(f"\n[INFO] Processing {max_frames} frames across {len(processors)} camera(s)...\n")

    for fidx in range(max_frames):
        any_alive = False
        for proc in processors:
            alive = proc.step()
            if alive:
                any_alive = True

        if not any_alive:
            break

        if fidx % 15 == 0:
            elapsed = time.time() - t0
            pct = (fidx+1) / max(max_frames,1) * 100
            bar = "\u2588" * int(pct/5) + "\u2591" * (20-int(pct/5))
            fps_proc = (fidx+1) / max(elapsed, 0.01)
            print(f"\r  [{bar}] {pct:5.1f}%   frame {fidx+1:>5}/{max_frames}"
                  f"   {fps_proc:.1f} fps   GIDs:{len(gid_manager.gallery):>3}",
                  end="", flush=True)

    for proc in processors:
        proc.close()

    elapsed = time.time() - t0
    print(f"\n\n{'='*62}")
    print(f"  DONE in {elapsed:.1f}s")
    for proc in processors:
        print(f"  → {os.path.abspath(proc.out_path)}")
    print(f"  Total global identities assigned: {len(gid_manager.gallery)}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()