import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Deque
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# Optional: import your SORT implementation
from sort import Sort  # expects API: update(dets) -> [[x1,y1,x2,y2,track_id], ...]

"""
ROI Canvas Demo (tracker-driven ROI selection + fixed-size canvas packing)
-----------------------------------------------------------------------
- Keyframes: full-frame detection initializes/refreshes tracks.
- Intermediate frames: use tracker motion to PREDICT ROIs (with padding),
  pack them into a FIXED-SIZE canvas, run one inference on the canvas,
  then map detections back to original camera frames and update trackers.

Highlights
- Tracker-driven ROI selection with padding (predict next-frame ROIs).
- Fixed-size canvas (CANVAS_W x CANVAS_H). Tiles are computed as a grid that
  fits N ROIs; each ROI is letterboxed into its tile (aspect preserved).
- Clean coordinate back-projection from canvas -> tile -> original frame.

You can integrate this class into your CARLA loop by feeding {cam_name: frame} dicts
into ROICanvasDemo.step(frames).
"""

# =============================
# Configurations
# =============================
KEYFRAME_INTERVAL = 5           # run full-frame detection every K frames
ROI_PADDING = 16                # pixels padded around predicted ROI
MAX_ROIS_PER_FRAME = 48         # safety cap across all cameras per frame
CANVAS_W, CANVAS_H = 1024, 1024 # FIXED canvas size
SHOW_WINDOWS = True             # set False if running headless

# Limit classes to road-relevant targets (COCO ids)
# person=0, car=2, motorcycle=3, bus=5, truck=7, traffic light=9
ALLOWED_CLS = {0, 1, 2, 3, 5, 7, 9}

# =============================
# Data structures
# =============================
@dataclass
class ROIMapping:
    cam_name: str
    # ROI in original camera frame (top-left + size)
    x: int
    y: int
    w: int
    h: int
    # Tile placement on the canvas
    cx: int  # tile top-left x on canvas
    cy: int  # tile top-left y on canvas
    tile_w: int
    tile_h: int
    # Letterbox parameters inside the tile
    scale: float   # isotropic scale from ROI -> tile content
    pad_x: int     # left padding inside tile
    pad_y: int     # top padding inside tile

# =============================
# Utilities
# =============================

def clamp_rect(x, y, w, h, W, H):
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > W:
        w = max(1, W - x)
    if y + h > H:
        h = max(1, H - y)
    return x, y, w, h


def expand_with_padding(x1, y1, x2, y2, pad, W, H):
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(W - 1, x2 + pad))
    y2 = int(min(H - 1, y2 + pad))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return x1, y1, w, h


def is_keyframe(frame_index: int, k: int = KEYFRAME_INTERVAL) -> bool:
    return (frame_index % k) == 0

# =============================
# Fixed-size canvas packing (grid + letterbox)
# =============================

def _grid_for_n(n: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Compute (rows, cols, tile_w, tile_h) for up to n tiles on fixed canvas W x H."""
    n = max(1, n)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tile_w = W // cols
    tile_h = H // rows
    return rows, cols, tile_w, tile_h


def pack_rois_fixed_canvas(
    roi_items: List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]],
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> Tuple[np.ndarray, List[ROIMapping]]:
    """
    Pack variable-size ROIs into a FIXED canvas (canvas_w x canvas_h).
    - Each ROI is letterboxed to its tile (aspect preserved).
    - Returns (canvas_bgr, mappings).

    roi_items: list of (cam_name, frame_bgr, (x, y, w, h))
    """
    if not roi_items:
        return None, []

    n = min(len(roi_items), MAX_ROIS_PER_FRAME)
    rows, cols, tile_w, tile_h = _grid_for_n(n, canvas_w, canvas_h)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 114, dtype=np.uint8)

    mappings: List[ROIMapping] = []
    for idx, (cam_name, frame, (x, y, w, h)) in enumerate(roi_items[:n]):
        H, W = frame.shape[:2]
        x, y, w, h = clamp_rect(x, y, w, h, W, H)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        # letterbox into tile (preserve aspect)
        s = min(tile_w / float(w), tile_h / float(h))
        new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (tile_w - new_w) // 2
        pad_y = (tile_h - new_h) // 2

        row = idx // cols
        col = idx % cols
        cx = col * tile_w
        cy = row * tile_h

        tile = canvas[cy:cy + tile_h, cx:cx + tile_w]
        tile[...] = 114
        tile[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        mappings.append(
            ROIMapping(
                cam_name=cam_name,
                x=x, y=y, w=w, h=h,
                cx=cx, cy=cy,
                tile_w=tile_w, tile_h=tile_h,
                scale=s, pad_x=pad_x, pad_y=pad_y,
            )
        )

    return canvas, mappings


def assign_dets_to_tiles(dets_xyxy: np.ndarray, mappings: List[ROIMapping]) -> Dict[int, List[np.ndarray]]:
    """Assign detections (canvas coords) to tile indices by center point."""
    buckets: Dict[int, List[np.ndarray]] = {}
    if dets_xyxy is None or dets_xyxy.size == 0:
        return buckets

    for det in dets_xyxy:
        x1, y1, x2, y2, conf, cls_id = det
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # find mapping where (cx,cy) falls into the tile rect
        matched = -1
        for i, m in enumerate(mappings):
            if (m.cx <= cx < m.cx + m.tile_w) and (m.cy <= cy < m.cy + m.tile_h):
                matched = i
                break
        if matched == -1:
            continue
        # store det in tile-local coords
        m = mappings[matched]
        buckets.setdefault(matched, []).append(np.array([
            x1 - m.cx, y1 - m.cy, x2 - m.cx, y2 - m.cy, conf, cls_id
        ], dtype=np.float32))
    return buckets


def remap_tile_dets_to_original(
    buckets: Dict[int, List[np.ndarray]],
    mappings: List[ROIMapping]
) -> Dict[str, np.ndarray]:
    """
    Convert tile-local detections back to original camera frames by inverting
    the letterbox transform and ROI placement.
    Returns dict: cam_name -> ndarray [x1,y1,x2,y2,conf,cls]
    """
    per_cam: Dict[str, List[List[float]]] = defaultdict(list)

    for idx, det_list in buckets.items():
        m = mappings[idx]
        s = m.scale
        px, py = m.pad_x, m.pad_y
        for det in det_list:
            tx1, ty1, tx2, ty2, conf, cls_id = det
            # remove letterbox padding, then scale back to ROI
            rx1 = (tx1 - px) / s
            ry1 = (ty1 - py) / s
            rx2 = (tx2 - px) / s
            ry2 = (ty2 - py) / s
            # map from ROI-local to original frame
            ox1 = m.x + rx1
            oy1 = m.y + ry1
            ox2 = m.x + rx2
            oy2 = m.y + ry2
            per_cam[m.cam_name].append([ox1, oy1, ox2, oy2, float(conf), float(cls_id)])

    out: Dict[str, np.ndarray] = {}
    for cam, lst in per_cam.items():
        out[cam] = np.array(lst, dtype=np.float32) if lst else np.empty((0, 6), dtype=np.float32)
    return out

# =============================
# Demo pipeline core
# =============================
class ROICanvasDemo:
    def __init__(self, cam_names: List[str], img_size: Tuple[int, int] = (640, 480)):
        self.model = YOLO('yolov8m.pt')
        self.trackers = {name: Sort(max_age=10, min_hits=2, iou_threshold=0.3) for name in cam_names}
        self.prev_tracks: Dict[str, np.ndarray] = {name: np.empty((0, 5), dtype=np.float32) for name in cam_names}
        self.frame_index = 0
        self.img_w, self.img_h = img_size
        # per camera, per track-id -> deque of last two centers for velocity
        self.track_hist: Dict[str, Dict[int, Deque[Tuple[float, float, int]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=2)))

    # -------------- Detection wrappers --------------
    def detect_full(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame, verbose=False)[0]
        if res.boxes.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        cls  = res.boxes.cls.cpu().numpy().astype(np.int32).reshape(-1, 1)
        keep = np.array([c in ALLOWED_CLS for c in cls.flatten()], dtype=bool)
        if not keep.any():
            return np.empty((0, 5), dtype=np.float32)
        xyxy = xyxy[keep]; conf = conf[keep]
        return np.hstack((xyxy, conf))

    def _update_track_history(self, cam_name: str, tracks: np.ndarray):
        """Keep last two centers per track for simple constant-velocity prediction."""
        for x1, y1, x2, y2, tid in tracks:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            self.track_hist[cam_name][int(tid)].append((cx, cy, self.frame_index))

    def update_tracker(self, cam_name: str, dets_xyxy5: np.ndarray) -> np.ndarray:
        tracks = self.trackers[cam_name].update(dets_xyxy5)  # [x1,y1,x2,y2,tid]
        self.prev_tracks[cam_name] = tracks.copy()
        self._update_track_history(cam_name, tracks)
        return tracks

    # -------------- Tracker-driven ROI prediction --------------
    def _predict_next_box(self, cam_name: str, trk: np.ndarray) -> Tuple[int, int, int, int]:
        """Predict next-frame ROI using constant-velocity from last two centers.
        Fallback to last box if not enough history. Add ROI_PADDING.
        """
        x1, y1, x2, y2, tid = trk
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = x2 - x1
        h = y2 - y1
        deq = self.track_hist[cam_name].get(int(tid), None)
        if deq and len(deq) == 2:
            (cx0, cy0, f0), (cx1, cy1, f1) = deq[0], deq[1]
            dt = max(1, f1 - f0)
            vx = (cx1 - cx0) / dt
            vy = (cy1 - cy0) / dt
            cx_pred = cx + vx  # predict 1 frame ahead
            cy_pred = cy + vy
        else:
            cx_pred, cy_pred = cx, cy
        x1p = cx_pred - 0.5 * w
        y1p = cy_pred - 0.5 * h
        x2p = cx_pred + 0.5 * w
        y2p = cy_pred + 0.5 * h
        # pad & clamp to frame size
        H, W = self.img_h, self.img_w
        x, y, ww, hh = expand_with_padding(x1p, y1p, x2p, y2p, ROI_PADDING, W, H)
        return x, y, ww, hh

    def get_predicted_rois(self, cam_name: str, frame_shape: Tuple[int, int, int]) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
        rois = []
        tracks = self.prev_tracks.get(cam_name, np.empty((0, 5), dtype=np.float32))
        for trk in tracks:
            x, y, w, h = self._predict_next_box(cam_name, trk)
            rois.append((cam_name, None, (x, y, w, h)))
        return rois

    def _priority(self, cam_name: str, trk: np.ndarray) -> float:
        """Simple priority: speed-first then area."""
        x1, y1, x2, y2, tid = trk
        area = max(1.0, (x2 - x1) * (y2 - y1))
        deq = self.track_hist[cam_name].get(int(tid), None)
        speed = 0.0
        if deq and len(deq) == 2:
            (cx0, cy0, f0), (cx1, cy1, f1) = deq[0], deq[1]
            dt = max(1, f1 - f0)
            speed = np.hypot(cx1 - cx0, cy1 - cy0) / dt
        # weight: speed dominates, area as tie-breaker
        return 5.0 * speed + 0.001 * area

    def build_roi_items(self, frames: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
        items = []
        scored: List[Tuple[float, str, np.ndarray, Tuple[int, int, int, int]]] = []
        for cam_name, frame in frames.items():
            tracks = self.prev_tracks.get(cam_name, np.empty((0, 5), dtype=np.float32))
            for trk in tracks:
                x, y, w, h = self._predict_next_box(cam_name, trk)
                prio = self._priority(cam_name, trk)
                scored.append((prio, cam_name, frame, (x, y, w, h)))
        # select top-K by priority
        scored.sort(key=lambda t: t[0], reverse=True)
        for _, cam_name, frame, rect in scored[:MAX_ROIS_PER_FRAME]:
            items.append((cam_name, frame, rect))
        return items

    # -------------- Main steps --------------
    def process_keyframe(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        per_cam_dets: Dict[str, np.ndarray] = {}
        for cam_name, frame in frames.items():
            t0 = time.perf_counter()
            dets = self.detect_full(frame)
            print(f"[KEY {cam_name}] full detect {1000*(time.perf_counter()-t0):6.1f} ms, dets={dets.shape[0]}")
            tracks = self.update_tracker(cam_name, dets)
            # optional visualize
            if SHOW_WINDOWS:
                vis = frame.copy()
                for x1, y1, x2, y2, s in dets:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                for x1, y1, x2, y2, tid in tracks:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                    cv2.putText(vis, f'ID {int(tid)}', (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.imshow(f'{cam_name}-key', vis)
            per_cam_dets[cam_name] = dets
        return per_cam_dets

    def process_intermediate(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # build predicted ROIs from trackers (all cams), prioritize, cap to K
        roi_items = self.build_roi_items(frames)
        if not roi_items:
            return {name: np.empty((0, 6), dtype=np.float32) for name in frames.keys()}

        # pack into FIXED canvas
        canvas, mappings = pack_rois_fixed_canvas(roi_items, CANVAS_W, CANVAS_H)
        if SHOW_WINDOWS:
            canvas_vis = canvas.copy()
            # draw tile grid and labels
            for i, m in enumerate(mappings):
                cv2.rectangle(canvas_vis, (m.cx, m.cy), (m.cx + m.tile_w, m.cy + m.tile_h), (60, 60, 60), 1)
                cv2.putText(canvas_vis, f"{i}:{m.cam_name}", (m.cx + 4, m.cy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow('ROI Canvas', canvas_vis)

        # single inference on canvas
        res = self.model.predict(canvas, verbose=False)[0]
        if res.boxes.shape[0] == 0:
            canvas_dets = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
            cls  = res.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
            # class filtering on canvas dets
            cls_i = cls.astype(np.int32).flatten()
            keep = np.array([c in ALLOWED_CLS for c in cls_i], dtype=bool)
            if not keep.any():
                canvas_dets = np.empty((0, 6), dtype=np.float32)
            else:
                canvas_dets = np.hstack((xyxy[keep], conf[keep], cls[keep]))

        # split & remap detections back to original frames
        buckets = assign_dets_to_tiles(canvas_dets, mappings)
        remapped = remap_tile_dets_to_original(buckets, mappings)

        # update trackers with remapped dets
        for cam_name in frames.keys():
            dets5 = remapped.get(cam_name, np.empty((0, 6), dtype=np.float32))
            dets5 = dets5[:, :5] if dets5.size > 0 else dets5  # SORT expects [x1,y1,x2,y2,score]
            tracks = self.update_tracker(cam_name, dets5)
            print(f"[MID {cam_name}] canvas dets={dets5.shape[0]}, tracks={tracks.shape[0]}")
        return remapped

    def step(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.frame_index += 1
        if is_keyframe(self.frame_index, KEYFRAME_INTERVAL):
            return self.process_keyframe(frames)
        else:
            return self.process_intermediate(frames)

# =============================
# Minimal dry-run (black frames)
# =============================
if __name__ == '__main__':
    cam_names = ['front', 'left', 'right']
    demo = ROICanvasDemo(cam_names, img_size=(640, 480))

    frames = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in cam_names}

    demo.step(frames)  # keyframe warmup
    for _ in range(10):
        demo.step(frames)
        if SHOW_WINDOWS:
            for n, f in frames.items():
                cv2.imshow(n, f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
