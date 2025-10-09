import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Deque
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# Optional: import your SORT implementation
from sort import Sort  # expects API: update(dets) -> [[x1,y1,x2,y2,track_id], ...]

# BEV和多摄像头融合
from bev_transform import BEVTransformer, MultiCameraFusion, create_default_camera_params

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

# Batch processing configurations
ROI_BATCH_SIZE = 320            # target size for ROI resizing (320x320)
USE_BATCH_PROCESSING = False    # temporarily disable batch ROI processing due to OpenCV issues
MAX_BATCH_SIZE = 16             # maximum number of ROIs in one batch

# Multi-camera fusion configurations
USE_BEV_FUSION = True           # enable BEV-based multi-camera fusion
NUM_CAMERAS = 4                 # number of cameras (front, left, right, rear)
BEV_DISTANCE_THRESHOLD = 1.0    # ground distance threshold for fusion (meters)
BEV_TIME_THRESHOLD = 0.3        # time difference threshold for fusion (seconds)

# Debug configurations
DEBUG_MODE = True               # enable debug output
FALLBACK_TO_CANVAS = True       # fallback to canvas processing if batch fails

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


def resize_roi_to_batch_size(roi: np.ndarray, target_size: int = ROI_BATCH_SIZE) -> np.ndarray:
    """
    将ROI调整到指定大小，保持长宽比并进行letterbox填充
    """
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    h, w = roi.shape[:2]
    
    # 确保尺寸有效
    if w <= 0 or h <= 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 计算缩放比例，保持长宽比
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # 调整大小
    try:
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"[WARNING] ROI调整大小失败: {e}")
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 创建目标大小的画布并居中放置
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # 计算放置位置（居中）
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # 确保索引在有效范围内
    x_offset = max(0, min(x_offset, target_size - new_w))
    y_offset = max(0, min(y_offset, target_size - new_h))
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def extract_rois_from_frames(frames: Dict[str, np.ndarray], roi_coords: List[Tuple[str, int, int, int, int]]) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
    """
    从帧中提取ROI图像并调整到批量处理大小
    roi_coords: list of (cam_name, x, y, w, h)
    """
    roi_items = []
    
    for cam_name, x, y, w, h in roi_coords:
        if cam_name not in frames:
            continue
            
        frame = frames[cam_name]
        H, W = frame.shape[:2]
        
        # 确保坐标在有效范围内
        x, y, w, h = clamp_rect(x, y, w, h, W, H)
        
        # 检查ROI尺寸是否有效
        if w <= 0 or h <= 0:
            print(f"[WARNING] 无效的ROI尺寸: {cam_name} ({x}, {y}, {w}, {h})")
            continue
        
        # 提取ROI
        roi = frame[y:y + h, x:x + w]
        
        # 验证ROI是否有效
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"[WARNING] 提取的ROI为空: {cam_name} ({x}, {y}, {w}, {h})")
            continue
        
        # 检查ROI是否包含有效数据
        if np.all(roi == 0) or np.all(roi == 114):  # 全黑或全灰
            print(f"[WARNING] ROI包含无效数据: {cam_name}")
            continue
        
        try:
            # 调整到批量处理大小
            resized_roi = resize_roi_to_batch_size(roi, ROI_BATCH_SIZE)
            
            # 验证调整后的ROI
            if resized_roi.size > 0 and resized_roi.shape[0] > 0 and resized_roi.shape[1] > 0:
                roi_items.append((cam_name, resized_roi, (x, y, w, h)))
            else:
                print(f"[WARNING] 调整后的ROI无效: {cam_name}")
                
        except Exception as e:
            print(f"[WARNING] ROI调整失败: {cam_name}, 错误: {e}")
            continue
    
    return roi_items


def batch_inference_rois(model: YOLO, roi_batch: List[np.ndarray]) -> List[np.ndarray]:
    """
    对ROI批次进行批量推理
    """
    if not roi_batch:
        return []
    
    # 严格过滤ROI
    valid_rois = []
    valid_indices = []
    
    for i, roi in enumerate(roi_batch):
        # 检查ROI的基本属性
        if roi is None:
            print(f"[WARNING] ROI {i} 为 None")
            continue
            
        if not isinstance(roi, np.ndarray):
            print(f"[WARNING] ROI {i} 不是 numpy 数组")
            continue
            
        if roi.size == 0:
            print(f"[WARNING] ROI {i} 大小为 0")
            continue
            
        if len(roi.shape) != 3:
            print(f"[WARNING] ROI {i} 维度不正确: {roi.shape}")
            continue
            
        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            print(f"[WARNING] ROI {i} 尺寸无效: {roi.shape}")
            continue
            
        if roi.shape[2] != 3:
            print(f"[WARNING] ROI {i} 通道数不正确: {roi.shape}")
            continue
        
        # 检查数据类型
        if roi.dtype != np.uint8:
            print(f"[WARNING] ROI {i} 数据类型不正确: {roi.dtype}")
            continue
        
        # 检查是否有有效的像素值
        if np.all(roi == 0) or np.all(roi == 114):
            print(f"[WARNING] ROI {i} 包含无效像素值")
            continue
        
        valid_rois.append(roi)
        valid_indices.append(i)
    
    print(f"[BATCH] 有效ROI数量: {len(valid_rois)}/{len(roi_batch)}")
    
    if not valid_rois:
        return [np.empty((0, 6), dtype=np.float32) for _ in roi_batch]
    
    try:
        # 将有效ROI列表转换为numpy数组批次
        batch_array = np.stack(valid_rois, axis=0)
        print(f"[BATCH] 批次数组形状: {batch_array.shape}")
        
        # 进行批量推理
        results = model.predict(batch_array, verbose=False)
        
    except Exception as e:
        print(f"[ERROR] 批量推理失败: {e}")
        return [np.empty((0, 6), dtype=np.float32) for _ in roi_batch]
    
    # 提取检测结果
    batch_detections = []
    result_idx = 0
    
    for i, roi in enumerate(roi_batch):
        if i not in valid_indices:
            # 对于无效的ROI，返回空检测结果
            batch_detections.append(np.empty((0, 6), dtype=np.float32))
        else:
            # 处理有效的ROI结果
            result = results[result_idx]
            result_idx += 1
            
            if result.boxes.shape[0] == 0:
                batch_detections.append(np.empty((0, 6), dtype=np.float32))
            else:
                xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = result.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
                cls = result.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
                
                # 过滤允许的类别
                cls_i = cls.astype(np.int32).flatten()
                keep = np.array([c in ALLOWED_CLS for c in cls_i], dtype=bool)
                
                if not keep.any():
                    batch_detections.append(np.empty((0, 6), dtype=np.float32))
                else:
                    detections = np.hstack((xyxy[keep], conf[keep], cls[keep]))
                    batch_detections.append(detections)
    
    return batch_detections


def remap_batch_detections_to_original(
    batch_detections: List[np.ndarray], 
    roi_items: List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]
) -> Dict[str, np.ndarray]:
    """
    将批量推理结果映射回原始帧坐标
    """
    per_cam: Dict[str, List[List[float]]] = defaultdict(list)
    
    for i, (detections, (cam_name, resized_roi, (orig_x, orig_y, orig_w, orig_h))) in enumerate(zip(batch_detections, roi_items)):
        if detections.size == 0:
            continue
            
        # 获取调整大小前的原始ROI尺寸
        orig_roi = roi_items[i][1]  # 原始ROI图像
        orig_h, orig_w = orig_roi.shape[:2]
        
        # 计算缩放比例
        scale_x = orig_w / ROI_BATCH_SIZE
        scale_y = orig_h / ROI_BATCH_SIZE
        
        # 计算letterbox偏移（假设居中对齐）
        pad_x = (ROI_BATCH_SIZE - orig_w * ROI_BATCH_SIZE / orig_w) / 2 if orig_w < orig_h else 0
        pad_y = (ROI_BATCH_SIZE - orig_h * ROI_BATCH_SIZE / orig_h) / 2 if orig_h < orig_w else 0
        
        for detection in detections:
            # 安全地解包检测结果，支持5个或6个值
            if len(detection) == 6:
                x1, y1, x2, y2, conf, cls_id = detection
            elif len(detection) == 5:
                x1, y1, x2, y2, conf = detection
                cls_id = 0  # 默认类别
            else:
                print(f"[WARNING] 意外的检测结果格式，长度: {len(detection)}")
                continue
            
            # 移除letterbox填充
            x1_unpad = (x1 - pad_x) * scale_x
            y1_unpad = (y1 - pad_y) * scale_y
            x2_unpad = (x2 - pad_x) * scale_x
            y2_unpad = (y2 - pad_y) * scale_y
            
            # 映射到原始帧坐标
            orig_x1 = orig_x + x1_unpad
            orig_y1 = orig_y + y1_unpad
            orig_x2 = orig_x + x2_unpad
            orig_y2 = orig_y + y2_unpad
            
            per_cam[cam_name].append([orig_x1, orig_y1, orig_x2, orig_y2, float(conf), float(cls_id)])
    
    # 转换为numpy数组
    out: Dict[str, np.ndarray] = {}
    for cam, lst in per_cam.items():
        out[cam] = np.array(lst, dtype=np.float32) if lst else np.empty((0, 6), dtype=np.float32)
    
    return out

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
        
        # BEV和多摄像头融合
        self.use_bev_fusion = USE_BEV_FUSION
        if self.use_bev_fusion:
            self._init_bev_fusion(cam_names)
        
        # 时间戳记录
        self.current_timestamp = time.time()

    def _init_bev_fusion(self, cam_names: List[str]):
        """初始化BEV融合组件"""
        print(f"[BEV] 初始化多摄像头BEV融合，摄像头数量: {len(cam_names)}")
        
        # 创建相机参数（使用默认配置，实际应用中应该从标定文件加载）
        camera_params = create_default_camera_params()
        
        # 只保留当前使用的摄像头参数
        available_cameras = {name: camera_params[name] for name in cam_names 
                           if name in camera_params}
        
        if len(available_cameras) != len(cam_names):
            print(f"[BEV] 警告：部分摄像头参数缺失，可用: {list(available_cameras.keys())}")
        
        # 初始化BEV变换器和融合器
        self.bev_transformer = BEVTransformer(available_cameras, ground_height=0.0)
        self.multi_camera_fusion = MultiCameraFusion(
            self.bev_transformer,
            distance_threshold=BEV_DISTANCE_THRESHOLD,
            time_threshold=BEV_TIME_THRESHOLD
        )
        
        print(f"[BEV] BEV融合初始化完成")

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

    def process_batch_rois(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        批量处理ROI：提取ROI -> 调整大小 -> 批量推理 -> 映射回原始坐标
        """
        # 获取预测的ROI坐标
        roi_coords = []
        for cam_name, frame in frames.items():
            tracks = self.prev_tracks.get(cam_name, np.empty((0, 5), dtype=np.float32))
            for trk in tracks:
                x, y, w, h = self._predict_next_box(cam_name, trk)
                roi_coords.append((cam_name, x, y, w, h))
        
        if not roi_coords:
            return {name: np.empty((0, 6), dtype=np.float32) for name in frames.keys()}
        
        # 限制ROI数量
        roi_coords = roi_coords[:MAX_BATCH_SIZE]
        
        # 提取并调整ROI大小
        roi_items = extract_rois_from_frames(frames, roi_coords)
        
        if not roi_items:
            return {name: np.empty((0, 6), dtype=np.float32) for name in frames.keys()}
        
        # 可视化批量ROI
        if SHOW_WINDOWS:
            batch_vis = self._visualize_batch_rois(roi_items)
            cv2.imshow('Batch ROIs', batch_vis)
        
        # 批量推理
        roi_batch = [item[1] for item in roi_items]  # 提取调整后的ROI图像
        batch_detections = batch_inference_rois(self.model, roi_batch)
        
        # 映射回原始坐标
        remapped_dets = remap_batch_detections_to_original(batch_detections, roi_items)
        
        # 更新跟踪器
        for cam_name in frames.keys():
            dets5 = remapped_dets.get(cam_name, np.empty((0, 6), dtype=np.float32))
            dets5 = dets5[:, :5] if dets5.size > 0 else dets5  # SORT expects [x1,y1,x2,y2,score]
            tracks = self.update_tracker(cam_name, dets5)
            print(f"[BATCH {cam_name}] batch dets={dets5.shape[0]}, tracks={tracks.shape[0]}")
        
        return remapped_dets

    def _visualize_batch_rois(self, roi_items: List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        可视化批量ROI
        """
        if not roi_items:
            return np.zeros((ROI_BATCH_SIZE, ROI_BATCH_SIZE, 3), dtype=np.uint8)
        
        # 创建网格布局
        n = len(roi_items)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        vis_w = cols * ROI_BATCH_SIZE
        vis_h = rows * ROI_BATCH_SIZE
        vis_canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        
        for i, (cam_name, roi_img, _) in enumerate(roi_items):
            row = i // cols
            col = i % cols
            
            y_start = row * ROI_BATCH_SIZE
            x_start = col * ROI_BATCH_SIZE
            
            vis_canvas[y_start:y_start + ROI_BATCH_SIZE, x_start:x_start + ROI_BATCH_SIZE] = roi_img
            
            # 添加标签
            cv2.putText(vis_canvas, f"{i}:{cam_name}", 
                       (x_start + 5, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_canvas

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
        # 选择处理方式：批量处理或画布处理
        if USE_BATCH_PROCESSING:
            try:
                return self.process_batch_rois(frames)
            except Exception as e:
                print(f"[ERROR] 批量处理失败: {e}")
                if FALLBACK_TO_CANVAS:
                    print("[FALLBACK] 回退到画布处理模式")
                    return self.process_intermediate_canvas(frames)
                else:
                    raise e
        else:
            return self.process_intermediate_canvas(frames)

    def process_intermediate_canvas(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        原始的画布处理方法（保持向后兼容）
        """
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
        self.current_timestamp = time.time()
        
        if is_keyframe(self.frame_index, KEYFRAME_INTERVAL):
            results = self.process_keyframe(frames)
        else:
            results = self.process_intermediate(frames)
        
        # BEV多摄像头融合
        if self.use_bev_fusion and len(frames) > 1:
            results = self.process_bev_fusion(results, frames)
        
        return results

    def process_bev_fusion(self, detections_by_camera: Dict[str, np.ndarray], 
                          frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        使用BEV变换进行多摄像头融合
        """
        try:
            # 使用多摄像头融合器
            fused_results = self.multi_camera_fusion.fuse_detections(
                detections_by_camera, 
                self.current_timestamp
            )
            
            # 统计融合效果
            total_detections_before = sum(dets.shape[0] for dets in detections_by_camera.values())
            total_detections_after = sum(dets.shape[0] for dets in fused_results.values())
            
            print(f"[BEV] 融合前总检测数: {total_detections_before}, 融合后: {total_detections_after}")
            
            # 可视化BEV融合结果
            if SHOW_WINDOWS:
                self._visualize_bev_fusion(fused_results, frames)
            
            return fused_results
            
        except Exception as e:
            print(f"[BEV] 融合失败，使用原始结果: {e}")
            return detections_by_camera

    def _visualize_bev_fusion(self, fused_results: Dict[str, np.ndarray], 
                             frames: Dict[str, np.ndarray]):
        """
        可视化BEV融合结果
        """
        # 创建BEV鸟瞰图可视化
        bev_size = 800
        bev_canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 绘制地面网格
        grid_size = 50
        for i in range(0, bev_size, grid_size):
            cv2.line(bev_canvas, (i, 0), (i, bev_size), (50, 50, 50), 1)
            cv2.line(bev_canvas, (0, i), (bev_size, i), (50, 50, 50), 1)
        
        # 绘制坐标轴
        center = bev_size // 2
        cv2.arrowedLine(bev_canvas, (center, center), (center + 100, center), (0, 255, 0), 2)  # X轴
        cv2.arrowedLine(bev_canvas, (center, center), (center, center - 100), (255, 0, 0), 2)  # Y轴
        cv2.putText(bev_canvas, "X", (center + 110, center + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bev_canvas, "Y", (center + 10, center - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 映射地面坐标到BEV图像
        scale = 20  # 像素/米
        
        # 为每个摄像头使用不同颜色
        colors = {
            'front': (255, 255, 0),   # 黄色
            'left': (255, 0, 255),    # 品红
            'right': (0, 255, 255),   # 青色
            'rear': (255, 128, 0)     # 橙色
        }
        
        for cam_name, detections in fused_results.items():
            if detections.size == 0:
                continue
            
            color = colors.get(cam_name, (255, 255, 255))
            
            # 获取地面坐标
            ground_points = self.bev_transformer.get_bottom_center_ground(cam_name, detections)
            
            for i, (detection, ground_point) in enumerate(zip(detections, ground_points)):
                # 转换到BEV图像坐标
                x, y = ground_point[0], ground_point[1]
                bev_x = int(center + x * scale)
                bev_y = int(center - y * scale)  # Y轴翻转
                
                # 确保在图像范围内
                if 0 <= bev_x < bev_size and 0 <= bev_y < bev_size:
                    # 绘制检测点
                    cv2.circle(bev_canvas, (bev_x, bev_y), 5, color, -1)
                    
                    # 绘制置信度
                    conf = detection[4]
                    cv2.putText(bev_canvas, f"{conf:.2f}", 
                               (bev_x + 8, bev_y - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示BEV图像
        cv2.imshow('BEV Fusion View', bev_canvas)
        
        # 在原始图像上绘制融合后的检测结果
        for cam_name, frame in frames.items():
            if cam_name not in fused_results:
                continue
                
            vis_frame = frame.copy()
            detections = fused_results[cam_name]
            
            for detection in detections:
                # 安全地解包检测结果
                if len(detection) == 6:
                    x1, y1, x2, y2, conf, cls_id = detection
                elif len(detection) == 5:
                    x1, y1, x2, y2, conf = detection
                    cls_id = 0
                else:
                    continue
                    
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_frame, f'BEV-{conf:.2f}', 
                           (int(x1), max(0, int(y1) - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(f'BEV-{cam_name}', vis_frame)

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
