"""
KITTI Tracking Demo: YOLOv8n + ByteTrack + Canvas Patch Packing (Guillotine)

功能概述
- 基线（Baseline）：每帧全图检测（YOLOv8n），不做跟踪
- 系统（System）：每 p 帧做一次全图检测；中间帧将前一帧检测框扩展 padding=16px 后裁切为 patch，
  用 Guillotine 二维装箱放入固定大小 canvas（默认 640x640）一起送入 GPU 做检测；结果映射回原图；
  每帧用 ultralytics 的 ByteTrack 更新轨迹（评测精度为检测 AP@0.5，跟踪用于稳定 ID，不用于精度计量）。

评测
- 精度：AP@0.5（类别无关 class-agnostic）
- 性能：平均单帧延迟(ms)、整体 FPS、总时长

输出
- CSV 表格：baseline 与 system 指标对比
- 图表：每帧延迟曲线、延迟分布直方图、每帧目标数量曲线（可用于观测 batch/canvas 效果）

注意
- 不依赖本项目其他文件（仅使用第三方库 ultralytics/torch/cv2/numpy/matplotlib 等）
- 参数集中在 CONFIG 字典，便于后续修改

数据集结构（仅 training 有 label）:
<root>/kitti_tracking/
  training/
    calib/  image_02/  label_02/
  testing/
    calib/  image_02/
"""

import os
import time
import math
import csv
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
	from ultralytics import YOLO
except Exception as e:
	YOLO = None
	print("Warning: ultralytics not available. Please install `pip install ultralytics`. Error:", e)

try:
	import supervision as sv
except Exception as e:
	sv = None
	print("Warning: supervision not available. Please install `pip install supervision`. Error:", e)


# ==============================
# 可调参数（集中管理）
# ==============================
CONFIG = {
	# 路径与数据
	"kitti_root": os.path.expanduser("~/kitti_tracking"),
	"split": "training",  # 'training' or 'testing'
	# 指定序列列表；None 表示使用全部训练（或测试）序列
	"sequences": None,  # 例: [0, 1, 2]

	# 模型与推理
	"device": "cuda:0" if torch.cuda.is_available() else "cpu",
	"model_weights": "yolov8n.pt",
	"img_size": 640,        # YOLO 输入尺寸（方形）
	"conf_thres": 0.25,
	"iou_nms": 0.7,

	# 系统策略
	"full_det_interval": 5,  # 每 p 帧做一次全图检测
	"canvas_size": (640, 640),
	"patch_padding": 16,     # patch 四周扩展像素
	"allow_rotate": False,   # canvas 装箱是否允许旋转（推荐 False）

	# ByteTrack 参数（简化）
	"bytetrack": {
		"track_thresh": 0.5,
		"match_thresh": 0.8,
		"track_buffer": 30,
		"mot20": False,
		"frame_rate": 30,
	},

	# 评测
	"iou_eval": 0.5,         # AP@0.5 IoU

	# 输出
	"output_dir": os.path.join("legacy", "bench", "outputs", "kitti_tracking_demo"),
	"save_plots": True,
	"save_csv": True,

	# 调试
	"verbose": True,
}


# ==============================
# 数据结构
# ==============================
@dataclass
class Box:
	x1: float
	y1: float
	x2: float
	y2: float
	score: float = 1.0

	def as_xyxy(self) -> np.ndarray:
		return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

	def clip_(self, w: int, h: int) -> None:
		self.x1 = float(max(0, min(self.x1, w - 1)))
		self.y1 = float(max(0, min(self.y1, h - 1)))
		self.x2 = float(max(0, min(self.x2, w - 1)))
		self.y2 = float(max(0, min(self.y2, h - 1)))

	def area(self) -> float:
		return max(0.0, self.x2 - self.x1 + 1) * max(0.0, self.y2 - self.y1 + 1)


# ==============================
# 工具函数
# ==============================
def ensure_dir(path: str):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	iw, ih = max(0.0, ix2 - ix1 + 1), max(0.0, iy2 - iy1 + 1)
	inter = iw * ih
	aa = max(0.0, ax2 - ax1 + 1) * max(0.0, ay2 - ay1 + 1)
	ba = max(0.0, bx2 - bx1 + 1) * max(0.0, by2 - by1 + 1)
	union = aa + ba - inter + 1e-9
	return float(inter / union)


def load_kitti_sequences(root: str, split: str, sequences: Optional[List[int]] = None) -> List[Dict[str, Any]]:
	assert split in ("training", "testing")
	img_root = os.path.join(root, split, "image_02")
	if not os.path.isdir(img_root):
		raise FileNotFoundError(f"Not found: {img_root}")
	seq_ids = sorted([int(s) for s in os.listdir(img_root) if s.isdigit()])
	if sequences is not None:
		seq_ids = [s for s in seq_ids if s in sequences]

	seqs = []
	for sid in seq_ids:
		seq_dir = os.path.join(img_root, f"{sid:04d}")
		images = sorted([f for f in os.listdir(seq_dir) if f.endswith(".png")])
		img_paths = [os.path.join(seq_dir, f) for f in images]
		labels = None
		if split == "training":
			label_file = os.path.join(root, split, "label_02", f"{sid:04d}.txt")
			labels = parse_kitti_labels(label_file)
		seqs.append({
			"seq_id": sid,
			"img_paths": img_paths,
			"labels": labels,  # dict[int frame] -> List[Box]
		})
	return seqs


def parse_kitti_labels(label_file: str) -> Dict[int, List[Box]]:
	"""
	KITTI tracking label_02 每行：
	frame, track_id, type, truncated, occluded, alpha,
	left, top, right, bottom, h, w, l, x, y, z, ry, score?
	此处仅使用 frame 与 bbox；类别忽略（class-agnostic 评测）。
	"""
	if not os.path.isfile(label_file):
		raise FileNotFoundError(f"Missing label file: {label_file}")
	out: Dict[int, List[Box]] = {}
	with open(label_file, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) < 10:
				continue
			frame_id = int(parts[0])
			left, top, right, bottom = map(float, parts[6:10])
			b = Box(left, top, right, bottom, 1.0)
			out.setdefault(frame_id, []).append(b)
	return out


# ==============================
# 简化 AP@0.5 评测（Class-Agnostic）
# ==============================
class DetectionEvaluator:
	def __init__(self, iou_thresh: float = 0.5):
		self.iou_thresh = iou_thresh
		self.preds: List[Tuple[str, np.ndarray, float]] = []  # (frame_key, xyxy, score)
		self.gts: Dict[str, List[np.ndarray]] = {}            # frame_key -> list(xyxy)
		self.num_gts_total = 0

	def add_frame(self, frame_key: str, preds: List[Box], gts: List[Box]) -> None:
		for p in preds:
			self.preds.append((frame_key, p.as_xyxy().astype(np.float32), float(p.score)))
		xyxys = [g.as_xyxy().astype(np.float32) for g in gts]
		self.gts.setdefault(frame_key, [])
		self.gts[frame_key].extend(xyxys)
		self.num_gts_total += len(xyxys)

	def compute_ap(self) -> Dict[str, float]:
		if self.num_gts_total == 0:
			return {"AP50": 0.0, "precision": 0.0, "recall": 0.0}
		# 按置信度降序
		self.preds.sort(key=lambda x: x[2], reverse=True)
		gt_flags: Dict[str, List[bool]] = {k: [False] * len(v) for k, v in self.gts.items()}
		tps, fps = [], []
		for frame_key, pxyxy, score in self.preds:
			candidates = self.gts.get(frame_key, [])
			best_iou, best_j = 0.0, -1
			for j, gxyxy in enumerate(candidates):
				if gt_flags[frame_key][j]:
					continue
				iou = iou_xyxy(pxyxy, gxyxy)
				if iou > best_iou:
					best_iou, best_j = iou, j
			if best_iou >= self.iou_thresh and best_j >= 0:
				gt_flags[frame_key][best_j] = True
				tps.append(1)
				fps.append(0)
			else:
				tps.append(0)
				fps.append(1)

		tps = np.array(tps, dtype=np.float32)
		fps = np.array(fps, dtype=np.float32)
		if len(tps) == 0:
			return {"AP50": 0.0, "precision": 0.0, "recall": 0.0}

		cum_tp = np.cumsum(tps)
		cum_fp = np.cumsum(fps)
		recall = cum_tp / max(1, self.num_gts_total)
		precision = cum_tp / np.maximum(1, cum_tp + cum_fp)

		# AP（插值法，逐步向后取最大精度包络）
		mrec = np.concatenate(([0.0], recall, [1.0]))
		mpre = np.concatenate(([0.0], precision, [0.0]))
		for i in range(mpre.size - 1, 0, -1):
			mpre[i - 1] = max(mpre[i - 1], mpre[i])
		# 计算面积
		i = np.where(mrec[1:] != mrec[:-1])[0]
		ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

		return {
			"AP50": ap,
			"precision": float(precision[-1]),
			"recall": float(recall[-1]),
		}


# ==============================
# Guillotine 二维装箱（简化实现）
# ==============================
class GuillotinePacker:
	def __init__(self, width: int, height: int, allow_rotate: bool = False):
		self.W = int(width)
		self.H = int(height)
		self.allow_rotate = bool(allow_rotate)
		self.free_rects: List[Tuple[int, int, int, int]] = [(0, 0, self.W, self.H)]  # x,y,w,h

	def insert(self, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
		"""
		返回安置矩形 (x, y, w, h)，若无空间返回 None
		简化策略：首次适配 + 短边切割
		"""
		candidates = []
		for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
			# 不旋转
			if w <= fw and h <= fh:
				candidates.append((i, fx, fy, fw, fh, False))
			# 允许旋转
			if self.allow_rotate and h <= fw and w <= fh:
				candidates.append((i, fx, fy, fw, fh, True))
		if not candidates:
			return None
		# 选择最小剩余面积/最小碎片（简单启发）
		best = min(candidates, key=lambda c: (c[3] * c[4] - (h * w)))
		i, fx, fy, fw, fh, rotated = best
		rw, rh = (h, w) if rotated else (w, h)
		# 放置在 (fx, fy)
		placed = (fx, fy, rw, rh)
		# 切割 free 矩形
		self._split_free_rect(i, fx, fy, fw, fh, rw, rh)
		return placed

	def _split_free_rect(self, idx: int, fx: int, fy: int, fw: int, fh: int, rw: int, rh: int) -> None:
		# 使用水平/垂直切割生成两个新空闲矩形，简单短边切割策略
		del self.free_rects[idx]
		right = (fx + rw, fy, fw - rw, rh)     # 右侧条
		bottom = (fx, fy + rh, fw, fh - rh)    # 下侧块
		candidates = []
		if right[2] > 0 and right[3] > 0:
			candidates.append(right)
		if bottom[2] > 0 and bottom[3] > 0:
			candidates.append(bottom)
		# 合入空闲列表
		self.free_rects.extend(candidates)
		# 简单冗余清理（去除被包含矩形）
		self._prune_free_list()

	def _prune_free_list(self):
		pruned = []
		for i, r in enumerate(self.free_rects):
			x, y, w, h = r
			contained = False
			for j, s in enumerate(self.free_rects):
				if i == j:
					continue
				x2, y2, w2, h2 = s
				if x >= x2 and y >= y2 and (x + w) <= (x2 + w2) and (y + h) <= (y2 + h2):
					contained = True
					break
			if not contained:
				pruned.append(r)
		self.free_rects = pruned


# ==============================
# Canvas 构建 & 坐标还原
# ==============================
def build_canvas_and_placements(image: np.ndarray, boxes: List[Box], pad: int, canvas_size: Tuple[int, int],
                               allow_rotate: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
	H, W = image.shape[:2]
	ch = 3
	cw, chh = int(canvas_size[0]), int(canvas_size[1])
	canvas = np.zeros((chh, cw, 3), dtype=image.dtype)

	# 预生成 patch（加 padding 并裁剪）
	patches = []
	for b in boxes:
		x1 = max(0, int(math.floor(b.x1)) - pad)
		y1 = max(0, int(math.floor(b.y1)) - pad)
		x2 = min(W - 1, int(math.ceil(b.x2)) + pad)
		y2 = min(H - 1, int(math.ceil(b.y2)) + pad)
		pw, ph = x2 - x1 + 1, y2 - y1 + 1
		if pw <= 2 or ph <= 2:
			continue
		patch = image[y1:y2 + 1, x1:x2 + 1].copy()
		patches.append({
			"orig": (x1, y1, pw, ph),
			"img": patch,
		})

	# 大块优先装箱
	patches.sort(key=lambda p: p["orig"][2] * p["orig"][3], reverse=True)
	packer = GuillotinePacker(cw, chh, allow_rotate=allow_rotate)
	placements = []
	for p in patches:
		pw, ph = p["orig"][2], p["orig"][3]
		# 若 patch 超 canvas，先整体等比缩放以尝试适配
		scale_prefit = min(1.0, cw / max(1, pw), chh / max(1, ph))
		tw, th = max(1, int(pw * scale_prefit)), max(1, int(ph * scale_prefit))
		pos = packer.insert(tw, th)
		if pos is None:
			continue
		px, py, aw, ah = pos
		# 实际再缩放贴入
		if (aw, ah) != (tw, th):
			# 理论上不应发生；保守处理
			tw, th = aw, ah
		pimg = cv2.resize(p["img"], (tw, th), interpolation=cv2.INTER_LINEAR)
		canvas[py:py + th, px:px + tw] = pimg
		placements.append({
			"canvas_xywh": (px, py, tw, th),
			"orig_xywh": p["orig"],  # 在原图中的 x1,y1,w,h
			"scale": (tw / p["orig"][2], th / p["orig"][3]),
		})

	return canvas, placements


def remap_dets_from_canvas(dets_xyxy: np.ndarray, placements: List[Dict[str, Any]]) -> List[Box]:
	"""
	将 canvas 中的检测框映射回原图坐标。
	通过检测框中心点找到其所属的 patch 放置区域，然后逆缩放 + 平移还原。
	"""
	out: List[Box] = []
	for x1, y1, x2, y2, score in dets_xyxy:
		cx = 0.5 * (x1 + x2)
		cy = 0.5 * (y1 + y2)
		for pl in placements:
			px, py, pw, ph = pl["canvas_xywh"]
			if (cx >= px) and (cx <= px + pw) and (cy >= py) and (cy <= py + ph):
				sx, sy = pl["scale"]
				ox, oy, ow, oh = pl["orig_xywh"]
				# 相对 patch 内坐标
				lx1, ly1 = (x1 - px) / max(1e-9, sx), (y1 - py) / max(1e-9, sy)
				lx2, ly2 = (x2 - px) / max(1e-9, sx), (y2 - py) / max(1e-9, sy)
				# 映射回原图
				rx1, ry1 = ox + lx1, oy + ly1
				rx2, ry2 = ox + lx2, oy + ly2
				out.append(Box(rx1, ry1, rx2, ry2, float(score)))
				break
	return out


# ==============================
# 模型与跟踪
# ==============================
def load_yolo(model_weights: str, device: str, imgsz: int, conf_thres: float, iou_nms: float):
	if YOLO is None:
		raise RuntimeError("ultralytics not installed.")
	model = YOLO(model_weights)
	model.fuse()
	# 记录配置，便于后续查看
	model.overrides["conf"] = conf_thres
	model.overrides["iou"] = iou_nms
	model.overrides["imgsz"] = imgsz
	model.to(device)
	return model


def run_detect(model, image_bgr: np.ndarray) -> List[Box]:
	"""
	对单张图像做检测，返回 Box 列表（坐标基于输入图像尺寸）。
	"""
	results = model.predict(source=image_bgr, verbose=False)
	boxes: List[Box] = []
	if len(results) == 0:
		return boxes
	res = results[0]
	if res and res.boxes is not None and res.boxes.xyxy is not None:
		xyxy = res.boxes.xyxy.detach().cpu().numpy()
		scores = res.boxes.conf.detach().cpu().numpy() if res.boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
		for (x1, y1, x2, y2), sc in zip(xyxy, scores):
			boxes.append(Box(float(x1), float(y1), float(x2), float(y2), float(sc)))
	return boxes


def create_bytetracker(cfg: Dict[str, Any]):
	if sv is None:
		raise RuntimeError("supervision not installed. Please `pip install supervision`.")
	return sv.ByteTrack(
		track_activation_threshold=float(cfg.get("track_thresh", 0.5)),
		lost_track_buffer=int(cfg.get("track_buffer", 30)),
		minimum_matching_threshold=float(cfg.get("match_thresh", 0.8)),
		frame_rate=int(cfg.get("frame_rate", 30)),
		minimum_consecutive_frames=int(cfg.get("min_consecutive", 1)),
	)


def bytetrack_update(tracker, dets: List[Box], img_wh: Tuple[int, int]):
	"""
	使用 supervision 的 ByteTrack：将 Box 列表转为 sv.Detections 并调用 update_with_detections。
	"""
	if sv is None:
		return []

	if len(dets) == 0:
		xyxy = np.zeros((0, 4), dtype=np.float32)
		conf = np.zeros((0,), dtype=np.float32)
	else:
		xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
		conf = np.array([d.score for d in dets], dtype=np.float32)
	class_id = np.zeros((xyxy.shape[0],), dtype=int) if xyxy.shape[0] > 0 else np.zeros((0,), dtype=int)
	detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

	try:
		return tracker.update_with_detections(detections)
	except Exception:
		# 兼容旧版本 API（极少数版本用 update）
		return tracker.update(detections)


# ==============================
# 主流程（单种模式运行）
# ==============================
def run_pipeline(mode: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
	"""
	mode: 'baseline' or 'system'
	"""
	assert mode in ("baseline", "system")
	device = cfg["device"]
	imgsz = cfg["img_size"]
	canvas_size = cfg["canvas_size"]
	pad = int(cfg["patch_padding"])
	full_det_interval = int(cfg["full_det_interval"])

	model = load_yolo(cfg["model_weights"], device, imgsz, cfg["conf_thres"], cfg["iou_nms"])
	tracker = create_bytetracker(cfg["bytetrack"]) if mode == "system" else None

	seqs = load_kitti_sequences(cfg["kitti_root"], cfg["split"], cfg["sequences"])

	# 评测器
	evaluator = DetectionEvaluator(iou_thresh=cfg["iou_eval"])

	# 统计
	per_frame_latency_ms: List[float] = []
	per_frame_obj_count: List[int] = []
	total_frames = 0

	t0_total = time.time()

	for seq in seqs:
		seq_id = seq["seq_id"]
		img_paths = seq["img_paths"]
		labels = seq.get("labels", None)  # None for testing split

		prev_dets: List[Box] = []

		for frame_idx, img_path in enumerate(img_paths):
			img = cv2.imread(img_path)
			if img is None:
				continue
			H, W = img.shape[:2]

			start = time.time()

			# 检测
			if mode == "baseline" or (frame_idx % full_det_interval == 0) or (len(prev_dets) == 0):
				# 全图检测
				curr_dets = run_detect(model, img)
			else:
				# canvas patch 检测
				canvas, placements = build_canvas_and_placements(
					img, prev_dets, pad=pad, canvas_size=canvas_size, allow_rotate=cfg["allow_rotate"]
				)
				canvas_dets = run_detect(model, canvas)
				# 映射回原图
				if len(canvas_dets) > 0:
					dets_np = np.array([[d.x1, d.y1, d.x2, d.y2, d.score] for d in canvas_dets], dtype=np.float32)
				else:
					dets_np = np.zeros((0, 5), dtype=np.float32)
				mapped = remap_dets_from_canvas(dets_np, placements)
				curr_dets = mapped

			# 跟踪（system 模式）
			if tracker is not None:
				try:
					bytetrack_update(tracker, curr_dets, (W, H))
				except Exception as e:
					if cfg["verbose"]:
						print(f"[WARN] ByteTrack update failed at seq {seq_id} frame {frame_idx}: {e}")

			# 裁剪/清理坐标
			for b in curr_dets:
				b.clip_(W, H)

			per_frame_obj_count.append(len(curr_dets))

			# 评测（仅 training split 有标签）
			gts = []
			if labels is not None:
				gts = labels.get(frame_idx, [])
			evaluator.add_frame(f"{seq_id}_{frame_idx}", curr_dets, gts)

			# 为下一帧准备
			prev_dets = curr_dets

			elapsed_ms = (time.time() - start) * 1000.0
			per_frame_latency_ms.append(elapsed_ms)
			total_frames += 1

	t1_total = time.time()
	total_time_s = t1_total - t0_total

	metrics = evaluator.compute_ap()
	avg_latency = float(np.mean(per_frame_latency_ms)) if len(per_frame_latency_ms) else 0.0
	fps = float(total_frames / total_time_s) if total_time_s > 0 else 0.0

	return {
		"mode": mode,
		"AP50": metrics.get("AP50", 0.0),
		"precision": metrics.get("precision", 0.0),
		"recall": metrics.get("recall", 0.0),
		"avg_latency_ms": avg_latency,
		"fps": fps,
		"total_frames": total_frames,
		"total_time_s": total_time_s,
		"per_frame_latency_ms": per_frame_latency_ms,
		"per_frame_obj_count": per_frame_obj_count,
	}


# ==============================
# 可视化与保存
# ==============================
def save_results(cfg: Dict[str, Any], baseline: Dict[str, Any], system: Dict[str, Any]) -> None:
	out_dir = cfg["output_dir"]
	ensure_dir(out_dir)

	# CSV
	if cfg["save_csv"]:
		csv_path = os.path.join(out_dir, "summary.csv")
		with open(csv_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["mode", "AP50", "precision", "recall", "avg_latency_ms", "fps", "total_frames", "total_time_s"])
			for res in [baseline, system]:
				writer.writerow([
					res["mode"], f"{res['AP50']:.4f}", f"{res['precision']:.4f}", f"{res['recall']:.4f}",
					f"{res['avg_latency_ms']:.2f}", f"{res['fps']:.2f}",
					res["total_frames"], f"{res['total_time_s']:.2f}"
				])

	# 图表
	if cfg["save_plots"]:
		# 1) 每帧延迟曲线
		plt.figure(figsize=(12, 4))
		plt.plot(baseline["per_frame_latency_ms"], label="baseline latency (ms)")
		plt.plot(system["per_frame_latency_ms"], label="system latency (ms)")
		plt.xlabel("frame idx")
		plt.ylabel("latency (ms)")
		plt.title("Per-frame latency (baseline vs system)")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, "latency_curve.png"))
		plt.close()

		# 2) 延迟直方图
		plt.figure(figsize=(10, 4))
		plt.hist(baseline["per_frame_latency_ms"], bins=40, alpha=0.6, label="baseline")
		plt.hist(system["per_frame_latency_ms"], bins=40, alpha=0.6, label="system")
		plt.xlabel("latency (ms)")
		plt.ylabel("count")
		plt.title("Latency histogram")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, "latency_hist.png"))
		plt.close()

		# 3) 每帧目标数量（检测框数）
		plt.figure(figsize=(12, 4))
		plt.plot(baseline["per_frame_obj_count"], label="baseline det count")
		plt.plot(system["per_frame_obj_count"], label="system det count")
		plt.xlabel("frame idx")
		plt.ylabel("#detections")
		plt.title("Per-frame detection count")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, "det_count_curve.png"))
		plt.close()

	# 控制台对比表
	def fmt(res: Dict[str, Any]) -> str:
		return (f"{res['mode']:8s} | AP50={res['AP50']:.4f} | P={res['precision']:.4f} | R={res['recall']:.4f} | "
		        f"AvgLatency={res['avg_latency_ms']:.2f}ms | FPS={res['fps']:.2f} | Frames={res['total_frames']} | "
		        f"Time={res['total_time_s']:.2f}s")
	print(fmt(baseline))
	print(fmt(system))


# ==============================
# 入口
# ==============================
def main():
	cfg = CONFIG.copy()
	ensure_dir(cfg["output_dir"])
	if cfg["verbose"]:
		print(json.dumps({
			"use_device": cfg["device"],
			"kitti_root": cfg["kitti_root"],
			"split": cfg["split"],
			"sequences": cfg["sequences"],
			"full_det_interval": cfg["full_det_interval"],
			"canvas_size": cfg["canvas_size"],
		}, indent=2))

	# 运行基线
	baseline = run_pipeline("baseline", cfg)
	# 运行系统
	system = run_pipeline("system", cfg)

	save_results(cfg, baseline, system)


if __name__ == "__main__":
	main()


