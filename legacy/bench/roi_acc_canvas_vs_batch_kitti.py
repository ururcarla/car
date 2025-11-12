#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Accuracy Benchmark on KITTI-Tracking: Canvas vs Batch
--------------------------------------------------------
目标：
- 基于 KITTI-tracking 的标注(label_02)，用 GT 框作为 ROI 源，比较两种推理方式的精度：
  1) Canvas 集中 ROI 推理（固定画布+letterbox装入各ROI，一次推理）
  2) Batch ROI 推理（各ROI缩放到统一尺寸，组成小批次推理）
- 同时统计 “少ROI / 多ROI” 两种场景下的精度。
- 输出：CSV 与图表（Precision/Recall@IoU=0.5，F1），并保存可视化对比图。

数据集目录（KITTI tracking）:
<root>/
  calib/
  image_02/<seq>/*.png
  label_02/<seq>.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from ultralytics import YOLO as UL_YOLO
except Exception:
    UL_YOLO = None


# ---------------------------- 配置 ----------------------------
ALLOWED_KITTI_TYPES = {"Car", "Pedestrian", "Cyclist"}  # 评价时仅关注的GT类别
# 将COCO类别限制到公路相关目标，便于减少干扰（ultralytics默认是COCO）
ALLOWED_COCO_IDS = {0, 1, 2, 3, 5, 7}  # person,bicycle,car,motorcycle,bus,truck

ROI_BATCH_SIZE = 320                 # ROI缩放后的正方形尺寸
CANVAS_W, CANVAS_H = 1024, 1024      # 画布大小
GROUP_SIZE = 4                       # 模拟多摄像头，每组图像数量
FEW_MANY_THRESHOLD = 20              # 少/多ROI阈值（按一组内GT框总数）
IOU_THRESH = 0.5                     # 精度评估IoU阈值

VIS_MAX_PER_MODEL = 20               # 每个模型最多保存的可视化样本数


# ---------------------------- 数据结构 ----------------------------
@dataclass
class ImageItem:
    path: Path
    image: np.ndarray
    gts: np.ndarray  # [N, 4] xyxy float32（仅评估IoU，类别忽略或可选）


@dataclass
class GroupBatch:
    images: List[ImageItem]  # 长度=group_size


@dataclass
class AccRecord:
    model: str
    scenario: str  # "few" | "many"
    method: str    # "canvas" | "batch"
    precision: float
    recall: float
    f1: float
    num_groups: int


# ---------------------------- 工具函数 ----------------------------
def ensure_env():
    if cv2 is None:
        raise RuntimeError("未安装opencv-python，请先安装。")
    if UL_YOLO is None:
        raise RuntimeError("未安装 ultralytics，请先 pip install ultralytics。")
    if torch is None or not torch.cuda.is_available():
        print("[WARN] 未检测到CUDA，将在CPU上运行，速度与结果可能有差异。")


def load_all_sequences(image_02_dir: Path) -> List[str]:
    """返回 image_02 下存在的序列号列表（目录名）。"""
    seqs = []
    for p in sorted(image_02_dir.iterdir()):
        if p.is_dir():
            seqs.append(p.name)
    return seqs


def parse_kitti_tracking_labels(label_file: Path) -> Dict[int, List[Tuple[float, float, float, float, str]]]:
    """
    解析 KITTI tracking 的 label_02/<seq>.txt
    返回：frame_idx -> [(x1,y1,x2,y2,type), ...]
    仅保留 ALLOWED_KITTI_TYPES。
    """
    by_frame: Dict[int, List[Tuple[float, float, float, float, str]]] = defaultdict(list)
    if not label_file.exists():
        return by_frame
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 格式：frame track_id type truncated occluded alpha bbox(l t r b) dims loc ry ...
            if len(parts) < 15:
                continue
            try:
                frame = int(parts[0])
                obj_type = parts[2]
                if obj_type not in ALLOWED_KITTI_TYPES:
                    continue
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
                # 过滤无效框
                if x2 <= x1 or y2 <= y1:
                    continue
                by_frame[frame].append((x1, y1, x2, y2, obj_type))
            except Exception:
                continue
    return by_frame


def load_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    return img


def build_dataset_items(root: Path, num_images: int, seed: int) -> List[ImageItem]:
    """
    从 KITTI-tracking 目录构建图像与GT：随机抽取若干帧（跨序列）。
    root 下应包含 image_02/ 与 label_02/
    """
    image_02 = root / "image_02"
    label_02 = root / "label_02"
    if not image_02.exists():
        raise FileNotFoundError(f"未找到目录: {image_02}")
    if not label_02.exists():
        raise FileNotFoundError(f"未找到目录: {label_02}")

    seqs = load_all_sequences(image_02)
    if len(seqs) == 0:
        raise RuntimeError("image_02 下未找到任何序列目录。")

    # 收集所有 (img_path, gt_boxes)
    rng = random.Random(seed)
    all_items: List[ImageItem] = []
    for seq in seqs:
        img_dir = image_02 / seq
        lab_file = label_02 / f"{seq}.txt"
        labels_by_frame = parse_kitti_tracking_labels(lab_file)
        # 枚举该序列的所有帧
        img_files = sorted([p for p in img_dir.glob("*.png")])
        for p in img_files:
            # 帧编号为文件名（去后缀）的整数
            try:
                frame_idx = int(p.stem)
            except Exception:
                continue
            gts_f = labels_by_frame.get(frame_idx, [])
            # 将 GT 转为 ndarray [N,4]，xyxy float32
            if len(gts_f) == 0:
                continue
            boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in gts_f], dtype=np.float32)
            img = load_image(p)
            if img is None:
                continue
            all_items.append(ImageItem(path=p, image=img, gts=boxes))

    if len(all_items) == 0:
        raise RuntimeError("未找到带有效GT的图像帧。")

    # 随机抽样
    rng.shuffle(all_items)
    return all_items[: min(num_images, len(all_items))]


def clamp_rect(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    if x + w > W:
        w = max(1, W - x)
    if y + h > H:
        h = max(1, H - y)
    return x, y, w, h


# ---------------------------- Canvas 打包（固定网格+letterbox） ----------------------------
@dataclass
class ROIMapping:
    cam_name: str
    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int
    tile_w: int
    tile_h: int
    scale: float
    pad_x: int
    pad_y: int


def _grid_for_n(n: int, W: int, H: int) -> Tuple[int, int, int, int]:
    n = max(1, n)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    tile_w = W // cols
    tile_h = H // rows
    return rows, cols, tile_w, tile_h


def pack_rois_fixed_canvas(
    roi_items: List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]],
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> Tuple[np.ndarray, List[ROIMapping]]:
    if not roi_items:
        return np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8), []
    n = len(roi_items)
    rows, cols, tile_w, tile_h = _grid_for_n(n, canvas_w, canvas_h)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 114, dtype=np.uint8)
    mappings: List[ROIMapping] = []
    for idx, (cam_name, frame, (x, y, w, h)) in enumerate(roi_items[:n]):
        H, W = frame.shape[:2]
        x, y, w, h = clamp_rect(x, y, w, h, W, H)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        s = min(tile_w / float(w), tile_h / float(h))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
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
        mappings.append(ROIMapping(
            cam_name=cam_name, x=x, y=y, w=w, h=h,
            cx=cx, cy=cy, tile_w=tile_w, tile_h=tile_h,
            scale=s, pad_x=pad_x, pad_y=pad_y
        ))
    return canvas, mappings


def assign_dets_to_tiles(dets_xyxy6: np.ndarray, mappings: List[ROIMapping]) -> Dict[int, List[np.ndarray]]:
    """将画布坐标的检测分配回tile索引（按中心点落入哪个tile）"""
    buckets: Dict[int, List[np.ndarray]] = {}
    if dets_xyxy6 is None or dets_xyxy6.size == 0:
        return buckets
    for det in dets_xyxy6:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        matched = -1
        for i, m in enumerate(mappings):
            if (m.cx <= cx < m.cx + m.tile_w) and (m.cy <= cy < m.cy + m.tile_h):
                matched = i
                break
        if matched == -1:
            continue
        m = mappings[matched]
        buckets.setdefault(matched, []).append(np.array([
            x1 - m.cx, y1 - m.cy, x2 - m.cx, y2 - m.cy, conf, cls_id
        ], dtype=np.float32))
    return buckets


def remap_tile_dets_to_original(
    buckets: Dict[int, List[np.ndarray]],
    mappings: List[ROIMapping]
) -> Dict[str, np.ndarray]:
    """将tile内坐标反投影回原始图像坐标。"""
    per_cam: Dict[str, List[List[float]]] = defaultdict(list)
    for idx, det_list in buckets.items():
        m = mappings[idx]
        s = m.scale
        px, py = m.pad_x, m.pad_y
        for det in det_list:
            tx1, ty1, tx2, ty2, conf, cls_id = det
            rx1 = (tx1 - px) / s
            ry1 = (ty1 - py) / s
            rx2 = (tx2 - px) / s
            ry2 = (ty2 - py) / s
            ox1 = m.x + rx1
            oy1 = m.y + ry1
            ox2 = m.x + rx2
            oy2 = m.y + ry2
            per_cam[m.cam_name].append([ox1, oy1, ox2, oy2, float(conf), float(cls_id)])
    out: Dict[str, np.ndarray] = {}
    for cam, lst in per_cam.items():
        out[cam] = np.array(lst, dtype=np.float32) if lst else np.empty((0, 6), dtype=np.float32)
    return out


# ---------------------------- Batch ROI 组装与回投 ----------------------------
@dataclass
class BatchMapping:
    cam_name: str
    x: int
    y: int
    w: int
    h: int
    scale: float
    pad_x: int
    pad_y: int


def build_batch_from_rois_with_mapping(
    frames: Dict[str, np.ndarray],
    rois: Dict[str, np.ndarray],
    target_size: int = ROI_BATCH_SIZE
) -> Tuple[np.ndarray, List[BatchMapping]]:
    batch: List[np.ndarray] = []
    maps: List[BatchMapping] = []
    for cam, boxes in rois.items():
        H, W = frames[cam].shape[:2]
        for b in boxes.astype(np.float32):
            x1, y1, x2, y2 = b[:4].tolist()
            x, y, w, h = clamp_rect(x1, y1, x2 - x1, y2 - y1, W, H)
            roi = frames[cam][y:y + h, x:x + w]
            if roi.size == 0:
                continue
            s = min(target_size / float(w), target_size / float(h))
            new_w = max(1, int(round(w * s)))
            new_h = max(1, int(round(h * s)))
            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            batch.append(canvas)
            maps.append(BatchMapping(cam_name=cam, x=x, y=y, w=w, h=h, scale=s, pad_x=pad_x, pad_y=pad_y))
    if len(batch) == 0:
        return np.zeros((0, target_size, target_size, 3), dtype=np.uint8), []
    return np.stack(batch, axis=0), maps


def remap_batch_detections_to_original(
    detections: List[np.ndarray],
    mappings: List[BatchMapping]
) -> Dict[str, np.ndarray]:
    per_cam: Dict[str, List[List[float]]] = defaultdict(list)
    for dets, m in zip(detections, mappings):
        if dets is None or dets.size == 0:
            continue
        s = m.scale
        px, py = m.pad_x, m.pad_y
        for det in dets:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det
            rx1 = (x1 - px) / s
            ry1 = (y1 - py) / s
            rx2 = (x2 - px) / s
            ry2 = (y2 - py) / s
            ox1 = m.x + rx1
            oy1 = m.y + ry1
            ox2 = m.x + rx2
            oy2 = m.y + ry2
            per_cam[m.cam_name].append([ox1, oy1, ox2, oy2, float(conf), float(cls_id)])
    out: Dict[str, np.ndarray] = {}
    for cam, lst in per_cam.items():
        out[cam] = np.array(lst, dtype=np.float32) if lst else np.empty((0, 6), dtype=np.float32)
    return out


# ---------------------------- 推理与评估 ----------------------------
def yolo_predict_numpy(model: UL_YOLO, source, imgsz: int) -> List:
    """对 numpy 输入执行推理，返回 ultralytics 的结果列表"""
    return model.predict(source=source, imgsz=imgsz, verbose=False, device=0)


def extract_xyxy6(ul_res) -> np.ndarray:
    """从单张结果提取 [x1,y1,x2,y2,conf,cls]，并按 ALLOWED_COCO_IDS 过滤"""
    if ul_res.boxes.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = ul_res.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = ul_res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    cls = ul_res.boxes.cls.cpu().numpy().astype(np.int32).reshape(-1, 1)
    keep = np.array([c in ALLOWED_COCO_IDS for c in cls.flatten()], dtype=bool)
    if not keep.any():
        return np.empty((0, 6), dtype=np.float32)
    return np.hstack((xyxy[keep], conf[keep], cls[keep].astype(np.float32)))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组框的IoU矩阵，a:[Na,4], b:[Nb,4]"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    # a: (Na, 1, 4), b: (1, Nb, 4)
    a_exp = a[:, None, :]
    b_exp = b[None, :, :]
    inter_x1 = np.maximum(a_exp[..., 0], b_exp[..., 0])
    inter_y1 = np.maximum(a_exp[..., 1], b_exp[..., 1])
    inter_x2 = np.minimum(a_exp[..., 2], b_exp[..., 2])
    inter_y2 = np.minimum(a_exp[..., 3], b_exp[..., 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (a_exp[..., 2] - a_exp[..., 0]) * (a_exp[..., 3] - a_exp[..., 1])
    area_b = (b_exp[..., 2] - b_exp[..., 0]) * (b_exp[..., 3] - b_exp[..., 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def match_detections_to_gts(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float) -> Tuple[int, int, int]:
    """
    简单贪心匹配：按预测置信度排序，逐个匹配GT（每个GT最多匹配一次）。
    返回 (TP, FP, FN)
    """
    if pred_xyxy.size == 0:
        return 0, 0, gt_xyxy.shape[0]
    # 置信度排序
    confs = pred_xyxy[:, 4]
    order = np.argsort(-confs)
    pred_sorted = pred_xyxy[order]
    IoU = iou_matrix(pred_sorted[:, :4], gt_xyxy)
    gt_matched = np.zeros((gt_xyxy.shape[0],), dtype=bool)
    tp = 0
    fp = 0
    for i in range(pred_sorted.shape[0]):
        ious = IoU[i]
        j = int(np.argmax(ious)) if ious.size > 0 else -1
        if j >= 0 and ious[j] >= iou_thr and not gt_matched[j]:
            gt_matched[j] = True
            tp += 1
        else:
            fp += 1
    fn = int((~gt_matched).sum())
    return tp, fp, fn


def compute_pr_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_frames_and_rois(group: GroupBatch) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    将一组图像映射为伪多摄像头，返回:
    - frames: cam -> BGR
    - rois:   cam -> [N,4] ROI(使用GT)
    - gts:    cam -> [N,4] GT(用于评估)
    """
    cam_names = ["front", "left", "right", "rear"]
    frames: Dict[str, np.ndarray] = {}
    rois: Dict[str, np.ndarray] = {}
    gts: Dict[str, np.ndarray] = {}
    for i, item in enumerate(group.images):
        name = cam_names[i % len(cam_names)]
        frames[name] = item.image
        rois[name] = item.gts.astype(np.float32) if item.gts is not None else np.empty((0, 4), dtype=np.float32)
        gts[name] = rois[name].copy()
    return frames, rois, gts


def run_one_group_canvas(model: UL_YOLO, frames: Dict[str, np.ndarray], rois: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # 组装 ROI items
    roi_items: List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]] = []
    for cam, boxes in rois.items():
        H, W = frames[cam].shape[:2]
        for b in boxes.astype(np.float32):
            x1, y1, x2, y2 = b[:4].tolist()
            x, y, w, h = clamp_rect(x1, y1, x2 - x1, y2 - y1, W, H)
            roi_items.append((cam, frames[cam], (x, y, w, h)))
    if not roi_items:
        return {k: np.empty((0, 6), dtype=np.float32) for k in frames.keys()}
    canvas, mappings = pack_rois_fixed_canvas(roi_items, CANVAS_W, CANVAS_H)
    res_list = yolo_predict_numpy(model, canvas, imgsz=max(CANVAS_W, CANVAS_H))
    # 单张画布结果
    dets6 = extract_xyxy6(res_list[0])
    buckets = assign_dets_to_tiles(dets6, mappings)
    remapped = remap_tile_dets_to_original(buckets, mappings)
    return remapped


def run_one_group_batch(model: UL_YOLO, frames: Dict[str, np.ndarray], rois: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    batch, maps = build_batch_from_rois_with_mapping(frames, rois, ROI_BATCH_SIZE)
    if batch.shape[0] == 0:
        return {k: np.empty((0, 6), dtype=np.float32) for k in frames.keys()}
    # 将4D转列表适配ultralytics
    source = [batch[i] for i in range(batch.shape[0])]
    res_list = yolo_predict_numpy(model, source, imgsz=ROI_BATCH_SIZE)
    dets_list = [extract_xyxy6(r) for r in res_list]
    remapped = remap_batch_detections_to_original(dets_list, maps)
    return remapped


def evaluate_groups(
    model_name: str,
    few_groups: List[GroupBatch],
    many_groups: List[GroupBatch],
    out_vis_dir: Path,
    vis_max: int,
    iou_thr: float
) -> List[AccRecord]:
    model = UL_YOLO(model_name)
    if torch is not None and torch.cuda.is_available():
        model.to("cuda")
    records: List[AccRecord] = []

    def eval_scenario(groups: List[GroupBatch], scenario: str) -> Tuple[AccRecord, AccRecord]:
        if len(groups) == 0:
            return (
                AccRecord(model=model_name, scenario=scenario, method="canvas", precision=float("nan"), recall=float("nan"), f1=float("nan"), num_groups=0),
                AccRecord(model=model_name, scenario=scenario, method="batch",  precision=float("nan"), recall=float("nan"), f1=float("nan"), num_groups=0),
            )
        # 聚合TP/FP/FN
        tp_c = fp_c = fn_c = 0
        tp_b = fp_b = fn_b = 0
        vis_saved = 0
        for gi, g in enumerate(groups):
            frames, rois, gts = build_frames_and_rois(g)
            # Canvas
            pred_canvas = run_one_group_canvas(model, frames, rois)
            # Batch
            pred_batch = run_one_group_batch(model, frames, rois)
            # 逐相机评估并可视化
            for cam in frames.keys():
                gt = gts.get(cam, np.empty((0, 4), dtype=np.float32))
                pc = pred_canvas.get(cam, np.empty((0, 6), dtype=np.float32))
                pb = pred_batch.get(cam, np.empty((0, 6), dtype=np.float32))
                tpc, fpc, fnc = match_detections_to_gts(pc, gt, iou_thr)
                tpb, fpb, fnb = match_detections_to_gts(pb, gt, iou_thr)
                tp_c += tpc; fp_c += fpc; fn_c += fnc
                tp_b += tpb; fp_b += fpb; fn_b += fnb
                # 可视化
                if vis_saved < vis_max:
                    vis = draw_compare_vis(frames[cam], gt, pc, pb)
                    vis_path = out_vis_dir / f"{scenario}_{model_name.replace(os.sep,'_')}_g{gi:04d}_{cam}.jpg"
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(vis_path), vis)
                    vis_saved += 1
        # 汇总指标
        p_c, r_c, f1_c = compute_pr_f1(tp_c, fp_c, fn_c)
        p_b, r_b, f1_b = compute_pr_f1(tp_b, fp_b, fn_b)
        rec_canvas = AccRecord(model=model_name, scenario=scenario, method="canvas",
                               precision=p_c, recall=r_c, f1=f1_c, num_groups=len(groups))
        rec_batch = AccRecord(model=model_name, scenario=scenario, method="batch",
                              precision=p_b, recall=r_b, f1=f1_b, num_groups=len(groups))
        return rec_canvas, rec_batch

    c_few, b_few = eval_scenario(few_groups, "few")
    c_many, b_many = eval_scenario(many_groups, "many")
    records.extend([c_few, b_few, c_many, b_many])
    return records


# ---------------------------- 可视化与输出 ----------------------------
def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int], label: str) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        if len(b) >= 4:
            x1, y1, x2, y2 = map(int, b[:4].tolist())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def draw_compare_vis(frame: np.ndarray, gt_xyxy: np.ndarray, pred_canvas6: np.ndarray, pred_batch6: np.ndarray) -> np.ndarray:
    gt_vis = draw_boxes(frame, gt_xyxy, (0, 255, 0), "GT")
    pc_vis = draw_boxes(frame, pred_canvas6[:, :4] if pred_canvas6.size else pred_canvas6, (255, 0, 0), "Canvas")
    pb_vis = draw_boxes(frame, pred_batch6[:, :4] if pred_batch6.size else pred_batch6, (0, 0, 255), "Batch")
    # 拼接
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = gt_vis
    canvas[:, w:2 * w] = pc_vis
    canvas[:, 2 * w:3 * w] = pb_vis
    return canvas


def write_csv(records: List[AccRecord], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scenario", "method", "precision@0.5", "recall@0.5", "f1@0.5", "num_groups"])
        for r in records:
            w.writerow([r.model, r.scenario, r.method, f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}", r.num_groups])


def plot_metrics(records: List[AccRecord], out_dir: Path):
    if plt is None:
        return
    for scenario in ("few", "many"):
        rec_s = [r for r in records if r.scenario == scenario]
        if not rec_s:
            continue
        by_model = defaultdict(dict)  # model -> method -> (p,r,f1)
        for r in rec_s:
            by_model[r.model][r.method] = (r.precision, r.recall, r.f1)
        models = sorted(by_model.keys())
        x = np.arange(len(models))
        width = 0.35
        # Precision
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 4.2))
        ax.bar(x - width / 2, [by_model[m].get("canvas", (np.nan,))[0] for m in models], width, label="Canvas", color="#3182bd")
        ax.bar(x + width / 2, [by_model[m].get("batch", (np.nan,))[0] for m in models], width, label="Batch", color="#fd8d3c")
        ax.set_ylabel("Precision@0.5"); ax.set_title(f"{scenario.upper()} ROIs - Precision")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right'); ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout(); fig.savefig(out_dir / f"acc_{scenario}_precision.png", dpi=150); plt.close(fig)
        # Recall
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 4.2))
        ax.bar(x - width / 2, [by_model[m].get("canvas", (np.nan, np.nan))[1] for m in models], width, label="Canvas", color="#6baed6")
        ax.bar(x + width / 2, [by_model[m].get("batch", (np.nan, np.nan))[1] for m in models], width, label="Batch", color="#fdae6b")
        ax.set_ylabel("Recall@0.5"); ax.set_title(f"{scenario.upper()} ROIs - Recall")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right'); ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout(); fig.savefig(out_dir / f"acc_{scenario}_recall.png", dpi=150); plt.close(fig)
        # F1
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 4.2))
        ax.bar(x - width / 2, [by_model[m].get("canvas", (np.nan, np.nan, np.nan))[2] for m in models], width, label="Canvas", color="#08519c")
        ax.bar(x + width / 2, [by_model[m].get("batch", (np.nan, np.nan, np.nan))[2] for m in models], width, label="Batch", color="#d94801")
        ax.set_ylabel("F1@0.5"); ax.set_title(f"{scenario.upper()} ROIs - F1")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right'); ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout(); fig.savefig(out_dir / f"acc_{scenario}_f1.png", dpi=150); plt.close(fig)


# ---------------------------- 主流程 ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", type=str, required=True, help="KITTI-tracking 根目录（包含 image_02、label_02、calib）")
    ap.add_argument("--num-images", type=int, default=800, help="随机抽取图像帧数量（跨序列）")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE, help="每组图像数量（模拟多摄像头）")
    ap.add_argument("--iou-thr", type=float, default=IOU_THRESH, help="评估IoU阈值")
    ap.add_argument("--models", nargs="+", default=[
        "yolov8n.pt", "yolov8s.pt",
        "yolov11n.pt", "yolov11s.pt",
        "rtdetr-l.pt"
    ], help="待评测的模型名称/权重路径")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="roi_acc_results")
    ap.add_argument("--vis-max-per-model", type=int, default=VIS_MAX_PER_MODEL)
    args = ap.parse_args()

    group_size = args.group_size
    iou_thr = args.iou_thr

    ensure_env()

    root = Path(args.kitti_root)
    all_items = build_dataset_items(root, num_images=args.num_images, seed=args.seed)
    # 分组
    groups: List[GroupBatch] = []
    for i in range(0, len(all_items) - group_size + 1, group_size):
        groups.append(GroupBatch(images=all_items[i:i + group_size]))
    # 少/多ROI划分（以内含GT框数量为基准）
    few_groups: List[GroupBatch] = []
    many_groups: List[GroupBatch] = []
    for g in groups:
        roi_count = sum(item.gts.shape[0] for item in g.images)
        if roi_count >= FEW_MANY_THRESHOLD:
            many_groups.append(g)
        else:
            few_groups.append(g)
    print(f"[prep] 分组完成: 少ROI={len(few_groups)}, 多ROI={len(many_groups)}, 组大小={group_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[AccRecord] = []
    for m in args.models:
        print(f"[eval] 模型 {m}: 少/多ROI 两种方法精度评测...")
        vis_dir = out_dir / f"vis_{Path(m).stem}"
        recs = evaluate_groups(m, few_groups, many_groups, out_vis_dir=vis_dir, vis_max=args.vis_max_per_model, iou_thr=iou_thr)
        all_records.extend(recs)

    out_csv = out_dir / "roi_canvas_vs_batch_accuracy.csv"
    write_csv(all_records, out_csv)
    print(f"[out] CSV写入: {out_csv}")

    plot_metrics(all_records, out_dir)
    print(f"[out] 图表输出到: {out_dir}")


if __name__ == "__main__":
    main()


