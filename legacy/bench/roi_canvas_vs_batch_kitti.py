#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Canvas vs Batch Benchmark on KITTI (randomized multi-camera groups)
---------------------------------------------------------------------
目的：对比两种基于ROI的推理方式在多摄像头场景的延迟表现：
1) 画布推理：将各摄像头ROI缩放并letterbox后拼接到固定画布，一次GPU推理。
2) 批次推理：将所有ROI缩放至统一尺寸，堆叠为一个小批次进行GPU推理。

流程概述：
1) 随机从KITTI数据集中抽取若干图像（默认1000张），每4张组成一次多摄像头输入。
2) 使用一个较快的计数模型（默认yolov8n.pt）对每张图全图推理，统计ROI数量并缓存每张图的ROI框。
3) 将每个4图组合按ROI总数分到“少ROI(<20)”与“多ROI(>=20)”两类。
4) 对每个候选模型（YOLOv8/YOLOv11各版本、RT-DETR等），分别在两类样本上基于同一批ROI框：
   - 执行画布推理并计时
   - 执行小批次推理并计时
5) 输出CSV、并绘制对比图（每模型在少/多ROI下的两种方法平均延迟）。

注意：
- 本脚本关注推理延迟，未计算精度；ROI框来源于计数模型，仅用于构造裁剪区域以统一对比速度。
- 若无GPU，脚本会抛错（Ultralytics在CPU上也能跑，但延迟失真，建议GPU）。

用法示例：
python roi_canvas_vs_batch_kitti.py \
  --kitti-dir D:/data/KITTI/image_2 \
  --num-images 1000 \
  --group-size 4 \
  --imgsz 640 \
  --models yolov8n.pt yolov8s.pt yolov11n.pt yolov11s.pt rtdetr-l.pt \
  --out-dir results_roi_bench
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from ultralytics import YOLO as UL_YOLO
except Exception:
    UL_YOLO = None


# ---------------------------- 配置常量 ----------------------------
ROI_BATCH_SIZE = 320          # ROI缩放后的小批次边长（正方形）
CANVAS_W, CANVAS_H = 1024, 1024  # 画布大小
ALLOWED_CLS = None            # 若限制类别，可设置为集合，如 {0,1,2,3,5,7}；None 表示不过滤


# ---------------------------- 数据结构 ----------------------------
@dataclass
class ImageROIs:
    path: Path
    image: np.ndarray
    boxes_xyxyc: np.ndarray  # shape [N,5], float32, last is cls


@dataclass
class BenchRecord:
    model: str
    scenario: str  # "few" or "many"
    method: str    # "canvas" or "batch"
    infer_ms_mean: float
    total_ms_mean: float
    num_groups: int


# ---------------------------- 工具函数 ----------------------------
def ensure_cuda():
    if torch is None:
        raise RuntimeError("未检测到PyTorch，请先安装 torch。")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA设备，请在带NVIDIA GPU的环境下运行。")


def load_images_from_dir(root: Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def yolov_full_detect(model: UL_YOLO, bgr: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """
    返回xyxy(float32)的检测框，shape [N,4]。若 ALLOWED_CLS 非空，则按类别过滤。
    """
    res = model.predict(source=bgr, imgsz=imgsz, verbose=False, device=0)[0]
    if res.boxes.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = res.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
    if ALLOWED_CLS is not None:
        cls_i = cls.astype(np.int32).flatten()
        keep = np.array([c in ALLOWED_CLS for c in cls_i], dtype=bool)
        xyxy = xyxy[keep]
        cls = cls[keep]
    return np.hstack([xyxy, cls])


def clamp_rect(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > W:
        w = max(1, W - x)
    if y + h > H:
        h = max(1, H - y)
    return x, y, w, h


def resize_roi_to_batch_size(roi: np.ndarray, target_size: int = ROI_BATCH_SIZE) -> np.ndarray:
    if roi is None or roi.size == 0:
        return np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    h, w = roi.shape[:2]
    if w <= 0 or h <= 0:
        return np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def grid_for_n(n: int, W: int, H: int) -> Tuple[int, int, int, int]:
    n = max(1, n)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    tile_w = W // cols
    tile_h = H // rows
    return rows, cols, tile_w, tile_h


def build_canvas_from_rois(frames: Dict[str, np.ndarray], rois: Dict[str, np.ndarray], canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    frames: 摄像头名 -> 原始BGR图
    rois:   摄像头名 -> [N_i,4] xyxy
    返回 BGR 画布图像。
    """
    all_rects: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for cam, boxes in rois.items():
        H, W = frames[cam].shape[:2]
        for b in boxes.astype(np.int32):
            x1, y1, x2, y2 = b.tolist()
            x, y, w, h = clamp_rect(x1, y1, x2 - x1, y2 - y1, W, H)
            all_rects.append((cam, (x, y, w, h)))

    if len(all_rects) == 0:
        return np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)

    n = len(all_rects)
    rows, cols, tile_w, tile_h = grid_for_n(n, canvas_w, canvas_h)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 114, dtype=np.uint8)

    for idx, (cam, (x, y, w, h)) in enumerate(all_rects):
        frame = frames[cam]
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue
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
    return canvas


def _letterbox_into(dst: np.ndarray, src: np.ndarray) -> None:
    """将 src 等比缩放后放入 dst 居中（dst 已创建好，背景114）。"""
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    if sw == 0 or sh == 0 or dw == 0 or dh == 0:
        return
    s = min(dw / float(sw), dh / float(sh))
    tw = max(1, int(round(sw * s)))
    th = max(1, int(round(sh * s)))
    resized = cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)
    px = (dw - tw) // 2
    py = (dh - th) // 2
    dst[py:py + th, px:px + tw] = resized


def _danger_priority(cls_id: int) -> int:
    """类别危险度优先级，数值越大优先级越高。
    默认映射（Ultralytics COCO id）：
      行人(0)、自行车(1)、摩托(3) -> 3
      汽车(2)、公交(5)、卡车(7)     -> 2
      其他                          -> 1
    若无类别信息，返回 1。
    """
    if cls_id is None:
        return 1
    high = {0, 1, 3}
    mid = {2, 5, 7}
    if cls_id in high:
        return 3
    if cls_id in mid:
        return 2
    return 1

def build_canvas_from_rois_guillotine(frames: Dict[str, np.ndarray], rois: Dict[str, np.ndarray], canvas_w: int, canvas_h: int) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    使用 Guillotine 二维装箱将多ROI按优先级放入单张画布。
    策略：
    - 先为每个ROI计算目标尺寸（保持比例，不放大），目标最长边约为 sqrt(canvas_area / n)。
    - 将ROI按优先级（默认原始面积降序）排序，逐个放入空闲矩形列表F（初始为整个画布）。
    - 选择第一个能容纳该ROI的空闲矩形，放在其左上角；若空闲矩形高>宽则水平分割，否则垂直分割；
      生成两个新的空闲矩形并加入F；再删去被完全包含的冗余空闲矩形。
    - 未能容纳的低优先级ROI丢弃。
    """
    # 收集所有 ROI 原始矩形
    items: List[Tuple[str, int, Tuple[int, int, int, int], int]] = []  # (cam, idx, (x,y,w,h), cls)
    for cam, boxes in rois.items():
        H, W = frames[cam].shape[:2]
        for idx, b in enumerate(boxes.astype(np.float32)):
            if b.shape[0] >= 5:
                x1, y1, x2, y2, clsf = b[:5].tolist()
                cls_id = int(clsf)
            else:
                x1, y1, x2, y2 = b[:4].tolist()
                cls_id = 0
            x, y, w, h = clamp_rect(x1, y1, x2 - x1, y2 - y1, W, H)
            if w <= 0 or h <= 0:
                continue
            items.append((cam, idx, (x, y, w, h), cls_id))

    n = len(items)
    canvas = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)
    if n == 0:
        return canvas

    # 目标尺寸：使单个ROI最长边约为 sqrt(A/n)，避免极小tile
    target_side = int(max(32, min(canvas_w, canvas_h, math.sqrt((canvas_w * canvas_h) / max(1, n)))))

    # 计算优先级（默认按原始面积降序）；也保存目标尺寸
    patches: List[Tuple[float, str, int, Tuple[int, int, int, int], int, int]] = []  # (prio, cam, idx, (x,y,w,h), tw, th)
    for cam, idx, (x, y, w, h), cls_id in items:
        # 危险度优先级：行人/骑行类最高，机动车次之，其它较低
        # 可按需要调整映射
        danger_rank = _danger_priority(cls_id)
        prio = float(danger_rank * 1e9 + (w * h))
        s = min(target_side / float(w), target_side / float(h), 1.0)
        tw = max(1, int(w * s))
        th = max(1, int(h * s))
        patches.append((prio, cam, idx, (x, y, w, h), tw, th))

    patches.sort(key=lambda t: t[0], reverse=True)

    # 空闲矩形列表 F: (fx, fy, fw, fh)
    F: List[Tuple[int, int, int, int]] = [(0, 0, canvas_w, canvas_h)]

    # 记录放置结果：每个条目为 (cam, idx, src_rect(x,y,w,h), dst_rect(x,y,w,h))
    placements: List[Tuple[str, int, Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []

    def prune_free_list(free_list: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        pruned: List[Tuple[int, int, int, int]] = []
        for i, a in enumerate(free_list):
            ax, ay, aw, ah = a
            contained = False
            for j, b in enumerate(free_list):
                if i == j:
                    continue
                bx, by, bw, bh = b
                if ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh:
                    contained = True
                    break
            if not contained and aw > 0 and ah > 0:
                pruned.append(a)
        return pruned

    for _, cam, idx, src_rect, tw, th in patches:
        placed = False
        for fi, (fx, fy, fw, fh) in enumerate(F):
            if fw >= tw and fh >= th:
                # 放在左上角
                px, py = fx, fy
                placements.append((cam, idx, src_rect, (px, py, tw, th)))

                # 从该空闲矩形切分出两个新空闲矩形
                del F[fi]
                if fh > fw:
                    # 水平分割：上方右侧+下方整宽
                    right = (px + tw, py, fw - tw, th)
                    below = (fx, fy + th, fw, fh - th)
                    if right[2] > 0 and right[3] > 0:
                        F.append(right)
                    if below[2] > 0 and below[3] > 0:
                        F.append(below)
                else:
                    # 垂直分割：右侧整高+下方等宽
                    right = (fx + tw, fy, fw - tw, fh)
                    below = (fx, fy + th, tw, fh - th)
                    if right[2] > 0 and right[3] > 0:
                        F.append(right)
                    if below[2] > 0 and below[3] > 0:
                        F.append(below)

                # 清理被包含的空闲矩形
                F = prune_free_list(F)
                placed = True
                break
        # 放不下则跳过该低优先级补丁
        if not placed:
            continue

    # 将放置结果渲染到画布
    for cam, idx, (sx, sy, sw, sh), (dx, dy, dw, dh) in placements:
        roi = frames[cam][sy:sy + sh, sx:sx + sw]
        if roi.size == 0 or dw <= 0 or dh <= 0:
            continue
        tile = np.full((dh, dw, 3), 114, dtype=np.uint8)
        _letterbox_into(tile, roi)
        canvas[dy:dy + dh, dx:dx + dw] = tile

    # 汇总已放置索引
    placed_map: Dict[str, List[int]] = defaultdict(list)
    for cam, idx, _, _ in placements:
        placed_map[cam].append(idx)
    for cam in placed_map:
        placed_map[cam] = sorted(set(placed_map[cam]))

    return canvas, placed_map

def build_batch_from_rois(frames: Dict[str, np.ndarray], rois: Dict[str, np.ndarray]) -> np.ndarray:
    """
    生成小批次的ROI张量，形状 [B, H, W, 3]，uint8。
    """
    batch: List[np.ndarray] = []
    for cam, boxes in rois.items():
        H, W = frames[cam].shape[:2]
        for b in boxes.astype(np.float32):
            x1, y1, x2, y2 = b[:4].tolist()
            x, y, w, h = clamp_rect(x1, y1, x2 - x1, y2 - y1, W, H)
            roi = frames[cam][y:y + h, x:x + w]
            if roi.size == 0:
                continue
            batch.append(resize_roi_to_batch_size(roi, ROI_BATCH_SIZE))
    if len(batch) == 0:
        return np.zeros((0, ROI_BATCH_SIZE, ROI_BATCH_SIZE, 3), dtype=np.uint8)
    return np.stack(batch, axis=0)


def time_predict_numpy(model: UL_YOLO, inp: np.ndarray, imgsz: int) -> Tuple[float, float]:
    """
    对numpy输入进行一次 .predict 计时，返回 (infer_ms, total_ms)。
    - infer_ms: 仅模型 .predict 的GPU同步后耗时
    - total_ms: 函数整体的耗时（此处与infer_ms几乎一致，保留以便扩展）
    """
    # 将4D批次转换为列表以适配Ultralytics的letterbox预处理
    source = inp
    if isinstance(inp, np.ndarray) and inp.ndim == 4:
        b = inp.shape[0]
        if b == 0:
            return 0.0, 0.0
        source = [inp[i] for i in range(b)]

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = model.predict(source=source, imgsz=imgsz, verbose=False, device=0)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0
    total_ms = infer_ms
    _ = res  # 不使用结果，仅计时
    return infer_ms, total_ms


# ---------------------------- 主流程 ----------------------------
def precompute_image_rois(img_paths: List[Path], count_model_path: str, imgsz: int) -> Dict[Path, ImageROIs]:
    """
    用计数模型对每张图进行一次全图推理，记录ROI框与数量。
    """
    if UL_YOLO is None:
        raise RuntimeError("未安装 ultralytics，请先 pip install ultralytics。")
    count_model = UL_YOLO(count_model_path)
    count_model.to("cuda")

    out: Dict[Path, ImageROIs] = {}
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        boxes_xyxyc = yolov_full_detect(count_model, bgr, imgsz=imgsz)
        out[p] = ImageROIs(path=p, image=bgr, boxes_xyxyc=boxes_xyxyc)
    return out


def group_images_as_cameras(imgs: List[ImageROIs], group_size: int = 4) -> List[List[ImageROIs]]:
    groups: List[List[ImageROIs]] = []
    for i in range(0, len(imgs) - group_size + 1, group_size):
        groups.append(imgs[i:i + group_size])
    return groups


def classify_groups(groups: List[List[ImageROIs]], roi_threshold: int = 20) -> Tuple[List[List[ImageROIs]], List[List[ImageROIs]]]:
    few, many = [], []
    for g in groups:
        total = sum(len(x.boxes_xyxyc) for x in g)
        if total >= roi_threshold:
            many.append(g)
        else:
            few.append(g)
    return few, many


def build_frames_and_rois(group: List[ImageROIs]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cam_names = ["front", "left", "right", "rear"]
    frames: Dict[str, np.ndarray] = {}
    rois: Dict[str, np.ndarray] = {}
    for i, item in enumerate(group):
        name = cam_names[i % len(cam_names)]
        frames[name] = item.image
        # 保留 xyxyc（包含类别），供优先级使用
        boxes = item.boxes_xyxyc
        if boxes is None or boxes.size == 0:
            rois[name] = np.empty((0, 5), dtype=np.float32)
        else:
            rois[name] = boxes.astype(np.float32)
    return frames, rois


def run_bench_for_model(model_name: str,
                        few_groups: List[List[ImageROIs]],
                        many_groups: List[List[ImageROIs]],
                        imgsz: int) -> List[BenchRecord]:
    if UL_YOLO is None:
        raise RuntimeError("未安装 ultralytics，请先安装。")
    model = UL_YOLO(model_name)
    model.to("cuda")

    records: List[BenchRecord] = []

    def bench_scenario(groups: List[List[ImageROIs]], scenario: str) -> Tuple[BenchRecord, BenchRecord]:
        if len(groups) == 0:
            return (
                BenchRecord(model=model_name, scenario=scenario, method="canvas", infer_ms_mean=float("nan"), total_ms_mean=float("nan"), num_groups=0),
                BenchRecord(model=model_name, scenario=scenario, method="batch", infer_ms_mean=float("nan"), total_ms_mean=float("nan"), num_groups=0),
            )

        canvas_infer_ms, canvas_total_ms = [], []
        batch_infer_ms, batch_total_ms = [], []

        for group in groups:
            frames, rois = build_frames_and_rois(group)

            # Canvas
            t_pack0 = time.perf_counter()
            # 使用 Guillotine 算法，放不下则新增画布继续，直到全部 ROI 放完
            canvas_list: List[np.ndarray] = []
            remaining_rois = {k: v.copy() for k, v in rois.items()}
            while True:
                # 结束条件：所有相机都没有剩余ROI
                if all((arr is None or arr.size == 0) for arr in remaining_rois.values()):
                    break
                canvas, placed_map = build_canvas_from_rois_guillotine(frames, remaining_rois, CANVAS_W, CANVAS_H)
                canvas_list.append(canvas)
                # 从 remaining_rois 中移除已放置索引
                for cam, idxs in placed_map.items():
                    if cam not in remaining_rois:
                        continue
                    arr = remaining_rois[cam]
                    if arr is None or arr.size == 0:
                        continue
                    mask = np.ones(arr.shape[0], dtype=bool)
                    idxs = [i for i in idxs if 0 <= i < arr.shape[0]]
                    mask[idxs] = False
                    remaining = arr[mask]
                    remaining_rois[cam] = remaining

            t_pack1 = time.perf_counter()
            # 对所有画布逐一推理并累计时间
            infer_ms_sum = 0.0
            for canvas in canvas_list:
                infer_ms, _ = time_predict_numpy(model, canvas, imgsz=imgsz)
                infer_ms_sum += infer_ms
            canvas_infer_ms.append(infer_ms_sum)
            canvas_total_ms.append(infer_ms_sum + (t_pack1 - t_pack0) * 1000.0)
            # 已在上方统计所有画布推理时间 infer_ms_sum；不再重复计时

            # Batch
            t_pack0 = time.perf_counter()
            roi_batch = build_batch_from_rois(frames, rois)
            t_pack1 = time.perf_counter()
            if roi_batch.shape[0] > 0:
                infer_ms, total_ms = time_predict_numpy(model, roi_batch, imgsz=ROI_BATCH_SIZE)
            else:
                infer_ms, total_ms = 0.0, 0.0
            batch_infer_ms.append(infer_ms)
            batch_total_ms.append(infer_ms + (t_pack1 - t_pack0) * 1000.0)

        rec_canvas = BenchRecord(
            model=model_name,
            scenario=scenario,
            method="canvas",
            infer_ms_mean=float(np.mean(canvas_infer_ms)) if canvas_infer_ms else float("nan"),
            total_ms_mean=float(np.mean(canvas_total_ms)) if canvas_total_ms else float("nan"),
            num_groups=len(groups),
        )
        rec_batch = BenchRecord(
            model=model_name,
            scenario=scenario,
            method="batch",
            infer_ms_mean=float(np.mean(batch_infer_ms)) if batch_infer_ms else float("nan"),
            total_ms_mean=float(np.mean(batch_total_ms)) if batch_total_ms else float("nan"),
            num_groups=len(groups),
        )
        return rec_canvas, rec_batch

    c_few, b_few = bench_scenario(few_groups, "few")
    c_many, b_many = bench_scenario(many_groups, "many")
    records.extend([c_few, b_few, c_many, b_many])
    return records


def run_bench_for_model_batch_ge(model_name: str,
                                 few_groups: List[List[ImageROIs]],
                                 many_groups: List[List[ImageROIs]],
                                 imgsz: int,
                                 batch_min: int) -> List[BenchRecord]:
    """
    在与原脚本相同的数据与分组上，追加仅针对 Batch 方法的“batch>=N”场景。
    为了达到批量下限，将在需要时对ROI样本进行有放回重复，以构造至少 batch_min 的批次。
    - 场景命名：few_ge{N} / many_ge{N}
    - 方法命名仍为 "batch"（便于沿用CSV结构；绘图函数默认忽略额外场景，不影响现有功能）
    """
    if UL_YOLO is None:
        raise RuntimeError("未安装 ultralytics，请先安装。")
    model = UL_YOLO(model_name)
    model.to("cuda")

    def bench_scenario_batchmin(groups: List[List[ImageROIs]], scenario_label: str) -> BenchRecord:
        if len(groups) == 0:
            return BenchRecord(model=model_name, scenario=scenario_label, method="batch",
                               infer_ms_mean=float("nan"), total_ms_mean=float("nan"), num_groups=0)
        infer_ms_list, total_ms_list = [], []
        for group in groups:
            frames, rois = build_frames_and_rois(group)
            # 打包ROI为小批次
            t_pack0 = time.perf_counter()
            roi_batch = build_batch_from_rois(frames, rois)  # [B,H,W,3]
            # 达不到阈值时，重复采样填充
            b = int(roi_batch.shape[0])
            if b > 0 and b < batch_min:
                need = batch_min - b
                idx = np.random.choice(b, size=need, replace=True)
                extra = roi_batch[idx]
                roi_batch = np.concatenate([roi_batch, extra], axis=0)
            t_pack1 = time.perf_counter()
            if roi_batch.shape[0] > 0:
                infer_ms, _ = time_predict_numpy(model, roi_batch, imgsz=ROI_BATCH_SIZE)
            else:
                infer_ms = 0.0
            total_ms = infer_ms + (t_pack1 - t_pack0) * 1000.0
            infer_ms_list.append(infer_ms)
            total_ms_list.append(total_ms)
        return BenchRecord(
            model=model_name,
            scenario=scenario_label,
            method="batch",
            infer_ms_mean=float(np.mean(infer_ms_list)) if infer_ms_list else float("nan"),
            total_ms_mean=float(np.mean(total_ms_list)) if total_ms_list else float("nan"),
            num_groups=len(groups),
        )

    recs: List[BenchRecord] = []
    recs.append(bench_scenario_batchmin(few_groups, f"few_ge{int(batch_min)}"))
    recs.append(bench_scenario_batchmin(many_groups, f"many_ge{int(batch_min)}"))
    return recs


def plot_results(records: List[BenchRecord], out_dir: Path, metric: str = "total_ms_mean"):
    if plt is None:
        return
    # 按场景分别绘图
    for scenario in ("few", "many"):
        rec_s = [r for r in records if r.scenario == scenario]
        if not rec_s:
            continue
        by_model = defaultdict(dict)  # model -> {method: value}
        for r in rec_s:
            by_model[r.model][r.method] = getattr(r, metric)

        models = sorted(by_model.keys())
        # total 平均值
        canvas_total = [by_model[m].get("canvas", float("nan")) for m in models]
        batch_total = [by_model[m].get("batch", float("nan")) for m in models]
        # 需要推理均值以拆分装箱与推理；到函数外部不可见，故再从 records 取 infer
        infer_by_model = defaultdict(dict)
        for r in rec_s:
            if r.method == "canvas":
                infer_by_model[r.model]["canvas_infer"] = r.infer_ms_mean
            elif r.method == "batch":
                infer_by_model[r.model]["batch_infer"] = r.infer_ms_mean
        canvas_infer = [infer_by_model[m].get("canvas_infer", float("nan")) for m in models]
        batch_infer = [infer_by_model[m].get("batch_infer", float("nan")) for m in models]
        # 计算装箱时间 = total - infer（负值用0）
        canvas_pack = [max(0.0, (canvas_total[i] - canvas_infer[i])) if not (np.isnan(canvas_total[i]) or np.isnan(canvas_infer[i])) else float("nan") for i in range(len(models))]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
        # 画布：堆叠显示 pack 和 infer
        b1 = ax.bar(x - width / 2, canvas_pack, width, label="Canvas-Pack", color="#9ecae1")
        b2 = ax.bar(x - width / 2, canvas_infer, width, bottom=canvas_pack, label="Canvas-Infer", color="#3182bd")
        # 批次：仍显示总耗时（如需堆叠，可用 batch_total 与 (batch_total-batch_infer) 拆分）
        ax.bar(x + width / 2, batch_total, width, label="Batch-Total", color="#fd8d3c")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{scenario.upper()} ROIs - {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        out_path = out_dir / f"bench_{scenario}_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def write_csv(records: List[BenchRecord], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scenario", "method", "infer_ms_mean", "total_ms_mean", "num_groups"])
        for r in records:
            w.writerow([r.model, r.scenario, r.method, f"{r.infer_ms_mean:.3f}", f"{r.total_ms_mean:.3f}", r.num_groups])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-dir", type=str, required=True, help="KITTI图像目录（如 image_2 ）")
    ap.add_argument("--num-images", type=int, default=1000, help="随机抽取的图像数量")
    ap.add_argument("--group-size", type=int, default=4, help="每次多摄像头的图像数量")
    ap.add_argument("--imgsz", type=int, default=640, help="全图推理尺寸")
    ap.add_argument("--count-model", type=str, default="yolov8n.pt", help="用于ROI计数的模型")
    ap.add_argument(
        "--models", nargs="+", default=[
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11x.pt",
            "rtdetr-l.pt"
        ], help="待评测的模型名称/权重路径")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="roi_bench_results")
    ap.add_argument(
        "--batch-min-ge", type=int, nargs="+", default=None,
        help="追加一个或多个 batch>=N 的附加场景（仅对Batch方法生效，示例: --batch-min-ge 50 64）。"
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_cuda()
    if cv2 is None:
        raise RuntimeError("未安装opencv-python，请先安装。")

    kitti_dir = Path(args.kitti_dir)
    if not kitti_dir.exists():
        raise FileNotFoundError(f"未找到目录: {kitti_dir}")

    all_imgs = load_images_from_dir(kitti_dir)
    if len(all_imgs) == 0:
        raise RuntimeError("未在目录中找到图像。")

    # 随机抽样并打乱
    sample_paths = random.sample(all_imgs, min(args.num_images, len(all_imgs)))
    random.shuffle(sample_paths)

    # 计数模型进行一次全图推理，得到每张图的ROI框
    print(f"[prep] 使用 {args.count_model} 对 {len(sample_paths)} 张图进行ROI计数与缓存...")
    per_image = precompute_image_rois(sample_paths, args.count_model, imgsz=args.imgsz)

    # 将图像对象列表化（过滤读取失败的）
    images = [v for v in per_image.values() if v.image is not None]
    # 组装为多摄像头组
    groups = group_images_as_cameras(images, group_size=args.group_size)
    few_groups, many_groups = classify_groups(groups, roi_threshold=20)
    print(f"[prep] 生成分组: 少ROI={len(few_groups)}, 多ROI={len(many_groups)} (每组{args.group_size}张)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 针对每个模型跑两类场景
    all_records: List[BenchRecord] = []
    for m in args.models:
        print(f"[bench] 模型: {m} 在 少/多ROI 上进行画布与批次基准测试...")
        recs = run_bench_for_model(m, few_groups, many_groups, imgsz=args.imgsz)
        all_records.extend(recs)
        # 追加 batch>=N 的扩展场景（不影响默认绘图，仅写入CSV，场景名 few_geN/many_geN）
        if args.batch_min_ge:
            for ge in args.batch_min_ge:
                try:
                    ge_int = int(ge)
                except Exception:
                    continue
                if ge_int <= 0:
                    continue
                print(f"[bench-extra] 模型: {m} 追加 Batch>= {ge_int} 场景...")
                extra = run_bench_for_model_batch_ge(m, few_groups, many_groups, imgsz=args.imgsz, batch_min=ge_int)
                all_records.extend(extra)

    # 输出CSV与图表
    out_csv = out_dir / "roi_canvas_vs_batch.csv"
    write_csv(all_records, out_csv)
    print(f"[out] CSV写入: {out_csv}")

    plot_results(all_records, out_dir, metric="total_ms_mean")
    plot_results(all_records, out_dir, metric="infer_ms_mean")
    print(f"[out] 图表输出到: {out_dir}")


if __name__ == "__main__":
    main()


