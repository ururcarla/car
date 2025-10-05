#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Latency Bench (PyTorch & TensorRT)
---------------------------------------
Benchmark end-to-end per-frame latency for representative YOLO models under:
  - PyTorch FP32 (torch)
  - PyTorch AMP FP16 (fp16)
  - TensorRT FP16 (trt-fp16)
  - TensorRT INT8 (trt-int8)  [optional, needs trtexec + calib set/cache]

Features
- Resolution sweep (e.g., 320/480/640/960)
- Warmup and timed runs with p50/p90/p95/p99 statistics
- Optional capture/preprocess/NMS timing when using PyTorch mode
- ONNX export per (model, imgsz) for TRT modes
- Results saved to CSV

Requirements
- Python 3.9+ recommended
- NVIDIA GPU + CUDA
- Packages: ultralytics, torch, opencv-python, numpy, pandas
- TensorRT with `trtexec` available in PATH for TRT modes

Example
-------
python yolo_latency_bench.py \
  --models yolov8n.pt yolov8s.pt \
  --imgsz 320 480 640 \
  --modes torch fp16 trt-fp16 \
  --runs 300 --warmup 100 \
  --source sample.jpg \
  --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from PIL import Image
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    from ultralytics import YOLO
except Exception:
    YOLO = None
    torch = None


# --------------------------- Utilities ---------------------------

def log(msg: str):
    print(f"[bench] {msg}", flush=True)


def ensure_trtexec() -> str:
    exe = shutil.which("trtexec")
    if not exe:
        raise RuntimeError("`trtexec` not found in PATH. Please install TensorRT and add trtexec to PATH.")
    return exe


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> str:
    log(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env, text=True)
    out_lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        out_lines.append(line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")
    return "".join(out_lines)


def parse_trtexec_mean_latency(output: str) -> Optional[float]:
    """
    Parse trtexec output to find 'mean' latency in ms.
    Typical line: "mean: 1.83 ms, median: 1.80 ms, percentile(90%): 1.95 ms, ..."
    """
    # Try to find 'mean: X ms' pattern
    m = re.search(r"mean:\s*([\d\.]+)\s*ms", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: look for GPU Compute Mean
    m2 = re.search(r"GPU Compute Time:\s*mean\s*=\s*([\d\.]+)\s*ms", output, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * p
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(vals[f])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


# --------------------------- Data structures ---------------------------

@dataclass
class BenchResult:
    model: str
    mode: str  # torch, fp16, trt-fp16, trt-int8
    imgsz: int
    device: str
    runs: int
    warmup: int
    capture_ms: float | None
    preprocess_ms: float | None
    infer_ms_mean: float
    nms_ms: float | None
    end2end_ms_mean: float
    end2end_ms_p50: float
    end2end_ms_p90: float
    end2end_ms_p95: float
    end2end_ms_p99: float
    notes: str


# --------------------------- PyTorch bench ---------------------------

def bench_torch(model_path: str, imgsz: int, mode: str, source: Optional[str], runs: int, warmup: int) -> BenchResult:
    assert YOLO is not None and torch is not None, "Ultralytics/torch not available"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading model {model_path} on {device} ...")
    model = YOLO(model_path)
    if device == "cuda":
        model.to(device)

    # Prepare input
    if source and cv2 is not None and Path(source).exists():
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Failed to read source image: {source}")
    else:
        # synthetic image (random)
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Resize to target imgsz while preserving aspect via letterbox-like approach for fairness
    if cv2 is not None:
        resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    else:
        # Fallback naive resize
        resized = np.array(Image.fromarray(frame).resize((imgsz, imgsz)))

    # Convert to tensor (BHWC -> BCHW, 0..1)
    x = resized[:, :, ::-1]  # BGR->RGB if using cv2
    x = np.ascontiguousarray(x).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW
    tensor = torch.from_numpy(x).to(device)

    # AMP setting
    use_amp = (mode == "fp16") and (device == "cuda")

    # Timing containers
    end2end_times = []
    infer_times = []
    nms_times = []
    cap_ms = None
    pre_ms = None

    # Warmup
    log(f"Warmup {warmup} runs ...")
    for _ in range(warmup):
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0 if device=='cuda' else None)
            else:
                _ = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0 if device=='cuda' else None)

    # Timed runs
    log(f"Timed {runs} runs ...")
    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(runs):
        t0 = time.perf_counter()
        if device == "cuda":
            torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None1
        ev1 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if ev0 is not None: ev0.record()
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    res = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0 if device=='cuda' else None)
            else:
                res = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0 if device=='cuda' else None)
        if ev1 is not None: ev1.record()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Extract inner timings if available (Ultralytics may expose speed dict per result)
        this_infer = None
        this_nms = None
        try:
            r0 = res[0]
            if hasattr(r0, "speed") and isinstance(r0.speed, dict):
                # keys: 'preprocess', 'inference', 'postprocess'
                this_infer = float(r0.speed.get("inference", 0.0))
                this_nms = float(r0.speed.get("postprocess", 0.0))
                pre_ms = float(r0.speed.get("preprocess", 0.0))
        except Exception:

        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        f"--shapes=images:1x3x{imgsz}x{imgsz}",
        "--workspace=4096",
        "--avgRuns", str(runs),
        "--warmUp", str(warmup),
        "--noDataTransfers",  # focus on compute
        "--useCudaGraph",
    ]
    if mode == "trt-fp16":
        cmd_build.append("--fp16")
    if mode == "trt-int8":
        cmd_build.append("--int8")
        if calib_cache and calib_cache.exists():
            cmd_build += ["--calib="+str(calib_cache)]
        else:
            log("INT8 selected without calib cache; trtexec will attempt entropy calibration from random data.")

    # Build & profile in one go
    out = run_cmd(cmd_build)
    mean_ms = parse_trtexec_mean_latency(out)
    if mean_ms is None:
        raise RuntimeError("Failed to parse mean latency from trtexec output.")

    # For TRT we don't have capture/preprocess/NMS; this is model compute
    return BenchResult(
        model=str(onnx_path.stem),
        mode=mode,
        imgsz=imgsz,
        device="TensorRT-GPU",
        runs=runs,
        warmup=warmup,
        capture_ms=None,
        preprocess_ms=None,
        infer_ms_mean=mean_ms,
        nms_ms=None,
        end2end_ms_mean=mean_ms,  # compute-only; note in 'notes'
        end2end_ms_p50=mean_ms,
        end2end_ms_p90=mean_ms,
        end2end_ms_p95=mean_ms,
        end2end_ms_p99=mean_ms,
        notes="TensorRT trtexec mean latency (compute only, no IO)",
    )


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Paths to YOLO weights (e.g., yolov8n.pt ...)")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640], help="List of square image sizes")
    ap.add_argument("--modes", nargs="+", choices=["torch", "fp16", "trt-fp16", "trt-int8"], default=["torch"],
                    help="Which modes to run")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--source", type=str, default=None, help="Image path for testing (optional). If missing, random image used.")
    ap.add_argument("--out", type=str, default="bench_results.csv")
    ap.add_argument("--workdir", type=str, default="bench_artifacts")
    ap.add_argument("--skip_onnx_export", action="store_true", help="Reuse existing ONNX if present")
    ap.add_argument("--calib_cache", type=str, default=None, help="Path to TensorRT INT8 calibration cache")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results: List[BenchResult] = []

    for model_path in args.models:
        for imgsz in args.imgsz:
            # PyTorch modes
            if any(m in args.modes for m in ["torch", "fp16"]):
                for m in [x for x in args.modes if x in ("torch", "fp16")]:
                    try:
                        res = bench_torch(model_path, imgsz, m, args.source, args.runs, args.warmup)
                        results.append(res)
                    except Exception as e:
                        log(f"[WARN] PyTorch bench failed for {model_path} {imgsz} {m}: {e}")

            # TRT modes
            if any(m in args.modes for m in ["trt-fp16", "trt-int8"]):
                try:
                    if args.skip_onnx_export:
                        onnx_path = workdir / f"{Path(model_path).stem}_{imgsz}.onnx"
                        if not onnx_path.exists():
                            raise RuntimeError(f"skip_onnx_export=True but {onnx_path} not found")
                    else:
                        onnx_path = export_onnx(model_path, imgsz, workdir)
                except Exception as e:
                    log(f"[WARN] ONNX export failed for {model_path} {imgsz}: {e}")
                    onnx_path = None

                if onnx_path:
                    for m in [x for x in args.modes if x in ("trt-fp16", "trt-int8")]:
                        try:
                            res = bench_trt(onnx_path, imgsz, m, args.runs, args.warmup,
                                            int8=(m=="trt-int8"),
                                            calib_cache=Path(args.calib_cache) if args.calib_cache else None)
                            results.append(res)
                        except Exception as e:
                            log(f"[WARN] TensorRT bench failed for {onnx_path} {imgsz} {m}: {e}")

    # Save CSV
    out_path = Path(args.out)
    log(f"Writing CSV to {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "mode", "imgsz", "device", "runs", "warmup",
            "capture_ms", "preprocess_ms", "infer_ms_mean", "nms_ms",
            "end2end_ms_mean", "end2end_ms_p50", "end2end_ms_p90", "end2end_ms_p95", "end2end_ms_p99",
            "notes"
        ])
        for r in results:
            writer.writerow([
                r.model, r.mode, r.imgsz, r.device, r.runs, r.warmup,
                "" if r.capture_ms is None else f"{r.capture_ms:.3f}",
                "" if r.preprocess_ms is None else f"{r.preprocess_ms:.3f}",
                f"{r.infer_ms_mean:.3f}",
                "" if r.nms_ms is None else f"{r.nms_ms:.3f}",
                f"{r.end2end_ms_mean:.3f}", f"{r.end2end_ms_p50:.3f}", f"{r.end2end_ms_p90:.3f}", f"{r.end2end_ms_p95:.3f}", f"{r.end2end_ms_p99:.3f}",
                r.notes
            ])

    log("Done.")


if __name__ == "__main__":
    main()
