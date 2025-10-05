#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
det_latency_bench_plus_v3.py
----------------------------
Patch highlights vs previous versions:
 - EfficientDet: pass image_size=(imgsz, imgsz) instead of int to avoid "'int' object is not subscriptable".
 - EfficientDet: keeps anchors/FPN aligned to the requested size to avoid stack-size mismatch.
 - Uses torch.amp.autocast('cuda', dtype=...) to silence deprecation warnings.
 - CSV schema identical to prior scripts.

Families supported:
 - "ultra": Ultralytics YOLO / RT-DETR (.pt via ultralytics.YOLO), supports PyTorch + TensorRT
 - "effdet": EfficientDet (tf_efficientdet_d0..d7 via effdet), PyTorch only

Example:
python det_latency_bench_plus_v3.py \
  --models yolov8n.pt rtdetr-l.pt tf_efficientdet_d0::effdet \
  --imgsz 512 640 --modes torch fp16 trt-fp16 --runs 200 --warmup 50 --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# Optional deps
try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

# Ultralytics (YOLO / RT-DETR)
try:
    from ultralytics import YOLO as UL_YOLO
except Exception:
    UL_YOLO = None

# EfficientDet
try:
    from effdet import create_model as EFF_CREATE
    from effdet.bench import DetBenchPredict as EFF_BENCH
except Exception:
    EFF_CREATE = None
    EFF_BENCH = None


def log(msg: str):
    print(f"[bench] {msg}", flush=True)


def ensure_cuda():
    if torch is None:
        raise RuntimeError("PyTorch not available. Please install torch.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Please ensure an NVIDIA GPU + drivers + CUDA are installed.")


def ensure_trtexec() -> str:
    exe = shutil.which("trtexec")
    if not exe:
        raise RuntimeError("`trtexec` not found in PATH. Install TensorRT and ensure trtexec is on PATH.")
    return exe


@dataclass
class BenchResult:
    model: str
    mode: str
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


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(vals[f])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


def parse_model_spec(spec: str) -> Tuple[str, str]:
    if "::" in spec:
        name, fam = spec.split("::", 1)
        return name, fam.lower()
    return spec, "auto"


def detect_family(name: str, fam_hint: str) -> str:
    if fam_hint != "auto":
        return fam_hint
    lower = name.lower()
    if lower.endswith(".pt"):
        return "ultra"
    if lower.startswith("tf_efficientdet") or "efficientdet" in lower:
        return "effdet"
    if lower.endswith(".pth") and "eff" in lower:
        return "effdet"
    return "ultra"


# ---------------- Ultralytics family ----------------

def bench_ultra_torch(model_path: str, imgsz: int, mode: str, runs: int, warmup: int) -> BenchResult:
    if UL_YOLO is None:
        raise RuntimeError("Ultralytics not installed. `pip install ultralytics`")
    ensure_cuda()
    device = "cuda"
    model = UL_YOLO(model_path)
    model.to(device)

    # Synthetic input (RGB float32 0..1)
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    if cv2 is not None:
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    x = img[:, :, ::-1]  # BGR->RGB
    x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...].copy()
    tensor = torch.from_numpy(x).to(device)

    use_amp = (mode == "fp16")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0)
            else:
                _ = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0)

    e2e = []
    infer, nms = [], []
    pre_ms = None
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    res = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0)
            else:
                res = model.predict(source=tensor, imgsz=imgsz, verbose=False, device=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e.append((t1 - t0) * 1000.0)
        try:
            r0 = res[0]
            sd = getattr(r0, "speed", None) or {}
            infer.append(float(sd.get("inference", np.nan)))
            nms.append(float(sd.get("postprocess", np.nan)))
            pre_ms = float(sd.get("preprocess", np.nan))
        except Exception:
            pass

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return BenchResult(
        model=str(Path(model_path).name),
        mode=mode,
        imgsz=imgsz,
        device=dev,
        runs=runs,
        warmup=warmup,
        capture_ms=None,
        preprocess_ms=pre_ms,
        infer_ms_mean=float(np.nanmean(infer)) if len(infer) else float("nan"),
        nms_ms=float(np.nanmean(nms)) if len(nms) else None,
        end2end_ms_mean=float(np.mean(e2e)),
        end2end_ms_p50=percentile(e2e, 0.5),
        end2end_ms_p90=percentile(e2e, 0.9),
        end2end_ms_p95=percentile(e2e, 0.95),
        end2end_ms_p99=percentile(e2e, 0.99),
        notes="Ultralytics .predict end-to-end (includes postprocess)",
    )


def export_ultra_onnx(model_path: str, imgsz: int, out_dir: Path) -> Path:
    if UL_YOLO is None:
        raise RuntimeError("Ultralytics not installed.")
    model = UL_YOLO(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = out_dir / f"{Path(model_path).stem}_{imgsz}.onnx"
    log(f"Exporting ONNX for {model_path} -> {onnx_out}")
    model.export(format="onnx", imgsz=imgsz, opset=12, dynamic=False, simplify=True, optimize=False)
    produced = Path(str(Path(model_path).with_suffix("")) + ".onnx")
    if produced.exists():
        produced.replace(onnx_out)
    elif onnx_out.exists():
        pass
    else:
        fallback = Path("yolo.onnx")
        if fallback.exists():
            fallback.replace(onnx_out)
    if not onnx_out.exists():
        raise RuntimeError("ONNX export failed.")
    return onnx_out


def bench_ultra_trt(onnx_path: Path, imgsz: int, mode: str, runs: int, warmup: int) -> BenchResult:
    trtexec = ensure_trtexec()
    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(onnx_path.with_suffix('.engine'))}",
        f"--shapes=images:1x3x{imgsz}x{imgsz}",
        "--workspace=4096",
        "--avgRuns", str(runs),
        "--warmUp", str(warmup),
        "--noDataTransfers",
        "--useCudaGraph",
    ]
    if mode == "trt-fp16":
        cmd.append("--fp16")
    elif mode == "trt-int8":
        cmd.append("--int8")

    log(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        out_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("trtexec failed")

    out = "".join(out_lines)
    m = re.search(r"mean:\s*([\d\.]+)\s*ms", out, re.IGNORECASE)
    mean_ms = float(m.group(1)) if m else float("nan")

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
        end2end_ms_mean=mean_ms,
        end2end_ms_p50=mean_ms,
        end2end_ms_p90=mean_ms,
        end2end_ms_p95=mean_ms,
        end2end_ms_p99=mean_ms,
        notes="TensorRT trtexec mean latency (compute only)",
    )


# ---------------- EfficientDet family ----------------

def bench_effdet_torch(model_name_or_ckpt: str, imgsz: int, mode: str, runs: int, warmup: int) -> BenchResult:
    if EFF_CREATE is None or EFF_BENCH is None or torch is None:
        raise RuntimeError("effdet and/or torch not installed. `pip install effdet torch`")
    ensure_cuda()
    device = "cuda"

    # Ensure tuple size for effdet internals
    image_size = (imgsz, imgsz)

    # Build model (pretrained head may be adapted if size differs)
    if Path(model_name_or_ckpt).exists():
        variant = "tf_efficientdet_d0"
        m = re.search(r"(tf_)?efficientdet_[dD](\d)", Path(model_name_or_ckpt).stem)
        if m:
            variant = f"tf_efficientdet_d{m.group(2)}"
        model = EFF_CREATE(variant, bench_task="predict", pretrained=False, image_size=image_size)
        ckpt = torch.load(model_name_or_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        model = EFF_CREATE(model_name_or_ckpt, bench_task="predict", pretrained=True, image_size=image_size)

    bench = EFF_BENCH(model)
    bench.eval().to(device)

    # Synthetic input
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    if cv2 is not None:
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    x = img[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...].copy()
    tensor = torch.from_numpy(x).to(device)

    use_amp = (mode == "fp16")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = bench(tensor)
            else:
                _ = bench(tensor)

    e2e, infer = [], []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = bench(tensor)
            else:
                _ = bench(tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e.append((t1 - t0) * 1000.0)
        infer.append(e2e[-1])

    dev = torch.cuda.get_device_name(0)
    return BenchResult(
        model=str(model_name_or_ckpt),
        mode=mode,
        imgsz=imgsz,
        device=dev,
        runs=runs,
        warmup=warmup,
        capture_ms=None,
        preprocess_ms=None,
        infer_ms_mean=float(np.mean(infer)),
        nms_ms=None,
        end2end_ms_mean=float(np.mean(e2e)),
        end2end_ms_p50=percentile(e2e, 0.5),
        end2end_ms_p90=percentile(e2e, 0.9),
        end2end_ms_p95=percentile(e2e, 0.95),
        end2end_ms_p99=percentile(e2e, 0.99),
        notes="EfficientDet DetBenchPredict forward (includes postprocess)",
    )


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model specs. Examples: yolov8n.pt | rtdetr-l.pt | tf_efficientdet_d0::effdet")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Square sizes: 320 512 640 ...")
    ap.add_argument("--modes", nargs="+", choices=["torch", "fp16", "trt-fp16", "trt-int8"], default=["torch"])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=str, default="bench_results_plus_v3.csv")
    ap.add_argument("--workdir", type=str, default="bench_plus_artifacts_v3")
    args = ap.parse_args()

    results: List[BenchResult] = []

    for spec in args.models:
        name, fam_hint = parse_model_spec(spec)
        family = detect_family(name, fam_hint)
        log(f"Model '{name}' resolved family='{family}'")

        for imgsz in args.imgsz:
            for mode in [m for m in args.modes if m in ("torch", "fp16")]:
                try:
                    if family == "ultra":
                        res = bench_ultra_torch(name, imgsz, mode, args.runs, args.warmup)
                    elif family == "effdet":
                        if imgsz % 32 != 0:
                            log(f"[INFO] EfficientDet prefers image sizes multiple of 32/128; got {imgsz}.")
                        res = bench_effdet_torch(name, imgsz, mode, args.runs, args.warmup)
                    else:
                        log(f"[WARN] Unknown family '{family}' for {name}, skipping torch mode.")
                        continue
                    results.append(res)
                except Exception as e:
                    log(f"[WARN] Torch bench failed for {name} imgsz={imgsz} mode={mode}: {e}")

            if family == "ultra":
                for mode in [m for m in args.modes if m in ("trt-fp16", "trt-int8")]:
                    try:
                        onnx = export_ultra_onnx(name, imgsz, Path(args.workdir))
                    except Exception as e:
                        log(f"[WARN] ONNX export failed for {name} imgsz={imgsz}: {e}")
                        onnx = None
                    if onnx is None:
                        continue
                    try:
                        res = bench_ultra_trt(onnx, imgsz, mode, args.runs, args.warmup)
                        results.append(res)
                    except Exception as e:
                        log(f"[WARN] TensorRT bench failed for {onnx} mode={mode}: {e}")
            else:
                if any(m in args.modes for m in ("trt-fp16", "trt-int8")):
                    log(f"[INFO] Skipping TensorRT for family '{family}' ({name}); not implemented in this script.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing CSV to {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "mode", "imgsz", "device", "runs", "warmup",
            "capture_ms", "preprocess_ms", "infer_ms_mean", "nms_ms",
            "end2end_ms_mean", "end2end_ms_p50", "end2end_ms_p90", "end2end_ms_p95", "end2end_ms_p99",
            "notes"
        ])
        for r in results:
            w.writerow([
                r.model, r.mode, r.imgsz, r.device, r.runs, r.warmup,
                "" if r.capture_ms is None else f"{r.capture_ms:.3f}",
                "" if r.preprocess_ms is None else f"{r.preprocess_ms:.3f}",
                f"{r.infer_ms_mean:.3f}",
                "" if r.nms_ms is None else f"{r.nms_ms:.3f}",
                f"{r.end2end_ms_mean:.3f}",
                f"{r.end2end_ms_p50:.3f}", f"{r.end2end_ms_p90:.3f}", f"{r.end2end_ms_p95:.3f}", f"{r.end2end_ms_p99:.3f}",
                r.notes
            ])
    log("Done.")


if __name__ == "__main__":
    main()
