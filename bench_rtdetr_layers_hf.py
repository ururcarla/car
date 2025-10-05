#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_rtdetr_layers_hf.py
-------------------------
Benchmark compute-only latency of RT-DETR (HuggingFace Transformers) by varying the number of
Transformer decoder layers (and optionally encoder layers). Uses synthetic input (1x3xH x W), batch=1.
Reports mean / p50 / p90 / p95 / p99 per configuration.

Requirements:
  pip install torch transformers accelerate

Example:
  python bench_rtdetr_layers_hf.py --imgsz 640 --layers 1 2 3 6 --runs 300 --warmup 100 --mode fp16 --out rtdetr_layers_hf.csv
"""
import argparse
import csv
import time
from typing import List

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import RTDetrConfig, RTDetrForObjectDetection
except Exception:
    RTDetrConfig = None
    RTDetrForObjectDetection = None


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float('nan')
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(vals[f])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', type=int, default=640, help='Square input size H=W')
    ap.add_argument('--layers', nargs='+', type=int, default=[1, 3, 6], help='Decoder layers to test')
    ap.add_argument('--enc_layers', type=int, default=1, help='Encoder layers (default 1 as in RT-DETR)')
    ap.add_argument('--runs', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--mode', choices=['fp32', 'fp16'], default='fp32')
    ap.add_argument('--out', type=str, default='rtdetr_layers_hf.csv')
    args = ap.parse_args()

    if torch is None or RTDetrConfig is None or RTDetrForObjectDetection is None:
        raise RuntimeError('Please install torch and transformers: pip install torch transformers accelerate')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and args.mode == 'fp16':
        print('[WARN] CUDA not available; switching to fp32.')
        args.mode = 'fp32'

    # synthetic input
    x = torch.randint(0, 255, (1, 3, args.imgsz, args.imgsz), dtype=torch.uint8).to(torch.float32) / 255.0
    x = x.to(device)

    rows = []
    for dec_layers in args.layers:
        cfg = RTDetrConfig(
            encoder_layers=args.enc_layers,
            decoder_layers=dec_layers,
            eval_size=(args.imgsz, args.imgsz),
        )
        model = RTDetrForObjectDetection(cfg).to(device).eval()
        use_amp = (args.mode == 'fp16')

        # warmup
        if device == 'cuda':
            torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(args.warmup):
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        _ = model(pixel_values=x)
                else:
                    _ = model(pixel_values=x)
        if device == 'cuda':
            torch.cuda.synchronize()

        # timed
        times = []
        with torch.no_grad():
            for _ in range(args.runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        _ = model(pixel_values=x)
                else:
                    _ = model(pixel_values=x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        p50 = percentile(times, 0.5)
        p90 = percentile(times, 0.9)
        p95 = percentile(times, 0.95)
        p99 = percentile(times, 0.99)
        devname = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'

        rows.append({
            'model': f'HF-RTDETR-dec{dec_layers}-enc{args.enc_layers}',
            'mode': args.mode,
            'imgsz': args.imgsz,
            'device': devname,
            'runs': args.runs,
            'warmup': args.warmup,
            'infer_ms_mean': f'{mean_ms:.3f}',
            'end2end_ms_p50': f'{p50:.3f}',
            'end2end_ms_p90': f'{p90:.3f}',
            'end2end_ms_p95': f'{p95:.3f}',
            'end2end_ms_p99': f'{p99:.3f}',
            'notes': 'HF RT-DETR forward (compute-only); random init'
        })

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'model','mode','imgsz','device','runs','warmup',
            'infer_ms_mean','end2end_ms_p50','end2end_ms_p90','end2end_ms_p95','end2end_ms_p99','notes'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'[bench] wrote CSV: {args.out}')


if __name__ == '__main__':
    main()
