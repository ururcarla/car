#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtdetrv3_trtexec_bench.py
-------------------------
Benchmark compute-only latency for official RT-DETRv3 ONNX models (different backbones) using TensorRT's trtexec.
You must export ONNX first from the official repo (e.g., R18/R34/R50/R101).

Requirements:
  - TensorRT installed and `trtexec` available in PATH

Example:
  python rtdetrv3_trtexec_bench.py --onnx rtdetrv3_r18.onnx rtdetrv3_r50.onnx     --imgsz 640 --modes trt-fp16 --runs 300 --warmup 100 --out rtdetrv3_trt.csv
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_trtexec() -> str:
    exe = shutil.which('trtexec')
    if not exe:
        raise RuntimeError('`trtexec` not found in PATH. Install TensorRT and add it to PATH.')
    return exe


def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        out_lines.append(line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'Command failed: {" ".join(cmd)}')
    return ''.join(out_lines)


def parse_mean_ms(out: str):
    m = re.search(r'mean:\s*([\d\.]+)\s*ms', out, re.IGNORECASE)
    return float(m.group(1)) if m else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', nargs='+', required=True, help='RT-DETRv3 ONNX files (different backbones)')
    ap.add_argument('--imgsz', type=int, default=640, help='Assumed input shape 1x3xH x W (square)')
    ap.add_argument('--modes', nargs='+', choices=['trt-fp16','trt-int8'], default=['trt-fp16'])
    ap.add_argument('--runs', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--input_name', type=str, default='image', help='ONNX input tensor name (default Paddle export: image)')
    ap.add_argument('--out', type=str, default='rtdetrv3_trt.csv')
    args = ap.parse_args()

    trtexec = ensure_trtexec()
    rows = []

    for onnx in args.onnx:
        onnx = Path(onnx)
        if not onnx.exists():
            print(f'[WARN] missing {onnx}, skip.'); continue
        for mode in args.modes:
            cmd = [
                trtexec,
                f'--onnx={str(onnx)}',
                f'--saveEngine={str(onnx.with_suffix("."+mode+".plan"))}',
                f'--shapes={args.input_name}:1x3x{args.imgsz}x{args.imgsz}',
                '--workspace=4096', '--avgRuns', str(args.runs), '--warmUp', str(args.warmup),
                '--noDataTransfers', '--useCudaGraph'
            ]
            if mode == 'trt-fp16': cmd.append('--fp16')
            if mode == 'trt-int8': cmd.append('--int8')  # requires proper calibration/quantization ranges

            print('[bench]', ' '.join(cmd))
            out = run_cmd(cmd)
            mean_ms = parse_mean_ms(out)
            rows.append([onnx.stem, mode, args.imgsz, mean_ms, args.runs, args.warmup, 'trtexec mean (compute-only)'])

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model','mode','imgsz','infer_ms_mean','runs','warmup','notes'])
        for r in rows:
            w.writerow(r)

    print(f'[bench] wrote CSV: {args.out}')


if __name__ == '__main__':
    main()
