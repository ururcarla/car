#!/usr/bin/env python3
"""
快速性能对比测试：打包ROI到大图推理 vs 批量推理ROI
"""
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple
from roi_canvas_demo import (
    ROICanvasDemo, 
    pack_rois_fixed_canvas, 
    batch_inference_rois,
    extract_rois_from_frames,
    remap_batch_detections_to_original,
    remap_tile_dets_to_original,
    assign_dets_to_tiles,
    ROI_BATCH_SIZE,
    CANVAS_W,
    CANVAS_H
)

def create_test_data(num_rois: int = 8):
    """创建测试数据"""
    # 创建测试帧
    frames = {}
    for cam_name in ['front', 'left', 'right', 'rear']:
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # 添加纹理
        for i in range(0, 480, 30):
            cv2.line(frame, (0, i), (640, i), (100, 100, 100), 1)
        frames[cam_name] = frame
    
    # 创建ROI坐标
    roi_coords = []
    cameras = ['front', 'left', 'right', 'rear']
    
    for i in range(num_rois):
        cam_name = cameras[i % len(cameras)]
        x = np.random.randint(50, 500)
        y = np.random.randint(50, 350)
        w = np.random.randint(80, 150)
        h = np.random.randint(80, 150)
        roi_coords.append((cam_name, x, y, w, h))
    
    return frames, roi_coords

def test_canvas_method(frames, roi_coords, model, num_runs=5):
    """测试画布方法"""
    print(f"🖼️  测试画布处理方法 ({num_runs} 次运行)")
    
    times = []
    
    for run in range(num_runs):
        start = time.perf_counter()
        
        # 1. 提取ROI
        roi_items = []
        for cam_name, x, y, w, h in roi_coords:
            if cam_name in frames:
                frame = frames[cam_name]
                roi = frame[y:y+h, x:x+w]
                roi_items.append((cam_name, frame, (x, y, w, h)))
        
        # 2. 打包到画布
        canvas, mappings = pack_rois_fixed_canvas(roi_items, CANVAS_W, CANVAS_H)
        
        # 3. 推理
        res = model.predict(canvas, verbose=False)[0]
        
        # 4. 结果映射
        if res.boxes.shape[0] == 0:
            canvas_dets = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
            cls = res.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
            canvas_dets = np.hstack((xyxy, conf, cls))
        
        buckets = assign_dets_to_tiles(canvas_dets, mappings)
        remapped = remap_tile_dets_to_original(buckets, mappings)
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if run == 0:
            print(f"  第一次运行: {elapsed:.2f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最快: {min_time:.2f} ms, 最慢: {max_time:.2f} ms")
    
    return {
        'avg': avg_time,
        'std': std_time,
        'min': min_time,
        'max': max_time
    }

def test_batch_method(frames, roi_coords, model, num_runs=5):
    """测试批量方法"""
    print(f"📦 测试批量处理方法 ({num_runs} 次运行)")
    
    times = []
    
    for run in range(num_runs):
        start = time.perf_counter()
        
        # 1. 提取和调整ROI
        roi_items = extract_rois_from_frames(frames, roi_coords)
        
        if not roi_items:
            print("  警告: 没有有效的ROI")
            continue
        
        # 2. 批量推理
        roi_batch = [item[1] for item in roi_items]
        
        # 过滤有效ROI
        valid_rois = []
        for roi in roi_batch:
            if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                valid_rois.append(roi)
        
        if not valid_rois:
            print("  警告: 没有有效的ROI进行推理")
            continue
        
        batch_detections = batch_inference_rois(model, roi_batch)
        
        # 3. 结果映射
        remapped_dets = remap_batch_detections_to_original(batch_detections, roi_items)
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if run == 0:
            print(f"  第一次运行: {elapsed:.2f} ms")
    
    if not times:
        print("  错误: 没有成功完成任何运行")
        return None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最快: {min_time:.2f} ms, 最慢: {max_time:.2f} ms")
    
    return {
        'avg': avg_time,
        'std': std_time,
        'min': min_time,
        'max': max_time
    }

def run_performance_test():
    """运行性能测试"""
    print("🚀 ROI处理性能对比测试")
    print("=" * 50)
    
    # 创建ROI演示实例
    demo = ROICanvasDemo(['front', 'left', 'right', 'rear'], img_size=(640, 480))
    
    # 测试不同的ROI数量
    roi_counts = [4, 8, 12, 16]
    num_runs = 3  # 快速测试，减少运行次数
    
    results = []
    
    for roi_count in roi_counts:
        print(f"\n🔬 测试 ROI数量: {roi_count}")
        print("-" * 30)
        
        # 创建测试数据
        frames, roi_coords = create_test_data(roi_count)
        
        # 测试画布方法
        canvas_stats = test_canvas_method(frames, roi_coords, demo.model, num_runs)
        
        # 测试批量方法
        batch_stats = test_batch_method(frames, roi_coords, demo.model, num_runs)
        
        if canvas_stats and batch_stats:
            # 计算性能提升
            speedup = canvas_stats['avg'] / batch_stats['avg']
            
            print(f"\n📊 对比结果:")
            print(f"  画布处理: {canvas_stats['avg']:.2f} ms")
            print(f"  批量处理: {batch_stats['avg']:.2f} ms")
            print(f"  性能提升: {speedup:.2f}x")
            
            if speedup > 1.2:
                winner = "批量处理 🏆"
            elif speedup > 0.8:
                winner = "性能相近 ⚖️"
            else:
                winner = "画布处理 🏆"
            
            print(f"  推荐: {winner}")
            
            results.append({
                'roi_count': roi_count,
                'canvas_avg': canvas_stats['avg'],
                'batch_avg': batch_stats['avg'],
                'speedup': speedup,
                'winner': winner
            })
    
    # 生成总结
    print(f"\n{'='*50}")
    print("📋 测试总结")
    print(f"{'='*50}")
    
    print(f"\n配置信息:")
    print(f"  画布大小: {CANVAS_W}x{CANVAS_H}")
    print(f"  ROI批量大小: {ROI_BATCH_SIZE}x{ROI_BATCH_SIZE}")
    print(f"  测试运行次数: {num_runs}")
    
    print(f"\n📊 性能对比表:")
    print(f"{'ROI数量':<8} {'画布(ms)':<10} {'批量(ms)':<10} {'提升倍数':<8} {'推荐方法'}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['roi_count']:<8} {result['canvas_avg']:<10.2f} {result['batch_avg']:<10.2f} {result['speedup']:<8.2f} {result['winner']}")
    
    # 内存使用分析
    print(f"\n💾 内存使用分析:")
    canvas_memory = CANVAS_W * CANVAS_H * 3 / 1024 / 1024  # MB
    max_batch_memory = max(results, key=lambda x: x['roi_count'])['roi_count'] * ROI_BATCH_SIZE * ROI_BATCH_SIZE * 3 / 1024 / 1024  # MB
    print(f"  画布处理内存: {canvas_memory:.2f} MB (固定)")
    print(f"  批量处理内存: {max_batch_memory:.2f} MB (最大ROI数量)")
    
    # 建议
    print(f"\n💡 使用建议:")
    print(f"  - 少量ROI (<8): 画布处理更稳定")
    print(f"  - 中等ROI (8-12): 根据具体需求选择")
    print(f"  - 大量ROI (>12): 批量处理可能更高效")
    print(f"  - 内存限制: 推荐画布处理")
    print(f"  - 稳定性优先: 推荐画布处理")

if __name__ == '__main__':
    run_performance_test()
