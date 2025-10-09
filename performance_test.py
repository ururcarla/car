#!/usr/bin/env python3
"""
性能对比测试：打包ROI到大图推理 vs 批量推理ROI
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
    ROIMapping,
    USE_BATCH_PROCESSING,
    USE_BEV_FUSION,
    ROI_BATCH_SIZE,
    CANVAS_W,
    CANVAS_H
)

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.demo = ROICanvasDemo(['front', 'left', 'right', 'rear'], img_size=(640, 480))
        
    def create_test_data(self, num_rois: int = 8) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, int, int, int, int]]]:
        """创建测试数据"""
        # 创建测试帧
        frames = {}
        for cam_name in ['front', 'left', 'right', 'rear']:
            # 创建随机测试图像
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            # 添加一些纹理
            for i in range(0, 480, 30):
                cv2.line(frame, (0, i), (640, i), (100, 100, 100), 1)
            frames[cam_name] = frame
        
        # 创建模拟ROI坐标
        roi_coords = []
        cameras = ['front', 'left', 'right', 'rear']
        
        for i in range(num_rois):
            cam_name = cameras[i % len(cameras)]
            # 随机生成ROI坐标
            x = np.random.randint(50, 500)
            y = np.random.randint(50, 350)
            w = np.random.randint(80, 150)
            h = np.random.randint(80, 150)
            roi_coords.append((cam_name, x, y, w, h))
        
        return frames, roi_coords
    
    def test_canvas_processing(self, frames: Dict[str, np.ndarray], 
                              roi_coords: List[Tuple[str, int, int, int, int]], 
                              num_runs: int = 10) -> Dict[str, float]:
        """测试画布处理性能"""
        print(f"\n=== 测试画布处理性能 ({num_runs} 次运行) ===")
        
        times = []
        roi_extraction_times = []
        canvas_packing_times = []
        inference_times = []
        remapping_times = []
        
        for run in range(num_runs):
            # 1. 提取ROI
            start = time.perf_counter()
            roi_items = []
            for cam_name, x, y, w, h in roi_coords:
                if cam_name in frames:
                    frame = frames[cam_name]
                    roi = frame[y:y+h, x:x+w]
                    roi_items.append((cam_name, frame, (x, y, w, h)))
            roi_extraction_time = (time.perf_counter() - start) * 1000
            roi_extraction_times.append(roi_extraction_time)
            
            # 2. 打包到画布
            start = time.perf_counter()
            canvas, mappings = pack_rois_fixed_canvas(roi_items, CANVAS_W, CANVAS_H)
            canvas_packing_time = (time.perf_counter() - start) * 1000
            canvas_packing_times.append(canvas_packing_time)
            
            # 3. 画布推理
            start = time.perf_counter()
            res = self.demo.model.predict(canvas, verbose=False)[0]
            inference_time = (time.perf_counter() - start) * 1000
            inference_times.append(inference_time)
            
            # 4. 结果映射
            start = time.perf_counter()
            if res.boxes.shape[0] == 0:
                canvas_dets = np.empty((0, 6), dtype=np.float32)
            else:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
                cls = res.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
                canvas_dets = np.hstack((xyxy, conf, cls))
            
            buckets = assign_dets_to_tiles(canvas_dets, mappings)
            remapped = remap_tile_dets_to_original(buckets, mappings)
            remapping_time = (time.perf_counter() - start) * 1000
            remapping_times.append(remapping_time)
            
            total_time = roi_extraction_time + canvas_packing_time + inference_time + remapping_time
            times.append(total_time)
            
            if run == 0:  # 第一次运行显示详细信息
                print(f"  ROI提取时间: {roi_extraction_time:.2f} ms")
                print(f"  画布打包时间: {canvas_packing_time:.2f} ms")
                print(f"  推理时间: {inference_time:.2f} ms")
                print(f"  结果映射时间: {remapping_time:.2f} ms")
                print(f"  总时间: {total_time:.2f} ms")
        
        # 计算统计信息
        stats = {
            'total_avg': np.mean(times),
            'total_std': np.std(times),
            'total_min': np.min(times),
            'total_max': np.max(times),
            'roi_extraction_avg': np.mean(roi_extraction_times),
            'canvas_packing_avg': np.mean(canvas_packing_times),
            'inference_avg': np.mean(inference_times),
            'remapping_avg': np.mean(remapping_times)
        }
        
        print(f"\n画布处理统计 (ROI数量: {len(roi_coords)}):")
        print(f"  总时间: {stats['total_avg']:.2f} ± {stats['total_std']:.2f} ms")
        print(f"  最快: {stats['total_min']:.2f} ms, 最慢: {stats['total_max']:.2f} ms")
        print(f"  ROI提取: {stats['roi_extraction_avg']:.2f} ms")
        print(f"  画布打包: {stats['canvas_packing_avg']:.2f} ms")
        print(f"  推理: {stats['inference_avg']:.2f} ms")
        print(f"  结果映射: {stats['remapping_avg']:.2f} ms")
        
        return stats
    
    def test_batch_processing(self, frames: Dict[str, np.ndarray], 
                             roi_coords: List[Tuple[str, int, int, int, int]], 
                             num_runs: int = 10) -> Dict[str, float]:
        """测试批量处理性能"""
        print(f"\n=== 测试批量处理性能 ({num_runs} 次运行) ===")
        
        times = []
        roi_extraction_times = []
        roi_resize_times = []
        inference_times = []
        remapping_times = []
        
        for run in range(num_runs):
            # 1. 提取和调整ROI
            start = time.perf_counter()
            roi_items = extract_rois_from_frames(frames, roi_coords)
            roi_extraction_time = (time.perf_counter() - start) * 1000
            roi_extraction_times.append(roi_extraction_time)
            
            if not roi_items:
                print("  警告: 没有有效的ROI")
                continue
            
            # 2. 批量推理
            start = time.perf_counter()
            roi_batch = [item[1] for item in roi_items]
            
            # 过滤有效ROI
            valid_rois = []
            for roi in roi_batch:
                if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                    valid_rois.append(roi)
            
            if not valid_rois:
                print("  警告: 没有有效的ROI进行推理")
                continue
            
            batch_array = np.stack(valid_rois, axis=0)
            batch_detections = batch_inference_rois(self.demo.model, roi_batch)
            inference_time = (time.perf_counter() - start) * 1000
            inference_times.append(inference_time)
            
            # 3. 结果映射
            start = time.perf_counter()
            remapped_dets = remap_batch_detections_to_original(batch_detections, roi_items)
            remapping_time = (time.perf_counter() - start) * 1000
            remapping_times.append(remapping_time)
            
            total_time = roi_extraction_time + inference_time + remapping_time
            times.append(total_time)
            
            if run == 0:  # 第一次运行显示详细信息
                print(f"  ROI提取调整时间: {roi_extraction_time:.2f} ms")
                print(f"  批量推理时间: {inference_time:.2f} ms")
                print(f"  结果映射时间: {remapping_time:.2f} ms")
                print(f"  总时间: {total_time:.2f} ms")
        
        if not times:
            print("  错误: 没有成功完成任何运行")
            return {}
        
        # 计算统计信息
        stats = {
            'total_avg': np.mean(times),
            'total_std': np.std(times),
            'total_min': np.min(times),
            'total_max': np.max(times),
            'roi_extraction_avg': np.mean(roi_extraction_times),
            'inference_avg': np.mean(inference_times),
            'remapping_avg': np.mean(remapping_times)
        }
        
        print(f"\n批量处理统计 (ROI数量: {len(roi_coords)}):")
        print(f"  总时间: {stats['total_avg']:.2f} ± {stats['total_std']:.2f} ms")
        print(f"  最快: {stats['total_min']:.2f} ms, 最慢: {stats['total_max']:.2f} ms")
        print(f"  ROI提取调整: {stats['roi_extraction_avg']:.2f} ms")
        print(f"  批量推理: {stats['inference_avg']:.2f} ms")
        print(f"  结果映射: {stats['remapping_avg']:.2f} ms")
        
        return stats
    
    def compare_performance(self, canvas_stats: Dict[str, float], 
                          batch_stats: Dict[str, float], 
                          roi_count: int):
        """对比性能结果"""
        print(f"\n{'='*60}")
        print(f"性能对比结果 (ROI数量: {roi_count})")
        print(f"{'='*60}")
        
        if not canvas_stats or not batch_stats:
            print("错误: 无法进行对比，某个测试失败")
            return
        
        # 总时间对比
        canvas_total = canvas_stats['total_avg']
        batch_total = batch_stats['total_avg']
        speedup = canvas_total / batch_total if batch_total > 0 else float('inf')
        
        print(f"\n📊 总处理时间对比:")
        print(f"  画布处理: {canvas_total:.2f} ± {canvas_stats['total_std']:.2f} ms")
        print(f"  批量处理: {batch_total:.2f} ± {batch_stats['total_std']:.2f} ms")
        print(f"  性能提升: {speedup:.2f}x {'(批量更快)' if speedup > 1 else '(画布更快)'}")
        
        # 各阶段时间对比
        print(f"\n🔍 各阶段时间对比:")
        print(f"  ROI处理:")
        print(f"    画布: {canvas_stats['roi_extraction_avg']:.2f} ms")
        print(f"    批量: {batch_stats['roi_extraction_avg']:.2f} ms")
        
        print(f"  推理:")
        print(f"    画布: {canvas_stats['inference_avg']:.2f} ms")
        print(f"    批量: {batch_stats['inference_avg']:.2f} ms")
        
        print(f"  结果映射:")
        print(f"    画布: {canvas_stats['remapping_avg']:.2f} ms")
        print(f"    批量: {batch_stats['remapping_avg']:.2f} ms")
        
        # 画布处理额外开销
        canvas_extra = canvas_stats['canvas_packing_avg']
        print(f"  画布打包开销: {canvas_extra:.2f} ms")
        
        # 效率分析
        print(f"\n💡 效率分析:")
        if speedup > 1.2:
            print(f"  ✅ 批量处理明显更快 ({speedup:.2f}x)")
        elif speedup > 0.8:
            print(f"  ⚖️  两种方法性能相近 ({speedup:.2f}x)")
        else:
            print(f"  ❌ 画布处理更快 ({speedup:.2f}x)")
        
        # 内存使用分析
        print(f"\n💾 内存使用分析:")
        canvas_memory = CANVAS_W * CANVAS_H * 3  # 画布大小
        batch_memory = len(roi_coords) * ROI_BATCH_SIZE * ROI_BATCH_SIZE * 3  # 批量大小
        print(f"  画布处理内存: {canvas_memory / 1024 / 1024:.2f} MB")
        print(f"  批量处理内存: {batch_memory / 1024 / 1024:.2f} MB")
        
        if batch_memory > canvas_memory:
            print(f"  批量处理使用更多内存 ({batch_memory/canvas_memory:.2f}x)")
        else:
            print(f"  画布处理使用更多内存 ({canvas_memory/batch_memory:.2f}x)")
    
    def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("🚀 开始性能对比测试")
        print("="*60)
        
        # 测试不同的ROI数量
        roi_counts = [4, 8, 12, 16]
        num_runs = 5  # 减少运行次数以节省时间
        
        all_results = []
        
        for roi_count in roi_counts:
            print(f"\n🔬 测试 ROI数量: {roi_count}")
            print("-" * 40)
            
            # 创建测试数据
            frames, roi_coords = self.create_test_data(roi_count)
            
            # 测试画布处理
            canvas_stats = self.test_canvas_processing(frames, roi_coords, num_runs)
            
            # 测试批量处理
            batch_stats = self.test_batch_processing(frames, roi_coords, num_runs)
            
            # 对比结果
            self.compare_performance(canvas_stats, batch_stats, roi_count)
            
            all_results.append({
                'roi_count': roi_count,
                'canvas_stats': canvas_stats,
                'batch_stats': batch_stats
            })
        
        # 生成总结报告
        self.generate_summary_report(all_results)
    
    def generate_summary_report(self, results: List[Dict]):
        """生成总结报告"""
        print(f"\n{'='*60}")
        print("📋 性能测试总结报告")
        print(f"{'='*60}")
        
        print(f"\n配置信息:")
        print(f"  画布大小: {CANVAS_W}x{CANVAS_H}")
        print(f"  ROI批量大小: {ROI_BATCH_SIZE}x{ROI_BATCH_SIZE}")
        print(f"  测试运行次数: 5")
        
        print(f"\n📊 不同ROI数量下的性能对比:")
        print(f"{'ROI数量':<8} {'画布(ms)':<12} {'批量(ms)':<12} {'提升倍数':<10} {'推荐方法'}")
        print("-" * 60)
        
        for result in results:
            roi_count = result['roi_count']
            canvas_avg = result['canvas_stats']['total_avg']
            batch_avg = result['batch_stats']['total_avg']
            speedup = canvas_avg / batch_avg if batch_avg > 0 else 0
            
            if speedup > 1.2:
                recommendation = "批量处理"
            elif speedup > 0.8:
                recommendation = "相近"
            else:
                recommendation = "画布处理"
            
            print(f"{roi_count:<8} {canvas_avg:<12.2f} {batch_avg:<12.2f} {speedup:<10.2f} {recommendation}")
        
        print(f"\n💡 建议:")
        print(f"  - 少量ROI (<8): 画布处理可能更稳定")
        print(f"  - 中等ROI (8-12): 根据具体场景选择")
        print(f"  - 大量ROI (>12): 批量处理可能更高效")
        print(f"  - 内存限制: 考虑使用画布处理")
        print(f"  - 稳定性优先: 推荐画布处理")

def main():
    """主函数"""
    print("🎯 ROI处理性能对比测试")
    print("="*60)
    
    # 创建测试器
    tester = PerformanceTester()
    
    try:
        # 运行综合测试
        tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 测试完成!")

if __name__ == '__main__':
    main()
