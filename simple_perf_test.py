#!/usr/bin/env python3
"""
简单的性能对比测试
"""
import time
import numpy as np
from roi_canvas_demo import ROICanvasDemo

def simple_test():
    """简单测试"""
    print("🚀 简单性能对比测试")
    print("=" * 40)
    
    # 创建演示实例
    demo = ROICanvasDemo(['front', 'left', 'right', 'rear'], img_size=(640, 480))
    
    # 创建测试帧
    frames = {}
    for cam_name in ['front', 'left', 'right', 'rear']:
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        frames[cam_name] = frame
    
    print("\n🖼️  测试画布处理方法:")
    
    # 临时启用画布处理
    from roi_canvas_demo import USE_BATCH_PROCESSING
    original_batch_setting = USE_BATCH_PROCESSING
    
    # 测试画布处理
    import roi_canvas_demo
    roi_canvas_demo.USE_BATCH_PROCESSING = False
    
    canvas_times = []
    for i in range(3):
        start = time.perf_counter()
        results = demo.step(frames)
        elapsed = (time.perf_counter() - start) * 1000
        canvas_times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.2f} ms")
    
    canvas_avg = np.mean(canvas_times)
    print(f"  画布处理平均: {canvas_avg:.2f} ms")
    
    print("\n📦 测试批量处理方法:")
    
    # 测试批量处理
    roi_canvas_demo.USE_BATCH_PROCESSING = True
    
    batch_times = []
    for i in range(3):
        try:
            start = time.perf_counter()
            results = demo.step(frames)
            elapsed = (time.perf_counter() - start) * 1000
            batch_times.append(elapsed)
            print(f"  运行 {i+1}: {elapsed:.2f} ms")
        except Exception as e:
            print(f"  运行 {i+1}: 失败 - {e}")
    
    if batch_times:
        batch_avg = np.mean(batch_times)
        print(f"  批量处理平均: {batch_avg:.2f} ms")
        
        # 对比结果
        speedup = canvas_avg / batch_avg
        print(f"\n📊 对比结果:")
        print(f"  画布处理: {canvas_avg:.2f} ms")
        print(f"  批量处理: {batch_avg:.2f} ms")
        print(f"  性能提升: {speedup:.2f}x")
        
        if speedup > 1.2:
            print(f"  🏆 批量处理更快")
        elif speedup > 0.8:
            print(f"  ⚖️  性能相近")
        else:
            print(f"  🏆 画布处理更快")
    else:
        print(f"  批量处理失败")
    
    # 恢复原始设置
    roi_canvas_demo.USE_BATCH_PROCESSING = original_batch_setting
    
    print(f"\n✅ 测试完成!")

if __name__ == '__main__':
    simple_test()
