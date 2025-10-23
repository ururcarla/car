#!/usr/bin/env python3
"""
测试批量ROI处理功能
"""
import numpy as np
import cv2
import time
from roi_canvas_demo import ROICanvasDemo, USE_BATCH_PROCESSING, ROI_BATCH_SIZE

def create_test_frames():
    """创建测试用的模拟帧数据"""
    frames = {}
    
    # 创建三个模拟摄像头帧
    for cam_name in ['front', 'left', 'right']:
        # 创建640x480的测试图像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些模拟的车辆矩形
        if cam_name == 'front':
            cv2.rectangle(frame, (100, 150), (200, 250), (0, 255, 0), -1)  # 绿色车辆
            cv2.rectangle(frame, (400, 200), (500, 300), (255, 0, 0), -1)  # 蓝色车辆
        elif cam_name == 'left':
            cv2.rectangle(frame, (50, 100), (150, 200), (0, 0, 255), -1)   # 红色车辆
        else:  # right
            cv2.rectangle(frame, (300, 180), (400, 280), (255, 255, 0), -1) # 黄色车辆
        
        frames[cam_name] = frame
    
    return frames

def test_batch_processing():
    """测试批量处理功能"""
    print("=== 测试批量ROI处理功能 ===")
    
    # 创建ROI演示实例
    cam_names = ['front', 'left', 'right']
    demo = ROICanvasDemo(cam_names, img_size=(640, 480))
    
    print(f"批量处理模式: {'启用' if USE_BATCH_PROCESSING else '禁用'}")
    print(f"ROI批量大小: {ROI_BATCH_SIZE}x{ROI_BATCH_SIZE}")
    print(f"最大批次大小: {demo.__dict__.get('MAX_BATCH_SIZE', 16)}")
    
    # 创建测试帧
    test_frames = create_test_frames()
    
    # 显示测试帧
    for cam_name, frame in test_frames.items():
        cv2.imshow(f'Test Frame: {cam_name}', frame)
    
    print("\n=== 开始测试 ===")
    
    # 测试关键帧处理（初始化跟踪器）
    print("1. 处理关键帧...")
    start_time = time.perf_counter()
    keyframe_results = demo.process_keyframe(test_frames)
    keyframe_time = (time.perf_counter() - start_time) * 1000
    print(f"   关键帧处理时间: {keyframe_time:.2f} ms")
    
    for cam_name, dets in keyframe_results.items():
        print(f"   {cam_name}: 检测到 {dets.shape[0]} 个目标")
    
    # 测试中间帧处理（批量ROI处理）
    print("\n2. 处理中间帧（批量ROI）...")
    start_time = time.perf_counter()
    batch_results = demo.process_intermediate(test_frames)
    batch_time = (time.perf_counter() - start_time) * 1000
    print(f"   批量处理时间: {batch_time:.2f} ms")
    
    for cam_name, dets in batch_results.items():
        print(f"   {cam_name}: 检测到 {dets.shape[0]} 个目标")
    
    # 多次测试以评估性能
    print("\n3. 性能测试（10次批量处理）...")
    times = []
    for i in range(10):
        start_time = time.perf_counter()
        _ = demo.process_intermediate(test_frames)
        process_time = (time.perf_counter() - start_time) * 1000
        times.append(process_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"   平均处理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   最快处理时间: {min_time:.2f} ms")
    print(f"   最慢处理时间: {max_time:.2f} ms")
    
    print("\n=== 测试完成 ===")
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_roi_resize():
    """测试ROI调整大小功能"""
    print("=== 测试ROI调整大小功能 ===")
    
    from roi_canvas_demo import resize_roi_to_batch_size
    
    # 创建不同尺寸的测试ROI
    test_rois = [
        ("正方形", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
        ("宽矩形", np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)),
        ("高矩形", np.random.randint(0, 255, (200, 50, 3), dtype=np.uint8)),
        ("大图像", np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)),
    ]
    
    for name, roi in test_rois:
        original_size = roi.shape[:2]
        resized_roi = resize_roi_to_batch_size(roi, ROI_BATCH_SIZE)
        resized_size = resized_roi.shape[:2]
        
        print(f"{name}: {original_size} -> {resized_size}")
        
        # 显示调整前后的对比
        cv2.imshow(f'Original {name}', roi)
        cv2.imshow(f'Resized {name}', resized_roi)
    
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("批量ROI处理测试")
    print("================")
    
    # 测试ROI调整大小
    test_roi_resize()
    
    # 测试批量处理
    test_batch_processing()
