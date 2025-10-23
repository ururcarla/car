#!/usr/bin/env python3
"""
测试修复后的代码
"""
import numpy as np
import cv2
from roi_canvas_demo import ROICanvasDemo

def test_fixed_code():
    """测试修复后的代码"""
    print("=== 测试修复后的代码 ===")
    
    # 创建ROI演示实例
    cam_names = ['front', 'left', 'right', 'rear']
    demo = ROICanvasDemo(cam_names, img_size=(640, 480))
    
    # 创建测试帧
    test_frames = {}
    for cam_name in cam_names:
        # 创建随机测试图像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_frames[cam_name] = frame
    
    print("测试帧创建完成")
    
    # 测试处理
    try:
        print("开始处理测试帧...")
        results = demo.step(test_frames)
        print("处理成功！")
        
        for cam_name, dets in results.items():
            print(f"  {cam_name}: {dets.shape[0]} 个检测")
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("测试完成")

if __name__ == '__main__':
    test_fixed_code()
