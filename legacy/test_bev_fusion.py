#!/usr/bin/env python3
"""
测试BEV多摄像头融合功能
"""
import numpy as np
import cv2
import time
from roi_canvas_demo import ROICanvasDemo, USE_BEV_FUSION, NUM_CAMERAS
from bev_transform import create_default_camera_params

def create_test_frames_with_detections():
    """创建包含模拟检测的测试帧"""
    frames = {}
    
    # 模拟4个摄像头的帧数据
    camera_configs = {
        'front': {
            'size': (640, 480),
            'detections': [
                # [x1, y1, x2, y2, conf, cls] - 前方车辆
                [200, 200, 300, 350, 0.85, 2],  # 车辆
                [400, 180, 500, 320, 0.92, 2],  # 车辆
            ]
        },
        'left': {
            'size': (640, 480),
            'detections': [
                # 左侧车辆（可能与前方车辆重叠）
                [150, 150, 250, 300, 0.78, 2],  # 车辆
            ]
        },
        'right': {
            'size': (640, 480),
            'detections': [
                # 右侧车辆
                [350, 160, 450, 310, 0.88, 2],  # 车辆
            ]
        },
        'rear': {
            'size': (640, 480),
            'detections': [
                # 后方车辆
                [250, 100, 350, 250, 0.76, 2],  # 车辆
            ]
        }
    }
    
    for cam_name, config in camera_configs.items():
        # 创建随机背景
        frame = np.random.randint(50, 100, (config['size'][1], config['size'][0], 3), dtype=np.uint8)
        
        # 添加一些纹理
        for i in range(0, config['size'][1], 20):
            cv2.line(frame, (0, i), (config['size'][0], i), (80, 80, 80), 1)
        
        frames[cam_name] = frame
    
    return frames, camera_configs

def test_bev_coordinate_transform():
    """测试BEV坐标变换功能"""
    print("=== 测试BEV坐标变换 ===")
    
    # 创建相机参数
    camera_params = create_default_camera_params()
    
    # 创建BEV变换器
    from bev_transform import BEVTransformer
    bev_transformer = BEVTransformer(camera_params, ground_height=0.0)
    
    # 测试图像坐标到地面坐标的变换
    test_image_points = np.array([
        [320, 400],  # 图像中心底部
        [100, 450],  # 左侧底部
        [540, 450],  # 右侧底部
        [320, 300],  # 图像中心中部
    ])
    
    print("测试图像坐标到地面坐标变换:")
    for cam_name in ['front', 'left', 'right', 'rear']:
        print(f"\n{cam_name} 摄像头:")
        ground_points = bev_transformer.image_to_ground(cam_name, test_image_points)
        for i, (img_pt, ground_pt) in enumerate(zip(test_image_points, ground_points)):
            print(f"  点{i}: 图像({img_pt[0]:.1f}, {img_pt[1]:.1f}) -> 地面({ground_pt[0]:.2f}, {ground_pt[1]:.2f}, {ground_pt[2]:.2f})")
    
    # 测试检测框底边中点变换
    print("\n测试检测框底边中点变换:")
    test_detections = np.array([
        [200, 200, 300, 350, 0.85, 2],  # 检测框
        [400, 180, 500, 320, 0.92, 2],  # 检测框
    ])
    
    for cam_name in ['front', 'left']:
        print(f"\n{cam_name} 摄像头:")
        ground_points = bev_transformer.get_bottom_center_ground(cam_name, test_detections)
        for i, (det, ground_pt) in enumerate(zip(test_detections, ground_points)):
            x1, y1, x2, y2 = det[:4]
            bottom_center = ((x1 + x2) / 2, y2)
            print(f"  检测{i}: 底边中点({bottom_center[0]:.1f}, {bottom_center[1]:.1f}) -> 地面({ground_pt[0]:.2f}, {ground_pt[1]:.2f}, {ground_pt[2]:.2f})")

def test_multi_camera_fusion():
    """测试多摄像头融合功能"""
    print("\n=== 测试多摄像头融合 ===")
    
    # 创建ROI演示实例
    cam_names = ['front', 'left', 'right', 'rear']
    demo = ROICanvasDemo(cam_names, img_size=(640, 480))
    
    print(f"BEV融合模式: {'启用' if USE_BEV_FUSION else '禁用'}")
    print(f"摄像头数量: {NUM_CAMERAS}")
    
    # 创建测试帧
    test_frames, camera_configs = create_test_frames_with_detections()
    
    # 显示测试帧
    for cam_name, frame in test_frames.items():
        # 在帧上绘制模拟检测框
        detections = camera_configs[cam_name]['detections']
        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'{conf:.2f}', (int(x1), int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(f'Test Frame: {cam_name}', vis_frame)
    
    # 模拟检测结果
    detections_by_camera = {}
    for cam_name, config in camera_configs.items():
        detections_by_camera[cam_name] = np.array(config['detections'], dtype=np.float32)
    
    print("\n原始检测结果:")
    for cam_name, dets in detections_by_camera.items():
        print(f"  {cam_name}: {dets.shape[0]} 个检测")
    
    # 测试BEV融合
    if USE_BEV_FUSION:
        print("\n执行BEV融合...")
        start_time = time.perf_counter()
        
        fused_results = demo.process_bev_fusion(detections_by_camera, test_frames)
        
        fusion_time = (time.perf_counter() - start_time) * 1000
        print(f"融合处理时间: {fusion_time:.2f} ms")
        
        print("\n融合后检测结果:")
        for cam_name, dets in fused_results.items():
            print(f"  {cam_name}: {dets.shape[0]} 个检测")
        
        # 统计融合效果
        total_before = sum(dets.shape[0] for dets in detections_by_camera.values())
        total_after = sum(dets.shape[0] for dets in fused_results.values())
        print(f"\n融合统计:")
        print(f"  融合前总检测数: {total_before}")
        print(f"  融合后总检测数: {total_after}")
        print(f"  减少数量: {total_before - total_after}")
    
    print("\n按任意键继续...")
    cv2.waitKey(0)

def test_roi_processing_with_bev():
    """测试完整的ROI处理流程（包含BEV融合）"""
    print("\n=== 测试完整ROI处理流程 ===")
    
    # 创建ROI演示实例
    cam_names = ['front', 'left', 'right', 'rear']
    demo = ROICanvasDemo(cam_names, img_size=(640, 480))
    
    # 创建测试帧
    test_frames, _ = create_test_frames_with_detections()
    
    # 模拟多帧处理
    print("模拟多帧处理...")
    for frame_idx in range(5):
        print(f"\n处理第 {frame_idx + 1} 帧:")
        
        # 添加时间变化
        demo.current_timestamp = time.time() + frame_idx * 0.1
        
        # 处理帧
        start_time = time.perf_counter()
        results = demo.step(test_frames)
        process_time = (time.perf_counter() - start_time) * 1000
        
        print(f"  处理时间: {process_time:.2f} ms")
        for cam_name, dets in results.items():
            print(f"  {cam_name}: {dets.shape[0]} 个检测")
        
        # 短暂延迟模拟实时处理
        time.sleep(0.1)
    
    print("\n完整流程测试完成")

def main():
    """主测试函数"""
    print("BEV多摄像头融合测试")
    print("===================")
    
    try:
        # 测试BEV坐标变换
        test_bev_coordinate_transform()
        
        # 测试多摄像头融合
        test_multi_camera_fusion()
        
        # 测试完整ROI处理流程
        test_roi_processing_with_bev()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
