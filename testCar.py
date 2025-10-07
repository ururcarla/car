import time

import carla
import random
import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort
from collections import defaultdict
from roi_canvas_demo import ROICanvasDemo
from ROI import extract_roi

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True  # 开启同步模式
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(2.5)

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 摄像头参数
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')

camera_transforms = {
    'front': carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
    'left': carla.Transform(carla.Location(x=1.0, y=-0.8, z=2.2), carla.Rotation(yaw=-60)),
    'right': carla.Transform(carla.Location(x=1.0, y=0.8, z=2.2), carla.Rotation(yaw=60)),
    'rear': carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180))
}

# 用于保存图像数据
image_buffers, prev_frames = {}, {}

def make_callback(name):
    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        image_buffers[name] = array
    return callback

# 生成摄像头
cameras = {}
for name, transform in camera_transforms.items():
    cam = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    cam.listen(make_callback(name))
    cameras[name] = cam

# 启动自动驾驶
vehicle.set_autopilot(True, traffic_manager.get_port())

vehicle_blueprints = blueprint_library.filter('vehicle.*')
spawn_points = world.get_map().get_spawn_points()

# 防止重复生成在 ego 车位置
spawn_points = [sp for sp in spawn_points if sp.location.distance(vehicle.get_location()) > 8.0]

random.shuffle(spawn_points)
num_npc_vehicles = 30  # 可根据需要设置数量
npc_vehicles = []

for i in range(min(num_npc_vehicles, len(spawn_points))):
    bp = random.choice(vehicle_blueprints)
    transform = spawn_points[i]
    npc = world.try_spawn_actor(bp, transform)
    if npc is not None:
        npc.set_autopilot(True, traffic_manager.get_port())  # 让NPC车自己开
        npc_vehicles.append(npc)

# 给每个摄像头准备一个独立的 SORT 实例
trackers = {name: Sort(max_age=10, min_hits=2, iou_threshold=0.3)
            for name in camera_transforms}

model = YOLO('yolov8m.pt')

# 生成ROI以及合并画布
cam_names = list(cameras.keys())  # ['front', 'left', 'right']
roi_demo = ROICanvasDemo(cam_names, img_size=(640, 480))

roi_demo.model = model
roi_demo.trackers = trackers

# 打印配置信息
from roi_canvas_demo import USE_BATCH_PROCESSING, ROI_BATCH_SIZE, MAX_BATCH_SIZE, USE_BEV_FUSION, NUM_CAMERAS, BEV_DISTANCE_THRESHOLD
print(f"=== 多摄像头ROI处理配置 ===")
print(f"摄像头数量: {NUM_CAMERAS}")
print(f"ROI批量处理模式: {'启用' if USE_BATCH_PROCESSING else '禁用'}")
print(f"ROI批量大小: {ROI_BATCH_SIZE}x{ROI_BATCH_SIZE}")
print(f"最大批次大小: {MAX_BATCH_SIZE}")
print(f"BEV融合模式: {'启用' if USE_BEV_FUSION else '禁用'}")
print(f"BEV距离阈值: {BEV_DISTANCE_THRESHOLD}m")
print("================================")

def process_image(frame, name):
    # 推理计时
    start = time.perf_counter()
    results = model.predict(frame, verbose=False)[0]
    end = time.perf_counter()

    # 显示推理时间
    latency_ms = (end - start) * 1000
    print(f"[Frame {name}] Inference latency: {latency_ms:.2f} ms")

    # 可视化
    annotated = results.plot()
    cv2.imshow(f"YOLOv8 Detection Camera: {name}", annotated)

# def process_ROIs(curr_frame, name):
#     curr_frame = curr_frame.copy()
#     if name not in prev_frames:
#         return
#     prev_frame = prev_frames[name]
#     rois = extract_roi(prev_frame, curr_frame, threshold=50)
#
#     for roi in rois:
#         x, y, w, h = roi['coords']
#         cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     if rois:
#         cv2.imshow(f"ROIs in Camera: {name}", curr_frame)

def process_image_with_sort(frame, name):
    # 1) YOLO 推理
    start = time.perf_counter()
    results = model.predict(frame, verbose=False)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"[{name}] detect latency: {latency_ms:6.1f} ms")

    # 2) 将检测结果整理为 SORT 需要的 ndarray:
    #    [x1, y1, x2, y2, score]  (NumPy float)
    if results.boxes.shape[0] == 0:
        dets = np.empty((0, 5))
    else:
        xyxy   = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        dets   = np.hstack((xyxy, scores))

    # 3) 送入对应摄像头的 tracker
    tracks = trackers[name].update(dets)      # ndarray N×6: (x1,y1,x2,y2,track_id)

    # 4) 画框 & ID
    vis = frame.copy()
    # 画原始检测
    for (x1, y1, x2, y2, s) in dets:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)

    # 画 Track
    for (x1, y1, x2, y2, tid) in tracks:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(vis, f'ID {int(tid)}', (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    track_ms = (time.perf_counter() - start) * 1000
    print(f"[{name}] detect_track latency: {track_ms:6.1f} ms")
    cv2.imshow(f"{name}-track", vis)


try:
    while True:
        world.tick()
        # for name, img in list(image_buffers.items()):
        #     process_image_with_sort(img, name)

        # 收集当前帧（确保每个相机都拿到最新图）
        frames = {}
        for name, img in list(image_buffers.items()):
            frames[name] = img
        # 只有当三路都齐了才跑（你也可以放宽为已有就跑）
        if len(frames) == len(cameras):
            dets_by_cam = roi_demo.step(frames)  # dict[name] -> ndarray

            # 可视化
            for name, frame in frames.items():
                vis = frame.copy()
                dets = dets_by_cam.get(name, np.empty((0, 6), dtype=np.float32))
                if dets.size > 0:
                    # dets: [x1,y1,x2,y2,conf,cls(optional)]
                    for x1, y1, x2, y2, conf, *_ in dets:
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        cv2.putText(vis, f'{conf:.2f}', (int(x1), max(0, int(y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(f'{name}-vis', vis)


        if cv2.waitKey(1) == ord('q'):
            break
finally:
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()
    for npc in npc_vehicles:
        npc.destroy()
    cv2.destroyAllWindows()