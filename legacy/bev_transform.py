"""
BEV (Bird's Eye View) 坐标变换模块
实现轻量级的图像坐标到地面坐标的变换，用于多摄像头联合ROI提取
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2

@dataclass
class CameraParams:
    """相机参数配置"""
    # 内参矩阵 K (3x3)
    K: np.ndarray
    # 外参：旋转矩阵 R (3x3) 和平移向量 t (3x1)
    R: np.ndarray
    t: np.ndarray
    # 图像尺寸
    width: int
    height: int
    # 相机名称
    name: str

class BEVTransformer:
    """BEV坐标变换器"""
    
    def __init__(self, camera_params: Dict[str, CameraParams], ground_height: float = 0.0):
        """
        初始化BEV变换器
        
        Args:
            camera_params: 相机参数字典 {camera_name: CameraParams}
            ground_height: 地面高度（世界坐标系Z=0）
        """
        self.camera_params = camera_params
        self.ground_height = ground_height
        self.homography_matrices = {}
        
        # 计算每个相机的单应性矩阵
        self._compute_homography_matrices()
    
    def _compute_homography_matrices(self):
        """计算每个相机的像面到地面的单应性矩阵"""
        for cam_name, params in self.camera_params.items():
            # 提取旋转矩阵的前两列 r1, r2
            r1 = params.R[:, 0]  # 第一列
            r2 = params.R[:, 1]  # 第二列
            t = params.t.flatten()  # 平移向量
            
            # 构建投影矩阵 P = K * [r1, r2, t]
            # 这里假设地面在世界坐标系中 Z = ground_height
            # 对于 Z = 0 的地面点，投影为 K * [r1, r2, t] * [X, Y, 1]^T
            P = params.K @ np.column_stack([r1, r2, t])
            
            # 单应性矩阵 H = K * [r1, r2, t]
            self.homography_matrices[cam_name] = P
            
            print(f"[BEV] {cam_name} 单应性矩阵计算完成")
            print(f"      图像尺寸: {params.width}x{params.height}")
            print(f"      地面高度: {self.ground_height}")
    
    def image_to_ground(self, cam_name: str, image_points: np.ndarray) -> np.ndarray:
        """
        将图像坐标点投影到地面坐标
        
        Args:
            cam_name: 相机名称
            image_points: 图像坐标点 (N, 2) 或 (N, 3) [u, v] 或 [u, v, 1]
            
        Returns:
            ground_points: 地面坐标点 (N, 3) [X, Y, Z]
        """
        if cam_name not in self.homography_matrices:
            raise ValueError(f"未知的相机名称: {cam_name}")
        
        H = self.homography_matrices[cam_name]
        
        # 确保输入是齐次坐标
        if image_points.shape[1] == 2:
            # 添加齐次坐标
            image_points_homo = np.column_stack([image_points, np.ones(image_points.shape[0])])
        else:
            image_points_homo = image_points
        
        # 投影到地面：ground_points = H^(-1) * image_points
        ground_points_homo = np.linalg.solve(H, image_points_homo.T).T
        
        # 归一化齐次坐标
        ground_points = ground_points_homo[:, :3] / ground_points_homo[:, 2:3]
        
        # 设置地面高度
        ground_points[:, 2] = self.ground_height
        
        return ground_points
    
    def ground_to_image(self, cam_name: str, ground_points: np.ndarray) -> np.ndarray:
        """
        将地面坐标点投影到图像坐标
        
        Args:
            cam_name: 相机名称
            ground_points: 地面坐标点 (N, 3) [X, Y, Z]
            
        Returns:
            image_points: 图像坐标点 (N, 2) [u, v]
        """
        if cam_name not in self.homography_matrices:
            raise ValueError(f"未知的相机名称: {cam_name}")
        
        H = self.homography_matrices[cam_name]
        
        # 确保地面点在Z=0平面
        ground_2d = ground_points[:, :2]  # 只取X, Y
        ground_homo = np.column_stack([ground_2d, np.ones(ground_2d.shape[0])])
        
        # 投影到图像：image_points = H * ground_points
        image_points_homo = (H @ ground_homo.T).T
        
        # 归一化齐次坐标
        image_points = image_points_homo[:, :2] / image_points_homo[:, 2:3]
        
        return image_points
    
    def get_bottom_center_ground(self, cam_name: str, detections: np.ndarray) -> np.ndarray:
        """
        获取检测框底边中点的地面坐标
        
        Args:
            cam_name: 相机名称
            detections: 检测结果 (N, 6) [x1, y1, x2, y2, conf, cls]
            
        Returns:
            ground_points: 底边中点的地面坐标 (N, 3) [X, Y, Z]
        """
        if detections.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # 计算底边中点 (u, v) = ((x1+x2)/2, y2)
        x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
        bottom_centers = np.column_stack([
            (x1 + x2) / 2,  # u坐标
            y2              # v坐标（底边）
        ])
        
        # 投影到地面坐标
        ground_points = self.image_to_ground(cam_name, bottom_centers)
        
        return ground_points

class MultiCameraFusion:
    """多摄像头融合器"""
    
    def __init__(self, bev_transformer: BEVTransformer, 
                 distance_threshold: float = 1.0,
                 time_threshold: float = 0.3):
        """
        初始化多摄像头融合器
        
        Args:
            bev_transformer: BEV变换器
            distance_threshold: 地面距离阈值（米）
            time_threshold: 时间差阈值（秒）
        """
        self.bev_transformer = bev_transformer
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        
        # 存储历史检测结果用于时序关联
        self.detection_history: List[Dict] = []
        self.max_history = 10  # 保留最近10帧的历史
    
    def fuse_detections(self, detections_by_camera: Dict[str, np.ndarray], 
                       timestamp: float) -> Dict[str, np.ndarray]:
        """
        融合多摄像头的检测结果
        
        Args:
            detections_by_camera: 每个相机的检测结果 {camera_name: detections}
            timestamp: 时间戳
            
        Returns:
            fused_detections: 融合后的检测结果 {camera_name: detections}
        """
        if not detections_by_camera:
            return {}
        
        # 1. 将所有检测投影到地面坐标
        ground_detections = []
        camera_detection_map = []
        
        for cam_name, detections in detections_by_camera.items():
            if detections.size == 0:
                continue
            
            # 获取底边中点的地面坐标
            ground_points = self.bev_transformer.get_bottom_center_ground(cam_name, detections)
            
            for i, (detection, ground_point) in enumerate(zip(detections, ground_points)):
                ground_detections.append({
                    'camera': cam_name,
                    'detection': detection.copy(),
                    'ground_point': ground_point,
                    'timestamp': timestamp,
                    'id': f"{cam_name}_{i}_{timestamp}"
                })
                camera_detection_map.append((cam_name, i))
        
        if not ground_detections:
            return {cam: np.empty((0, 6), dtype=np.float32) for cam in detections_by_camera.keys()}
        
        # 2. 基于地面坐标进行聚类/去重
        fused_clusters = self._cluster_ground_detections(ground_detections)
        
        # 3. 时序关联
        self._update_detection_history(ground_detections, timestamp)
        fused_clusters = self._temporal_association(fused_clusters, timestamp)
        
        # 4. 将融合结果映射回各相机
        return self._map_fused_to_cameras(fused_clusters, detections_by_camera.keys())
    
    def _cluster_ground_detections(self, ground_detections: List[Dict]) -> List[Dict]:
        """
        基于地面坐标对检测结果进行聚类
        """
        if not ground_detections:
            return []
        
        clusters = []
        used_detections = set()
        
        for i, det1 in enumerate(ground_detections):
            if i in used_detections:
                continue
            
            # 创建新聚类
            cluster = {
                'detections': [det1],
                'center': det1['ground_point'].copy(),
                'confidence': det1['detection'][4],  # conf
                'best_detection': det1
            }
            used_detections.add(i)
            
            # 寻找相近的检测
            for j, det2 in enumerate(ground_detections[i+1:], i+1):
                if j in used_detections:
                    continue
                
                # 计算地面距离
                distance = np.linalg.norm(det1['ground_point'] - det2['ground_point'])
                time_diff = abs(det1['timestamp'] - det2['timestamp'])
                
                if distance < self.distance_threshold and time_diff < self.time_threshold:
                    cluster['detections'].append(det2)
                    used_detections.add(j)
                    
                    # 更新聚类中心（加权平均）
                    conf1, conf2 = det1['detection'][4], det2['detection'][4]
                    total_conf = conf1 + conf2
                    cluster['center'] = (conf1 * det1['ground_point'] + conf2 * det2['ground_point']) / total_conf
                    
                    # 更新最佳检测（置信度最高的）
                    if conf2 > cluster['confidence']:
                        cluster['confidence'] = conf2
                        cluster['best_detection'] = det2
            
            clusters.append(cluster)
        
        return clusters
    
    def _update_detection_history(self, ground_detections: List[Dict], timestamp: float):
        """更新检测历史"""
        self.detection_history.append({
            'timestamp': timestamp,
            'detections': ground_detections.copy()
        })
        
        # 保持历史长度
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def _temporal_association(self, clusters: List[Dict], timestamp: float) -> List[Dict]:
        """
        时序关联：基于地面坐标的轨迹预测
        """
        if not self.detection_history or len(self.detection_history) < 2:
            return clusters
        
        # 简单的匀速运动预测
        for cluster in clusters:
            current_pos = cluster['center']
            
            # 寻找历史轨迹
            predicted_pos = self._predict_position_from_history(current_pos, timestamp)
            
            if predicted_pos is not None:
                # 更新聚类中心（加权平均）
                cluster['center'] = 0.7 * current_pos + 0.3 * predicted_pos
        
        return clusters
    
    def _predict_position_from_history(self, current_pos: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """
        基于历史轨迹预测位置
        """
        if len(self.detection_history) < 2:
            return None
        
        # 寻找最近的历史检测
        for hist in reversed(self.detection_history[-3:]):  # 只看最近3帧
            for det in hist['detections']:
                distance = np.linalg.norm(current_pos - det['ground_point'])
                if distance < self.distance_threshold * 2:  # 放宽阈值
                    # 简单的匀速预测
                    dt = timestamp - det['timestamp']
                    if dt > 0:
                        # 假设匀速运动，这里简化处理
                        return current_pos  # 实际应用中可以根据速度向量预测
        
        return None
    
    def _map_fused_to_cameras(self, clusters: List[Dict], camera_names: List[str]) -> Dict[str, np.ndarray]:
        """
        将融合结果映射回各相机坐标系
        """
        results = {cam: [] for cam in camera_names}
        
        for cluster in clusters:
            best_detection = cluster['best_detection']
            camera_name = best_detection['camera']
            
            # 将最佳检测分配给对应相机
            results[camera_name].append(best_detection['detection'])
        
        # 转换为numpy数组
        for cam in camera_names:
            if results[cam]:
                results[cam] = np.array(results[cam], dtype=np.float32)
            else:
                results[cam] = np.empty((0, 6), dtype=np.float32)
        
        return results

def create_default_camera_params() -> Dict[str, CameraParams]:
    """
    创建默认的4个摄像头参数配置
    """
    # 假设的相机内参（需要根据实际相机标定调整）
    camera_configs = {
        'front': {
            'K': np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32),
            'R': np.eye(3, dtype=np.float32),  # 假设相机朝向正前方
            't': np.array([[0], [0], [1.5]], dtype=np.float32),  # 相机高度1.5米
            'width': 640,
            'height': 480
        },
        'left': {
            'K': np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32),
            'R': np.array([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 1]], dtype=np.float32),  # 左转90度
            't': np.array([[0.5], [0], [1.5]], dtype=np.float32),  # 左侧偏移
            'width': 640,
            'height': 480
        },
        'right': {
            'K': np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32),
            'R': np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]], dtype=np.float32),  # 右转90度
            't': np.array([[-0.5], [0], [1.5]], dtype=np.float32),  # 右侧偏移
            'width': 640,
            'height': 480
        },
        'rear': {
            'K': np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32),
            'R': np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]], dtype=np.float32),  # 后转180度
            't': np.array([[0], [-1.0], [1.5]], dtype=np.float32),  # 后方偏移
            'width': 640,
            'height': 480
        }
    }
    
    camera_params = {}
    for name, config in camera_configs.items():
        camera_params[name] = CameraParams(
            K=config['K'],
            R=config['R'],
            t=config['t'],
            width=config['width'],
            height=config['height'],
            name=name
        )
    
    return camera_params
