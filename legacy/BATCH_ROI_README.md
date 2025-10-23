# ROI批量处理功能说明

## 功能概述

本更新为ROI识别系统添加了批量处理功能，可以将多个ROI同时调整到固定分辨率（320x320）并进行批量推理，从而提高处理效率。

## 主要特性

### 1. ROI批量处理
- **统一分辨率**: 所有ROI被调整到320x320像素
- **保持长宽比**: 使用letterbox技术保持原始长宽比
- **批量推理**: 同时处理多个ROI，提高GPU利用率

### 2. 配置参数
```python
ROI_BATCH_SIZE = 320            # ROI调整的目标大小
USE_BATCH_PROCESSING = True     # 启用批量处理模式
MAX_BATCH_SIZE = 16             # 单批次最大ROI数量
```

### 3. 处理流程
1. **ROI提取**: 从跟踪器预测的ROI坐标中提取图像区域
2. **尺寸调整**: 将ROI调整到320x320分辨率，保持长宽比
3. **批量推理**: 将多个ROI组成批次进行YOLO推理
4. **坐标映射**: 将推理结果映射回原始帧坐标
5. **跟踪更新**: 更新SORT跟踪器

## 使用方法

### 基本使用
```python
from roi_canvas_demo import ROICanvasDemo

# 创建ROI演示实例
demo = ROICanvasDemo(['front', 'left', 'right'], img_size=(640, 480))

# 处理帧数据
frames = {'front': front_frame, 'left': left_frame, 'right': right_frame}
results = demo.step(frames)
```

### 配置选项
```python
# 启用/禁用批量处理
from roi_canvas_demo import USE_BATCH_PROCESSING
USE_BATCH_PROCESSING = True  # 启用批量处理
USE_BATCH_PROCESSING = False # 使用原始画布处理

# 调整批次大小
from roi_canvas_demo import MAX_BATCH_SIZE
MAX_BATCH_SIZE = 16  # 最大16个ROI一批
```

## 性能优化

### 1. 批量处理优势
- **GPU利用率**: 批量推理比单个推理更高效
- **内存优化**: 统一尺寸减少内存碎片
- **并行处理**: 同时处理多个ROI

### 2. 可视化功能
- **批量ROI显示**: 实时显示当前批次的ROI图像
- **处理时间统计**: 显示推理和总处理时间
- **检测结果**: 显示每个摄像头的检测数量

## 测试功能

### 运行测试
```bash
python test_batch_roi.py
```

测试内容包括：
- ROI调整大小功能测试
- 批量处理性能测试
- 10次重复性能评估

### 性能指标
- 平均处理时间
- 最快/最慢处理时间
- 标准差分析

## 兼容性

### 向后兼容
- 保留原始画布处理方式
- 可通过配置开关切换处理模式
- 不影响现有API接口

### 依赖要求
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- SORT跟踪器

## 注意事项

1. **批次大小**: 根据GPU内存调整MAX_BATCH_SIZE
2. **分辨率**: ROI_BATCH_SIZE影响检测精度和速度
3. **类别过滤**: 只检测道路相关目标类别
4. **坐标映射**: 确保ROI坐标映射的准确性

## 故障排除

### 常见问题
1. **内存不足**: 减少MAX_BATCH_SIZE
2. **检测精度下降**: 增加ROI_BATCH_SIZE
3. **处理速度慢**: 检查GPU使用情况

### 调试模式
```python
from roi_canvas_demo import SHOW_WINDOWS
SHOW_WINDOWS = True  # 启用可视化窗口
```
