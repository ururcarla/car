import carla

# 连接到 Carla 服务
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 获取车辆蓝图
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]  # 选择 Tesla Model 3

# 获取一个生成点
spawn_point = world.get_map().get_spawn_points()[0]

# 生成车辆
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 启动自动驾驶（可选）
vehicle.set_autopilot(True)

# 防止程序退出（保持车辆在场景中）
while True:
    pass
