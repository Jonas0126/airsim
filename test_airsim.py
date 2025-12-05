import airsim
import time

# 連接到 AirSim
client = airsim.MultirotorClient(port=41452)
client.confirmConnection()

# 取得控制權
client.enableApiControl(True)
client.armDisarm(True)

# 起飛到初始高度 3 米
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()  # AirSim Z軸向下為正，所以向上是負值

# 設定目標高度 (負值表示向上)
target_altitude = -10  # 升到 10 米
ascend_speed = 1  # m/s

print(f"Ascending to {abs(target_altitude)} meters...")

try:
    while True:
        # 取得當前位置
        state = client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val
        print(f'-1.46 - z : {-1.46 - z}')
        # 如果未到目標高度，繼續往上
        if z > target_altitude:
            client.moveByVelocityAsync(0, 0, -ascend_speed, 1)
        else:
            # 到高度後懸停
            client.hoverAsync().join()

        # 打印狀態與 GPS
        pos = state.kinematics_estimated.position
        gps = client.getGpsData().gnss.geo_point
        print(f"Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
        print(f"GPS: lat={gps.latitude}, lon={gps.longitude}, alt={gps.altitude}")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopping...")

# 安全降落
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

