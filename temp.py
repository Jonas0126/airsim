import airsim
import time
# 分別 reset
def reset_drone(drone_name, duration):
    print(f"Reset {drone_name} for {duration} sec...")
    
    # 先降落
    client.moveToZAsync(-1, 1, vehicle_name=drone_name)
    time.sleep(0.1)
    # Teleport 回初始位置
    start_pos = airsim.Vector3r(0, 0, -2)
    start_rot = airsim.to_quaternion(0, 0, 0)
    client.simSetVehiclePose(airsim.Pose(start_pos, start_rot),
                             ignore_collision=True,
                             vehicle_name=drone_name)

    
    print(f"{drone_name} reset 完成")

# 連線到 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# drone 名稱
drones = ["Drone1", "Drone2"]

client.reset()
# 啟用 API 控制 & 解鎖
for d in drones:
    client.enableApiControl(True, vehicle_name=d)
    client.armDisarm(True, vehicle_name=d)

num = 0
while 1:
    
    client.moveByAngleRatesThrottleAsync(
        0.0,
        0.0,
        0.0,
        0.7,
        0.3,
        vehicle_name=drones[1]
    )
    time.sleep(0.1)
    client.moveByAngleRatesThrottleAsync(
        0.0,
        0.0,
        0.0,
        0.7,
        0.3,
        vehicle_name=drones[0]
    )
    print(f'num : {num}')
    num += 1
    if num % 55 == 0:
        
        start_pos = airsim.Vector3r(-5, 0, -1)
        start_rot = airsim.to_quaternion(0, 0, 0)
        client.simSetVehiclePose(airsim.Pose(start_pos, start_rot),
                                ignore_collision=True,
                                vehicle_name=drones[1])
        num = 0

    if num % 49 == 0:
        
        start_pos = airsim.Vector3r(0, 0, -1)
        start_rot = airsim.to_quaternion(0, 0, 0)
        client.simSetVehiclePose(airsim.Pose(start_pos, start_rot),
                                ignore_collision=True,
                                vehicle_name=drones[0])
    # time.sleep(0.3)










