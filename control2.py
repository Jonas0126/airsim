import airsim
import time

# 連線到 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone2")
client.armDisarm(True, vehicle_name="Drone2")
num = 0
while 1:
    
    client.moveByAngleRatesThrottleAsync(
        0.0,
        0.0,
        0.0,
        0.7,
        0.3,
        vehicle_name="Drone2"
    )
    state = client.getMultirotorState(vehicle_name="Drone2")
    pos = state.kinematics_estimated.position
    print(f'pos = {[pos.x_val, pos.y_val, pos.z_val]}')
    print(f'num = {num}')
    num += 1
    if num % 55 == 0:
    
        start_pos = airsim.Vector3r(0, 0, -1)
        start_rot = airsim.to_quaternion(0, 0, 0)
        client.simSetVehiclePose(airsim.Pose(start_pos, start_rot),
                                ignore_collision=True,
                                vehicle_name="Drone2")
        num = 0
