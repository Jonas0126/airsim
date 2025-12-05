# airsim_simple_env.py

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirSimSimpleTestEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 1}
    
    # ⚠️ 這裡我們只接收 drone_id
    def __init__(self, drone_id):
        super().__init__()
        self.drone_id = drone_id
        self.target_z = -10.0  # 測試目標高度
        self.step_length = 0.5 # 動作持續時間
        self.steps = 0
        self.max_steps = 10
        
        # 建立專屬的 AirSim 連線 (在每個進程中執行)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # 簡化空間，Action: [0] = Z 軸速度 (升降)
        self.action_space = spaces.Box(low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32)
        # Observation: [0] = Z 軸位置
        self.observation_space = spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float32)

        print(f"[{self.drone_id}] Client connected.")
        self._setup_flight()

    def _setup_flight(self):
        # 每個無人機自己的設置
        self.client.enableApiControl(True, vehicle_name=self.drone_id)
        self.client.armDisarm(True, vehicle_name=self.drone_id)
        # 設置起始位置
        start_pos = self.client.getMultirotorState(vehicle_name=self.drone_id).kinematics_estimated.position
        self.client.simSetVehiclePose(
            airsim.Pose(start_pos, airsim.to_quaternion(0, 0, 0)),
            ignore_collision=True,
            vehicle_name=self.drone_id
        )
        # 強制停止所有移動
        self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.drone_id).join()
        print(f"[{self.drone_id}] Ready at Z={start_pos.z_val:.2f}")

    def _get_obs(self):
        pos = self.client.getMultirotorState(vehicle_name=self.drone_id).kinematics_estimated.position
        return np.array([pos.z_val], dtype=np.float32)

    def step(self, action):
        vz = float(action[0])
        
        # 執行動作：以恆定速度上升/下降
        # ⚠️ 非阻塞指令：沒有 .join()
        self.client.moveByVelocityAsync(
            0, 0, vz, 
            duration=self.step_length, 
            vehicle_name=self.drone_id
        )
        
        # 這裡需要一個極短的延遲，確保指令送出並開始執行
        # 由於沒有 .join()，我們會依賴 AirSim 的快速時鐘更新
        # time.sleep(0.01) # 可選，但在 SubprocVecEnv 中通常不需要

        self.steps += 1
        
        obs = self._get_obs()
        
        # 獎勵：越接近目標高度越好
        reward = -np.abs(obs[0] - self.target_z)
        
        terminated = np.abs(obs[0] - self.target_z) < 0.5  # 接近目標即結束
        truncated = self.steps >= self.max_steps
        info = {}
        if truncated:
            # 必須加入這個鍵，才能讓 SB3 的 TimeLimit Wrapper 正確處理
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._setup_flight()
        return self._get_obs(), {}

    def close(self):
        # 關閉連線 (釋放資源)
        self.client.armDisarm(False, vehicle_name=self.drone_id)
        self.client.enableApiControl(False, vehicle_name=self.drone_id)
        print(f"[{self.drone_id}] Closed.")